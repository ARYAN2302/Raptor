#!/usr/bin/env python3
"""Task1: Train RANGE+VELOCITY on real Dataset 19, single emitter, trajectory splits, no Hungarian."""
import pathlib, json, numpy as np, torch, sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.models.raptor import RAPTOR
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Dataset 19 loader
a2g_root = pathlib.Path("/Users/adarshthakur/Desktop/A2G_Channel_Measurements")
flights = sorted([p for p in a2g_root.iterdir() if p.is_dir()])

class A2GDataset(Dataset):
    def __init__(self, flights, win=1024):
        self.samples=[]
        for flight in flights:
            metas=sorted(flight.glob("*.sigmf-meta"))
            for m in metas[:50]:  # limit for smoke
                j=json.loads(m.read_text())
                cap=j["captures"][0] if j.get("captures") else {}
                # IQ
                data_path = pathlib.Path(str(m).replace(".sigmf-meta",".sigmf-data"))
                if not data_path.exists(): continue
                raw=np.fromfile(str(data_path), dtype=np.float32)
                # f32, single channel, need to make [T,1,2]
                # Pad/truncate to win
                if raw.size < win:
                    raw=np.pad(raw, (0, win-raw.size))
                else:
                    raw=raw[:win]
                # Make complex: I=raw, Q=0
                iq=np.stack([raw, np.zeros_like(raw)], axis=-1).astype(np.float32)[:,None,:]  # [win,1,2]
                # Ground truth
                tx=cap.get("core:tx_location",{})
                rx=cap.get("core:rx_location",{})
                vel=cap.get("core:velocity",{})
                if not tx or not rx: continue
                # Compute range/az/el
                ecef_tx=latlonalt_to_ecef(tx["latitude"], tx["longitude"], tx["altitude"])
                enu=ecef_to_enu(ecef_tx, rx["latitude"], rx["longitude"], rx["altitude"])
                rng, az, el = enu_to_spherical(enu)
                vel_xyz=np.array([vel.get("velocity_x",0), vel.get("velocity_y",0), vel.get("velocity_z",0)], dtype=np.float32)
                kvec=enu/np.linalg.norm(enu) if np.linalg.norm(enu)>1e-6 else np.array([0,0,1])
                radial=float(np.dot(vel_xyz, kvec))
                self.samples.append((iq, rng, radial, vel_xyz, flight.name))
        print(f"Loaded {len(self.samples)} windows from {len(flights)} flights")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        iq, rng, radial, vel_xyz, flight = self.samples[i]
        return {"iq": torch.from_numpy(iq), "range": float(rng), "radial": float(radial), "vel": vel_xyz.copy(), "flight": flight}

# Trajectory-level split: train flights 0-5, test flights 6-8
train_flights=flights[:6]
test_flights=flights[6:]

train_ds=A2GDataset(train_flights, win=1024)
test_ds=A2GDataset(test_flights, win=1024)
print(f"Train flights {[p.name for p in train_flights]} windows {len(train_ds)}")
print(f"Test flights {[p.name for p in test_flights]} windows {len(test_ds)}")

# Model: single emitter, no Hungarian, direct head from latents mean
class SingleHead(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.fc_range=nn.Linear(d_model,1)
        self.fc_radial=nn.Linear(d_model,1)
        self.fc_vel=nn.Linear(d_model,3)
        self.fc_logvar=nn.Linear(d_model,3)  # range, radial, vel
    def forward(self, latents):
        # latents [B,M,D] -> mean pool
        feat=latents.mean(dim=1)  # [B,D]
        return {
            "range": self.fc_range(feat).squeeze(-1),
            "radial": self.fc_radial(feat).squeeze(-1),
            "vel": self.fc_vel(feat),
            "logvar": self.fc_logvar(feat)
        }

# Full RAPTOR without set decoder for single emitter
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.models.iq_tokenizer import ComplexIQTokenizer
from src.models.array_encoder import ArrayEncoder
from src.models.perceiver import PerceiverBottleneck
from src.models.temporal_recurrent import TemporalMamba as TemporalGRU
import torch.nn as nn
class RAPTORSingle(nn.Module):
    def __init__(self, d_model=32, n_latent=8, temporal="gru"):
        super().__init__()
        self.tok=ComplexIQTokenizer(patch=8,stride=8,d_model=d_model,max_antennas=1)
        self.arr=ArrayEncoder(d_model=d_model)
        self.perc=PerceiverBottleneck(d_model=d_model,n_latent=n_latent,n_heads=2,n_layers=1)
        if temporal=="gru":
            self.temp=TemporalGRU(d_model=d_model)
        else:
            from src.models.temporal_mamba_ssm import TemporalMambaSSM
            self.temp=TemporalMambaSSM(d_model=d_model)
        self.head=SingleHead(d_model=d_model)
    def forward(self, iq, state=None):
        # iq [B,T,E,2]
        B,T,E,C=iq.shape
        ant=torch.zeros(B,E,3, device=iq.device)
        t=self.tok(iq, ant)
        t=self.arr(ant, t)
        z=self.perc(t)
        z, ns=self.temp(z, state)
        out=self.head(z)
        out["latents"]=z
        return out, ns

# Train
cfg_single="gru"
model=RAPTORSingle(d_model=32, n_latent=8, temporal=cfg_single)
print(f"params {sum(p.numel() for p in model.parameters())}")
opt=torch.optim.AdamW(model.parameters(), lr=0.001)
train_dl=DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl=DataLoader(test_ds, batch_size=16, shuffle=False)

for epoch in range(3):
    model.train()
    tot=0
    for b in train_dl:
        iq=b["iq"]
        rng=torch.tensor(b["range"], dtype=torch.float32)
        radial=torch.tensor(b["radial"], dtype=torch.float32)
        vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
        out,_=model(iq)
        loss = nn.functional.l1_loss(out["range"], rng) * 0.001 + nn.functional.l1_loss(out["radial"], radial) * 0.1 + nn.functional.l1_loss(out["vel"], vel) * 0.1
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    print(f"epoch {epoch} train loss {tot/len(train_dl):.4f}")
    # Test
    model.eval()
    with torch.no_grad():
        pr=[]; gt=[]
        pr_rad=[]; gt_rad=[]
        pr_vel=[]; gt_vel=[]
        for b in test_dl:
            iq=b["iq"]
            rng=torch.tensor(b["range"], dtype=torch.float32)
            radial=torch.tensor(b["radial"], dtype=torch.float32)
            vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
            out,_=model(iq)
            pr.extend(out["range"].tolist()); gt.extend(rng.tolist())
            pr_rad.extend(out["radial"].tolist()); gt_rad.extend(radial.tolist())
            pr_vel.extend(out["vel"].tolist()); gt_vel.extend(vel.tolist())
        # Compute MAE/RMSE
        import numpy as np
        pr=np.array(pr); gt=np.array(gt)
        print(f"test range MAE {np.abs(pr-gt).mean():.1f} RMSE {np.sqrt(((pr-gt)**2).mean()):.1f} (gt mean {gt.mean():.1f})")
        pr_rad=np.array(pr_rad); gt_rad=np.array(gt_rad)
        print(f"test radial MAE {np.abs(pr_rad-gt_rad).mean():.2f} RMSE {np.sqrt(((pr_rad-gt_rad)**2).mean()):.2f}")
        pr_vel=np.array(pr_vel); gt_vel=np.array(gt_vel)
        print(f"test vel MAE {np.abs(pr_vel-gt_vel).mean():.2f} RMSE {np.sqrt(((pr_vel-gt_vel)**2).mean()):.2f}")

print("Task1 smoke done")

