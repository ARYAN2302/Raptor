#!/usr/bin/env python3
import pathlib, json, numpy as np, torch, sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

a2g_root = pathlib.Path("/Users/adarshthakur/Desktop/A2G_Channel_Measurements")
flights = sorted([p for p in a2g_root.iterdir() if p.is_dir()])

class A2GDataset(Dataset):
    def __init__(self, flights, win=1024):
        self.samples=[]
        for flight in flights:
            metas=sorted(flight.glob("*.sigmf-meta"))
            for m in metas[:100]:  # more
                j=json.loads(m.read_text())
                cap=j["captures"][0] if j.get("captures") else {}
                data_path = pathlib.Path(str(m).replace(".sigmf-meta",".sigmf-data"))
                if not data_path.exists(): continue
                raw=np.fromfile(str(data_path), dtype=np.float32)
                if raw.size < win:
                    raw=np.pad(raw, (0, win-raw.size))
                else:
                    raw=raw[:win]
                iq=np.stack([raw, np.zeros_like(raw)], axis=-1).astype(np.float32)[:,None,:]
                tx=cap.get("core:tx_location",{})
                rx=cap.get("core:rx_location",{})
                vel=cap.get("core:velocity",{})
                if not tx or not rx: continue
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

train_flights=flights[:6]
test_flights=flights[6:]
train_ds=A2GDataset(train_flights, win=1024)
test_ds=A2GDataset(test_flights, win=1024)
print(f"Train flights {[p.name for p in train_flights]} windows {len(train_ds)}")
print(f"Test flights {[p.name for p in test_flights]} windows {len(test_ds)}")

# Check GT distribution
import numpy as np
train_ranges=np.array([s[1] for s in train_ds.samples])
test_ranges=np.array([s[1] for s in test_ds.samples])
print(f"Train range mean {train_ranges.mean():.1f} std {train_ranges.std():.1f} min {train_ranges.min():.1f} max {train_ranges.max():.1f}")
print(f"Test range mean {test_ranges.mean():.1f} std {test_ranges.std():.1f} min {test_ranges.min():.1f} max {test_ranges.max():.1f}")

# Baselines Task2
# Range constant mean
mean_range=train_ranges.mean()
baseline_range_mae=np.abs(test_ranges - mean_range).mean()
baseline_range_rmse=np.sqrt(((test_ranges - mean_range)**2).mean())
print(f"Baseline range constant mean {mean_range:.1f} MAE {baseline_range_mae:.1f} RMSE {baseline_range_rmse:.1f}")

# Velocity baselines
train_vel=np.array([s[3] for s in train_ds.samples])
test_vel=np.array([s[3] for s in test_ds.samples])
mean_vel=train_vel.mean(axis=0)
baseline_vel_mae=np.abs(test_vel - mean_vel).mean()
print(f"Baseline velocity mean {mean_vel} MAE {baseline_vel_mae:.3f}")
train_radial=np.array([s[2] for s in train_ds.samples])
test_radial=np.array([s[2] for s in test_ds.samples])
mean_radial=train_radial.mean()
print(f"Baseline radial mean {mean_radial:.3f} MAE {np.abs(test_radial-mean_radial).mean():.3f}")

# Model
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
        self.fc_range=nn.Linear(d_model,1)
        self.fc_radial=nn.Linear(d_model,1)
        self.fc_vel=nn.Linear(d_model,3)
        self.fc_logvar=nn.Linear(d_model,3)
    def forward(self, iq, state=None):
        B,T,E,C=iq.shape
        ant=torch.zeros(B,E,3, device=iq.device)
        t=self.tok(iq, ant)
        t=self.arr(ant, t)
        z=self.perc(t)
        z, ns=self.temp(z, state)
        feat=z.mean(dim=1)
        return {"range": self.fc_range(feat).squeeze(-1), "radial": self.fc_radial(feat).squeeze(-1), "vel": self.fc_vel(feat), "logvar": self.fc_logvar(feat), "latents": z}, ns

# Train single-window
for temporal in ["gru"]:
    print(f"\n=== Train temporal={temporal} ===")
    model=RAPTORSingle(d_model=32, n_latent=8, temporal=temporal)
    opt=torch.optim.AdamW(model.parameters(), lr=0.001)
    train_dl=DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl=DataLoader(test_ds, batch_size=16, shuffle=False)
    for epoch in range(5):
        model.train()
        tot=0
        for b in train_dl:
            iq=b["iq"]
            rng=torch.tensor(b["range"], dtype=torch.float32)
            radial=torch.tensor(b["radial"], dtype=torch.float32)
            vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
            out,_=model(iq)
            # Use normalized range: divide by 100 for loss scaling
            loss = nn.functional.l1_loss(out["range"]/100, rng/100) + nn.functional.l1_loss(out["radial"], radial) + nn.functional.l1_loss(out["vel"], vel)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"epoch {epoch} train loss {tot/len(train_dl):.4f}")
        model.eval()
        with torch.no_grad():
            pr=[]; gt=[]
            pr_rad=[]; gt_rad=[]
            for b in test_dl:
                iq=b["iq"]
                rng=torch.tensor(b["range"], dtype=torch.float32)
                radial=torch.tensor(b["radial"], dtype=torch.float32)
                out,_=model(iq)
                pr.extend(out["range"].tolist()); gt.extend(rng.tolist())
                pr_rad.extend(out["radial"].tolist()); gt_rad.extend(radial.tolist())
            import numpy as np
            pr=np.array(pr); gt=np.array(gt)
            print(f"test range MAE {np.abs(pr-gt).mean():.1f} RMSE {np.sqrt(((pr-gt)**2).mean()):.1f} vs baseline MAE {baseline_range_mae:.1f} RMSE {baseline_range_rmse:.1f}")
            pr_rad=np.array(pr_rad); gt_rad=np.array(gt_rad)
            print(f"test radial MAE {np.abs(pr_rad-gt_rad).mean():.3f} RMSE {np.sqrt(((pr_rad-gt_rad)**2).mean()):.3f} vs baseline {np.abs(test_radial-mean_radial).mean():.3f}")

