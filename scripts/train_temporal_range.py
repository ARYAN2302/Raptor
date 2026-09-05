#!/usr/bin/env python3
import pathlib, json, numpy as np, torch, sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

a2g_root = pathlib.Path("/Users/adarshthakur/Desktop/A2G_Channel_Measurements")
flights = sorted([p for p in a2g_root.iterdir() if p.is_dir()])

class A2GSeqDataset(Dataset):
    def __init__(self, flights, win=1024, seq_len=4):
        self.seq_len=seq_len
        self.samples=[]
        for flight in flights:
            metas=sorted(flight.glob("*.sigmf-meta"))
            # Need consecutive captures
            for i in range(len(metas)-seq_len):
                seq=[]
                gts=[]
                for j in range(seq_len):
                    m=metas[i+j]
                    jj=json.loads(m.read_text())
                    cap=jj["captures"][0]
                    data_path = pathlib.Path(str(m).replace(".sigmf-meta",".sigmf-data"))
                    raw=np.fromfile(str(data_path), dtype=np.float32)
                    if raw.size < win:
                        raw=np.pad(raw, (0, win-raw.size))
                    else:
                        raw=raw[:win]
                    iq=np.stack([raw, np.zeros_like(raw)], axis=-1).astype(np.float32)[:,None,:]
                    tx=cap.get("core:tx_location",{})
                    rx=cap.get("core:rx_location",{})
                    vel=cap.get("core:velocity",{})
                    ecef_tx=latlonalt_to_ecef(tx["latitude"], tx["longitude"], tx["altitude"])
                    enu=ecef_to_enu(ecef_tx, rx["latitude"], rx["longitude"], rx["altitude"])
                    rng, az, el = enu_to_spherical(enu)
                    vel_xyz=np.array([vel.get("velocity_x",0), vel.get("velocity_y",0), vel.get("velocity_z",0)], dtype=np.float32)
                    kvec=enu/np.linalg.norm(enu) if np.linalg.norm(enu)>1e-6 else np.array([0,0,1])
                    radial=float(np.dot(vel_xyz, kvec))
                    seq.append(torch.from_numpy(iq))
                    gts.append((rng, radial, vel_xyz))
                self.samples.append((torch.stack(seq), gts, flight.name))
        print(f"Seq dataset {len(self.samples)} seqs from {len(flights)} flights, seq_len {seq_len}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        seq, gts, flight = self.samples[i]
        # Use last gts as target (predict current)
        rng, radial, vel_xyz = gts[-1]
        return {"iq_seq": seq, "range": float(rng), "radial": float(radial), "vel": vel_xyz.copy(), "flight": flight}

# Split by flights as before
train_flights=flights[:6]
test_flights=flights[6:]
train_ds=A2GSeqDataset(train_flights, win=1024, seq_len=4)
test_ds=A2GSeqDataset(test_flights, win=1024, seq_len=4)
print(f"Train seq {len(train_ds)} Test seq {len(test_ds)}")

# Baselines already computed: range baseline 1.0 MAE, radial 0.274

# Model for temporal
from src.models.iq_tokenizer import ComplexIQTokenizer
from src.models.array_encoder import ArrayEncoder
from src.models.perceiver import PerceiverBottleneck
from src.models.temporal_recurrent import TemporalMamba as TemporalGRU
import torch.nn as nn
class RAPTORTemporal(nn.Module):
    def __init__(self, d_model=32, n_latent=8):
        super().__init__()
        self.tok=ComplexIQTokenizer(patch=8,stride=8,d_model=d_model,max_antennas=1)
        self.arr=ArrayEncoder(d_model=d_model)
        self.perc=PerceiverBottleneck(d_model=d_model,n_latent=n_latent,n_heads=2,n_layers=1)
        self.temp=TemporalGRU(d_model=d_model)
        self.fc_range=nn.Linear(d_model,1)
        self.fc_radial=nn.Linear(d_model,1)
        self.fc_vel=nn.Linear(d_model,3)
    def forward(self, iq_seq):
        # iq_seq [B, T_seq, win, E, 2] -> need to process each time step
        B, S, T, E, C = iq_seq.shape
        state=None
        # Process each window sequentially
        for s in range(S):
            iq=iq_seq[:,s]  # [B,T,E,2]
            ant=torch.zeros(B,E,3, device=iq.device)
            t=self.tok(iq, ant)
            t=self.arr(ant, t)
            z=self.perc(t)
            z, state = self.temp(z, state)
        # Final state is after last window
        feat=z.mean(dim=1)
        return {"range": self.fc_range(feat).squeeze(-1), "radial": self.fc_radial(feat).squeeze(-1), "vel": self.fc_vel(feat)}

# Single-window baseline model
class RAPTORSingle(nn.Module):
    def __init__(self, d_model=32, n_latent=8):
        super().__init__()
        self.tok=ComplexIQTokenizer(patch=8,stride=8,d_model=d_model,max_antennas=1)
        self.arr=ArrayEncoder(d_model=d_model)
        self.perc=PerceiverBottleneck(d_model=d_model,n_latent=n_latent,n_heads=2,n_layers=1)
        self.fc_range=nn.Linear(d_model,1)
        self.fc_radial=nn.Linear(d_model,1)
        self.fc_vel=nn.Linear(d_model,3)
    def forward(self, iq):
        # iq [B,T,E,2]
        B,T,E,C=iq.shape
        ant=torch.zeros(B,E,3, device=iq.device)
        t=self.tok(iq, ant)
        t=self.arr(ant, t)
        z=self.perc(t)
        feat=z.mean(dim=1)
        return {"range": self.fc_range(feat).squeeze(-1), "radial": self.fc_radial(feat).squeeze(-1), "vel": self.fc_vel(feat)}

# Train single
print("\n=== Train Single-window ===")
model_single=RAPTORSingle()
opt=torch.optim.AdamW(model_single.parameters(), lr=0.001)
train_dl=DataLoader(train_ds, batch_size=8, shuffle=True)
test_dl=DataLoader(test_ds, batch_size=8, shuffle=False)
for epoch in range(5):
    model_single.train()
    tot=0
    for b in train_dl:
        iq_seq=b["iq_seq"]  # [B,4,1024,1,2]
        # For single, take last window only
        iq=iq_seq[:,-1]  # [B,1024,1,2]
        rng=torch.tensor(b["range"], dtype=torch.float32)
        radial=torch.tensor(b["radial"], dtype=torch.float32)
        vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
        out=model_single(iq)
        loss = nn.functional.l1_loss(out["range"]/100, rng/100) + nn.functional.l1_loss(out["radial"], radial) + nn.functional.l1_loss(out["vel"], vel)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    print(f"epoch {epoch} single train loss {tot/len(train_dl):.4f}")
    model_single.eval()
    with torch.no_grad():
        pr=[]; gt=[]
        for b in test_dl:
            iq=b["iq_seq"][:,-1]
            rng=torch.tensor(b["range"], dtype=torch.float32)
            out=model_single(iq)
            pr.extend(out["range"].tolist()); gt.extend(rng.tolist())
        import numpy as np
        pr=np.array(pr); gt=np.array(gt)
        print(f"single test range MAE {np.abs(pr-gt).mean():.1f} RMSE {np.sqrt(((pr-gt)**2).mean()):.1f}")

print("\n=== Train Temporal (4 windows) ===")
model_temp=RAPTORTemporal()
opt=torch.optim.AdamW(model_temp.parameters(), lr=0.001)
for epoch in range(5):
    model_temp.train()
    tot=0
    for b in train_dl:
        iq_seq=b["iq_seq"]
        rng=torch.tensor(b["range"], dtype=torch.float32)
        radial=torch.tensor(b["radial"], dtype=torch.float32)
        vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
        out=model_temp(iq_seq)
        loss = nn.functional.l1_loss(out["range"]/100, rng/100) + nn.functional.l1_loss(out["radial"], radial) + nn.functional.l1_loss(out["vel"], vel)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    print(f"epoch {epoch} temporal train loss {tot/len(train_dl):.4f}")
    model_temp.eval()
    with torch.no_grad():
        pr=[]; gt=[]
        for b in test_dl:
            iq_seq=b["iq_seq"]
            rng=torch.tensor(b["range"], dtype=torch.float32)
            out=model_temp(iq_seq)
            pr.extend(out["range"].tolist()); gt.extend(rng.tolist())
        import numpy as np
        pr=np.array(pr); gt=np.array(gt)
        print(f"temporal test range MAE {np.abs(pr-gt).mean():.1f} RMSE {np.sqrt(((pr-gt)**2).mean()):.1f}")

print("temporal vs single done")
# Report temporal context: 4 windows, duration 4*0.4s=1.6s, spacing 0.4s (capture interval)

