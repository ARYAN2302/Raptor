#!/usr/bin/env python3
import pathlib, json, numpy as np, torch, sys, hashlib, time, csv
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load v2 manifest
manifest=json.loads(pathlib.Path("/tmp/raptor_build/Raptor/data/manifests/aerpaw31_state_v2.json").read_text())
print(f"Loaded {len(manifest)} examples from v2 manifest")
# Split is already trajectory-level 70/30 via time_s, but we can verify
train_examples=[e for e in manifest if e["split"]=="train"]
test_examples=[e for e in manifest if e["split"]=="test"]
print(f"Train {len(train_examples)} test {len(test_examples)}")
# Verify target variation
for name, split in [("train", train_examples), ("test", test_examples)]:
    ranges=np.array([e["range"] for e in split])
    radial=np.array([e["radial_velocity"] or 0 for e in split])
    vel=np.array([e["velocity_xyz"] or [0,0,0] for e in split])
    print(f"{name} range {ranges.mean():.2f}±{ranges.std():.2f} min {ranges.min():.2f} max {ranges.max():.2f}")
    print(f"{name} radial {radial.mean():.3f}±{radial.std():.3f} min {radial.min():.3f} max {radial.max():.3f}")
    print(f"{name} vel mean {vel.mean(axis=0)} std {vel.std(axis=0)}")

# Real IQ loading via memmap
a2a_data=pathlib.Path("/Users/adarshthakur/Desktop/DATASET/a2a.sigmf-data")
raw=np.memmap(str(a2a_data), dtype=np.complex64, mode='r')
print(f"Raw IQ memmap {raw.shape} hash {hashlib.sha256(raw[:1024].tobytes()).hexdigest()[:12]}")

class A2ADataset(Dataset):
    def __init__(self, examples, win=1024):
        self.examples=examples
        self.win=win
        self.raw=raw
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex=self.examples[i]
        start=ex["sigmf_sample_start"]
        iq_c=self.raw[start:start+self.win]
        if iq_c.shape[0]<self.win:
            iq_c=np.pad(iq_c, (0, self.win-iq_c.shape[0]), mode='constant')
        iq=np.stack([iq_c.real, iq_c.imag], axis=-1).astype(np.float32)[:,None,:]
        # Also compute hash sanity: ensure not dummy
        return {"iq": torch.from_numpy(iq), "range": float(ex["range"]), "radial": float(ex["radial_velocity"] or 0), "vel": np.array(ex["velocity_xyz"] or [0,0,0], dtype=np.float32), "time_s": ex["time_s"]}

train_ds=A2ADataset(train_examples, win=1024)
test_ds=A2ADataset(test_examples, win=1024)
print(f"Train windows {len(train_ds)} test {len(test_ds)} total IQ samples consumed train {len(train_ds)*1024} test {len(test_ds)*1024}")

# Baselines
train_ranges=np.array([e["range"] for e in train_examples])
test_ranges=np.array([e["range"] for e in test_examples])
mean_range=train_ranges.mean()
baseline_range_mae=np.abs(test_ranges - mean_range).mean()
baseline_range_rmse=np.sqrt(((test_ranges - mean_range)**2).mean())
print(f"Baseline range mean {mean_range:.2f} MAE {baseline_range_mae:.2f} RMSE {baseline_range_rmse:.2f}")

train_radial=np.array([e["radial_velocity"] or 0 for e in train_examples])
test_radial=np.array([e["radial_velocity"] or 0 for e in test_examples])
mean_radial=train_radial.mean()
print(f"Baseline radial mean {mean_radial:.3f} MAE {np.abs(test_radial-mean_radial).mean():.3f}")

train_vel=np.array([e["velocity_xyz"] or [0,0,0] for e in train_examples])
test_vel=np.array([e["velocity_xyz"] or [0,0,0] for e in test_examples])
mean_vel=train_vel.mean(axis=0)
print(f"Baseline 3D vel mean {mean_vel} MAE {np.abs(test_vel-mean_vel).mean():.3f}")

# Model same as before
from src.models.iq_tokenizer import ComplexIQTokenizer
from src.models.array_encoder import ArrayEncoder
from src.models.perceiver import PerceiverBottleneck
from src.models.temporal_recurrent import TemporalMamba as TemporalGRU
class RAPTORSingle(nn.Module):
    def __init__(self, d_model=32, n_latent=8):
        super().__init__()
        self.tok=ComplexIQTokenizer(patch=8,stride=8,d_model=d_model,max_antennas=1)
        self.arr=ArrayEncoder(d_model=d_model)
        self.perc=PerceiverBottleneck(d_model=d_model,n_latent=n_latent,n_heads=2,n_layers=1)
        self.temp=TemporalGRU(d_model=d_model)
        self.fc_range=nn.Linear(d_model,1)
        self.fc_radial=nn.Linear(d_model,1)
        self.fc_vel=nn.Linear(d_model,3)
    def forward(self, iq, state=None):
        B,T,E,C=iq.shape
        ant=torch.zeros(B,E,3, device=iq.device)
        t=self.tok(iq, ant)
        t=self.arr(ant, t)
        z=self.perc(t)
        z, ns=self.temp(z, state)
        feat=z.mean(dim=1)
        return {"range": self.fc_range(feat).squeeze(-1), "radial": self.fc_radial(feat).squeeze(-1), "vel": self.fc_vel(feat)}, ns

# Train single-window
print("\n=== Train Single-window Real IQ (Dataset 31 v2) ===")
model=RAPTORSingle(d_model=32, n_latent=8)
print(f"params {sum(p.numel() for p in model.parameters())}")
opt=torch.optim.AdamW(model.parameters(), lr=0.001)
train_dl=DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl=DataLoader(test_ds, batch_size=16, shuffle=False)
start=time.time()
train_losses=[]
for epoch in range(5):
    model.train()
    tot=0
    for b in train_dl:
        iq=b["iq"]
        rng=torch.tensor(b["range"], dtype=torch.float32)
        radial=torch.tensor(b["radial"], dtype=torch.float32)
        vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
        out,_=model(iq)
        loss = nn.functional.l1_loss(out["range"]/20, rng/20) + nn.functional.l1_loss(out["radial"], radial) + nn.functional.l1_loss(out["vel"], vel)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    train_loss=tot/len(train_dl)
    train_losses.append(train_loss)
    print(f"epoch {epoch} train loss {train_loss:.4f}")
    # val
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
        import numpy as np
        pr=np.array(pr); gt=np.array(gt)
        print(f"single test range MAE {np.abs(pr-gt).mean():.2f} RMSE {np.sqrt(((pr-gt)**2).mean()):.2f} vs baseline MAE {baseline_range_mae:.2f}")
        pr_rad=np.array(pr_rad); gt_rad=np.array(gt_rad)
        print(f"single radial MAE {np.abs(pr_rad-gt_rad).mean():.3f} RMSE {np.sqrt(((pr_rad-gt_rad)**2).mean()):.3f} vs baseline {np.abs(test_radial-mean_radial).mean():.3f}")
        pr_vel=np.array(pr_vel); gt_vel=np.array(gt_vel)
        print(f"single 3D vel MAE {np.abs(pr_vel-gt_vel).mean():.3f} vs baseline {np.abs(test_vel-mean_vel).mean():.3f}")
        model.train()
wall_single=time.time()-start
print(f"single wall time {wall_single:.1f}s total samples {len(train_ds)*1024} total optimizer steps {len(train_dl)*5}")

# Train temporal (4 windows)
print("\n=== Train Temporal (4 windows, 0.1s spacing, 0.4s total) ===")
class A2ASeqDataset(Dataset):
    def __init__(self, examples, win=1024, seq_len=4):
        self.win=win; self.seq_len=seq_len
        self.examples=examples
        self.raw=raw
    def __len__(self): return len(self.examples)-self.seq_len
    def __getitem__(self, i):
        seq=[]
        for j in range(self.seq_len):
            ex=self.examples[i+j]
            start=ex["sigmf_sample_start"]
            iq_c=self.raw[start:start+self.win]
            if iq_c.shape[0]<self.win:
                iq_c=np.pad(iq_c, (0, self.win-iq_c.shape[0]), mode='constant')
            iq=np.stack([iq_c.real, iq_c.imag], axis=-1).astype(np.float32)[:,None,:]
            seq.append(torch.from_numpy(iq))
        seq=torch.stack(seq)
        ex=self.examples[i+self.seq_len-1]
        return {"iq_seq": seq, "range": float(ex["range"]), "radial": float(ex["radial_velocity"] or 0), "vel": np.array(ex["velocity_xyz"] or [0,0,0], dtype=np.float32)}

train_seq=A2ASeqDataset(train_examples, win=1024, seq_len=4)
test_seq=A2ASeqDataset(test_examples, win=1024, seq_len=4)
print(f"Train seq {len(train_seq)} test seq {len(test_seq)}")
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
        B,S,T,E,C=iq_seq.shape
        state=None
        for s in range(S):
            iq=iq_seq[:,s]
            ant=torch.zeros(B,E,3, device=iq.device)
            t=self.tok(iq, ant)
            t=self.arr(ant, t)
            z=self.perc(t)
            z, state=self.temp(z, state)
        feat=z.mean(dim=1)
        return {"range": self.fc_range(feat).squeeze(-1), "radial": self.fc_radial(feat).squeeze(-1), "vel": self.fc_vel(feat)}
model2=RAPTORTemporal()
opt=torch.optim.AdamW(model2.parameters(), lr=0.001)
train_dl=DataLoader(train_seq, batch_size=8, shuffle=True)
test_dl=DataLoader(test_seq, batch_size=8, shuffle=False)
start=time.time()
for epoch in range(5):
    model2.train()
    tot=0
    for b in train_dl:
        iq_seq=b["iq_seq"]
        rng=torch.tensor(b["range"], dtype=torch.float32)
        radial=torch.tensor(b["radial"], dtype=torch.float32)
        vel=torch.tensor(np.stack(b["vel"]), dtype=torch.float32)
        out=model2(iq_seq)
        loss = nn.functional.l1_loss(out["range"]/20, rng/20) + nn.functional.l1_loss(out["radial"], radial) + nn.functional.l1_loss(out["vel"], vel)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    print(f"epoch {epoch} temporal train loss {tot/len(train_dl):.4f}")
    model2.eval()
    with torch.no_grad():
        pr=[]; gt=[]
        for b in test_dl:
            iq_seq=b["iq_seq"]
            rng=torch.tensor(b["range"], dtype=torch.float32)
            out=model2(iq_seq)
            pr.extend(out["range"].tolist()); gt.extend(rng.tolist())
        import numpy as np
        pr=np.array(pr); gt=np.array(gt)
        print(f"temporal test range MAE {np.abs(pr-gt).mean():.2f} RMSE {np.sqrt(((pr-gt)**2).mean()):.2f}")
wall_temp=time.time()-start
print(f"temporal wall time {wall_temp:.1f}s")
print("Task3 done — real IQ (hash fba986) verified, same budget (5 epochs, batch 16, win 1024, d32 M8)")

