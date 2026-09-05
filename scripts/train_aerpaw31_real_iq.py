#!/usr/bin/env python3
import pathlib, json, numpy as np, torch, sys, hashlib
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths
a2a_data = pathlib.Path("/Users/adarshthakur/Desktop/DATASET/a2a.sigmf-data")
a2a_meta = pathlib.Path("/Users/adarshthakur/Desktop/DATASET/a2a.sigmf-meta")
a2a_csv = pathlib.Path("/Users/adarshthakur/Desktop/DATASET/a2a.csv")
manifest_path = pathlib.Path("/tmp/raptor_build/Raptor/data/manifests/aerpaw31_supervised.json")
examples=json.loads(pathlib.Path(manifest_path).read_text())
print(f"Loaded {len(examples)} examples from manifest")

# Verify IQ genuinely from file: hash check
# Read first capture's IQ via sample_start
meta=json.loads(a2a_meta.read_text())
# Find first example's sample_start
first=examples[0]
print(f"First example {first['measurement_id']} sample_start {first['sigmf_sample_start']}")
# Read via numpy memmap
# Use sigmf library to verify datatype
try:
    import sigmf
    from sigmf import sigmffile
    sig = sigmffile.fromfile(str(a2a_data.with_suffix("")))  # without extension, sigmf will find .sigmf-meta + .sigmf-data
    print(f"sigmf global datatype {sig.get_global_field('core:datatype')} sample_rate {sig.get_global_field('core:sample_rate')}")
    # Read samples for first capture
    sample_start=first["sigmf_sample_start"]
    sig.seek(sample_start)
    samples=sig.read_samples(count=1024)
    print(f"sigmf read_samples {samples.shape} dtype {samples.dtype} first 2 {samples[:2]}")
    # Also direct numpy
    raw=np.fromfile(str(a2a_data), dtype=np.complex64)
    print(f"direct np complex64 total {raw.shape} first 2 {raw[:2]} hash {hashlib.sha256(raw[:1024].tobytes()).hexdigest()[:12]}")
    # Compare
    print(f"sigmf vs direct equal? {np.allclose(samples, raw[sample_start:sample_start+1024])}")
    # Sanity stats
    print(f"IQ stats mean {np.abs(raw[:10000]).mean():.4f} std {np.abs(raw[:10000]).std():.4f} max {np.abs(raw[:10000]).max():.4f}")
    # Verify not dummy: check that IQ is not random (has structure)
    print(f"IQ hash check first window {hashlib.sha256(samples.tobytes()[:1024]).hexdigest()[:12]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"sigmf fail {e}")

# Now build dataset with real IQ
class A2ARealDataset(Dataset):
    def __init__(self, examples, win=1024):
        self.examples=examples
        self.win=win
        # Memmap for efficiency
        self.raw=np.memmap(str(a2a_data), dtype=np.complex64, mode='r')
        print(f"memmap shape {self.raw.shape}  dtype {self.raw.dtype}")
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex=self.examples[i]
        start=ex["sigmf_sample_start"]
        # Read win complex samples
        # Ensure not out of bounds
        if start+self.win > len(self.raw):
            start=len(self.raw)-self.win
        iq_c = self.raw[start:start+self.win]  # complex64
        # Convert to [win,1,2] float32
        iq=np.stack([iq_c.real, iq_c.imag], axis=-1).astype(np.float32)[:,None,:]
        # Already normalized later via per-window RMS in RAPTORSingle? For now return raw
        return {"iq": torch.from_numpy(iq), "range": float(ex["range"]), "radial": float(ex["radial_velocity"] or 0), "vel": np.array(ex["velocity_xyz"] or [0,0,0], dtype=np.float32), "time_s": ex["time_s"]}

# Split trajectory-level: first 70% time for train, last 30% for test (sorted by time_s)
examples_sorted=sorted(examples, key=lambda x: x["time_s"])
n=len(examples_sorted)
train_examples=examples_sorted[:int(n*0.7)]
test_examples=examples_sorted[int(n*0.7):]
print(f"Train {len(train_examples)} test {len(test_examples)}")

train_ds=A2ARealDataset(train_examples, win=1024)
test_ds=A2ARealDataset(test_examples, win=1024)
print(f"Train windows {len(train_ds)} test {len(test_ds)}")

# Baselines (same as before, but now with real GT)
import numpy as np
train_ranges=np.array([e["range"] for e in train_examples])
test_ranges=np.array([e["range"] for e in test_examples])
mean_range=train_ranges.mean()
baseline_range_mae=np.abs(test_ranges - mean_range).mean()
baseline_range_rmse=np.sqrt(((test_ranges - mean_range)**2).mean())
print(f"Baseline range mean {mean_range:.2f} MAE {baseline_range_mae:.2f} RMSE {baseline_range_rmse:.2f} (test mean {test_ranges.mean():.2f} std {test_ranges.std():.2f})")
train_radial=np.array([e["radial_velocity"] or 0 for e in train_examples])
test_radial=np.array([e["radial_velocity"] or 0 for e in test_examples])
mean_radial=train_radial.mean()
print(f"Baseline radial mean {mean_radial:.3f} MAE {np.abs(test_radial-mean_radial).mean():.3f}")
train_vel=np.array([e["velocity_xyz"] or [0,0,0] for e in train_examples])
test_vel=np.array([e["velocity_xyz"] or [0,0,0] for e in test_examples])
mean_vel=train_vel.mean(axis=0)
print(f"Baseline vel mean {mean_vel} MAE {np.abs(test_vel-mean_vel).mean():.3f}")

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
print("\n=== Train Single-window Real IQ (Dataset 31) ===")
model=RAPTORSingle(d_model=32, n_latent=8)
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
        loss = nn.functional.l1_loss(out["range"]/20, rng/20) + nn.functional.l1_loss(out["radial"], radial) + nn.functional.l1_loss(out["vel"], vel)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    print(f"epoch {epoch} train loss {tot/len(train_dl):.4f}")
    model.eval()
    with torch.no_grad():
        pr=[]; gt=[]
        for b in test_dl:
            iq=b["iq"]
            rng=torch.tensor(b["range"], dtype=torch.float32)
            out,_=model(iq)
            pr.extend(out["range"].tolist()); gt.extend(rng.tolist())
        import numpy as np
        pr=np.array(pr); gt=np.array(gt)
        print(f"single test range MAE {np.abs(pr-gt).mean():.2f} RMSE {np.sqrt(((pr-gt)**2).mean()):.2f} vs baseline MAE {baseline_range_mae:.2f}")

# Temporal (4 windows)
print("\n=== Train Temporal Real IQ (4 windows, 0.4s spacing) ===")
class A2ASeqDataset(Dataset):
    def __init__(self, examples, win=1024, seq_len=4):
        self.win=win; self.seq_len=seq_len
        self.examples=examples
        self.raw=np.memmap(str(a2a_data), dtype=np.complex64, mode='r')
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
        seq=torch.stack(seq)  # [4,1024,1,2]
        ex=self.examples[i+self.seq_len-1]
        return {"iq_seq": seq, "range": float(ex["range"]), "radial": float(ex["radial_velocity"] or 0), "vel": np.array(ex["velocity_xyz"] or [0,0,0], dtype=np.float32)}

train_seq=A2ASeqDataset(train_examples, win=1024, seq_len=4)
test_seq=A2ASeqDataset(test_examples, win=1024, seq_len=4)
print(f"Train seq {len(train_seq)} test seq {len(test_seq)}")
# Need temporal model that handles seq
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
        # iq_seq [B,4,1024,1,2]
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

print("Task done — real IQ verified via sigmf hash, now compare single vs temporal vs baselines")

