#!/usr/bin/env python3
import sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
import torch, yaml, pathlib
from src.models.raptor import RAPTOR
from src.datasets.synthetic import SyntheticIQDataset
from torch.utils.data import DataLoader
# use small config for smoke (same code path)
cfg={"model":{"antennas":1,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4}}
model=RAPTOR(cfg)
print("params", sum(p.numel() for p in model.parameters()))
ds=SyntheticIQDataset(n=16,T=512,E=1,max_emitters=1)
dl=DataLoader(ds,batch_size=4,shuffle=True, collate_fn=lambda b: {"iq": torch.stack([x["iq"] for x in b])})
for batch in dl:
    out,_=model(batch["iq"])
    print("out keys", list(out.keys()), "exist", out["existence"].shape, "range", out["range"].shape)
    break
print("full forward ok — same code small or full (full uses raptor_full.yaml with antennas 4 when E=4)")
