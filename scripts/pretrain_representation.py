#!/usr/bin/env python3
"""Stage B §10 — masked recon, small slice §15, same full RAPTOR path."""
import sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
import torch, yaml
from src.models.raptor import RAPTOR
from src.datasets.synthetic import SyntheticIQDataset
from torch.utils.data import DataLoader
cfg={"model":{"antennas":1,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4}}
model=RAPTOR(cfg)
opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
ds=SyntheticIQDataset(n=32,T=512,E=1,max_emitters=1)
dl=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=lambda b: {"iq": torch.stack([x["iq"] for x in b])})
for epoch in range(1):
    for bi,b in enumerate(dl):
        out=model.forward_recon(b["iq"], mask_ratio=0.6)
        loss=(out["tokens"]**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"pretrain epoch {epoch} batch {bi} loss {loss.item():.4f}")
        break
print("pretrain_representation done")
