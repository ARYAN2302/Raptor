#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import yaml, torch
from src.models.raptor import RAPTOR
from src.datasets.synthetic import SyntheticIQDataset
from torch.utils.data import DataLoader

def collate(batch):
    import torch
    return {"iq": torch.stack([b["iq"] for b in batch])}

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/pretrain.yaml")
    p.add_argument("--dry", action="store_true")
    a=p.parse_args()
    cfg=yaml.safe_load(open(a.config))
    print(cfg)
    model=RAPTOR(cfg)
    n=cfg.get("n_synth",128)
    ds=SyntheticIQDataset(n=n, T=cfg.get("window",4096), E=4, max_emitters=1)
    dl=DataLoader(ds, batch_size=cfg.get("batch_size",8), shuffle=True, collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",0.0003))
    for epoch in range(1 if a.dry else cfg.get("epochs",2)):
        for batch in dl:
            iq=batch["iq"]
            out=model.forward_recon(iq, mask_ratio=cfg.get("mask_ratio",0.4))
            loss=(out["tokens"]**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"epoch {epoch} loss {loss.item():.4f}")
            break
        if a.dry: break
    print("pretrain dry ok")

if __name__=="__main__": main()
