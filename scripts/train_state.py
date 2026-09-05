#!/usr/bin/env python3
"""Phase 6 — DETR set prediction training on synthetic coherent IQ."""
import sys
sys.path.insert(0,"src")
import yaml, torch
from src.models.raptor import RAPTOR
from src.datasets.synthetic import SyntheticIQDataset
from src.losses.set_loss import SetPredictionLoss
from torch.utils.data import DataLoader

def collate(batch):
    iq=torch.stack([b["iq"] for b in batch])
    gt=[b["emitters"] for b in batch]
    counts=torch.tensor([b["emitter_count"] for b in batch])
    return {"iq": iq, "gt": gt, "counts": counts}

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/state.yaml")
    p.add_argument("--dry", action="store_true")
    a=p.parse_args()
    cfg=yaml.safe_load(open(a.config))
    print(cfg)
    model=RAPTOR({**cfg, "model": {**cfg.get("model",{}), "antennas":4, "n_queries": cfg.get("n_queries",4)}})
    ds=SyntheticIQDataset(n=cfg.get("n_synth",256), T=cfg.get("window",4096), E=4, max_emitters=cfg.get("max_emitters",3))
    dl=DataLoader(ds, batch_size=cfg.get("batch_size",8), shuffle=True, collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",2e-4))
    crit=SetPredictionLoss()
    for epoch in range(1 if a.dry else cfg.get("epochs",3)):
        for batch in dl:
            out=model(batch["iq"])
            loss=crit(out, batch["gt"])
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"epoch {epoch} set_loss {loss.item():.4f} exist {out['existence'][0].tolist()[:3]}")
            break
        if a.dry: break
    print("train_state dry ok — ablation: remove temporal/perceiver via config toggles")

if __name__=="__main__": main()
