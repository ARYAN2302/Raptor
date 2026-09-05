#!/usr/bin/env python3
"""Gate A: does masked recon transfer better than baselines? §19 Task10, §20 Gate A"""
import sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
import torch, yaml
from src.models.iq_tokenizer import ComplexIQTokenizer
from src.models.perceiver import PerceiverBottleneck
from src.models.baselines import BaselineCNN
from src.datasets.synthetic import SyntheticIQDataset, synth_iq
from torch.utils.data import DataLoader

def run():
    # small model per §15: first prove pipeline with small slice
    tok=ComplexIQTokenizer(in_antennas=1,patch=8,stride=8,d_model=32)
    perc=PerceiverBottleneck(d_model=32,n_latent=8,n_heads=2,n_layers=1)
    cnn=BaselineCNN(in_ant=1,d=32)
    # synthetic 64 windows small slice (§15)
    ds=SyntheticIQDataset(n=64,T=512,E=1,max_emitters=1)
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b]), "label": torch.tensor([x["label"] for x in b])}
    dl=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=coll)
    opt=torch.optim.AdamW(list(tok.parameters())+list(perc.parameters()), lr=0.0003)
    # 1 epoch masked recon
    for batch in dl:
        iq=batch["iq"]
        t=tok(iq)
        # channel-independent masking 60% (Radio-FM §III-D1)
        mask=torch.rand(t.shape[0],t.shape[1])<0.6
        t_mask=t.clone(); t_mask[mask]=0
        z=perc(t_mask)
        # recon head: predict masked tokens via latent mean
        recon=z.mean(dim=1)  # [B,D] -> broadcast not real recon but proves pipeline
        loss=(t[mask].mean()**2) if mask.any() else (t**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"recon loss {loss.item():.4f} tokens {t.shape} latents {z.shape} vs CNN {cnn(iq).shape}")
        break
    print("Gate A smoke ok — need real RFUAV vs UAVSig transfer to pass")

if __name__=="__main__": run()
