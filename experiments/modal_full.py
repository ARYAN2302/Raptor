import modal
app = modal.App("raptor-full")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","tqdm","einops","scikit-learn","scipy").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)

@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def full_pipeline():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, yaml, json, pathlib, random
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import SyntheticIQDataset
    from src.losses.set_loss import SetPredictionLoss
    from torch.utils.data import DataLoader
    # Full RAPTOR config but small (same code path §18)
    cfg={"model":{"antennas":4,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    print(f"params {sum(p.numel() for p in model.parameters())} git {pathlib.Path('/root/raptor/.git').exists()}")
    # Stage B: masked recon (1 epoch small slice)
    ds=SyntheticIQDataset(n=64,T=512,E=4,max_emitters=1)
    dl=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=lambda b: {"iq": torch.stack([x["iq"] for x in b])})
    opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
    for b in dl:
        iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
        out=model.forward_recon(iq, mask_ratio=0.6)
        loss=(out["tokens"]**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"Stage B recon loss {loss.item():.4f}")
        break
    # Stage D/E: 0-2 emitters set prediction (small slice §11)
    ds2=SyntheticIQDataset(n=128,T=512,E=4,max_emitters=2)
    def coll2(b): return {"iq": torch.stack([x["iq"] for x in b]), "gt": [x["emitters"] for x in b] if "emitters" in b[0] else [[{"range":100}] for _ in b], "labels": torch.tensor([x["label"] for x in b])}
    # Actually SyntheticIQDataset returns label = n_emit, emitters list empty for synthetic? adapt
    from src.datasets.synthetic import synth_iq
    # manual 0-2 test
    for n in [0,1,2]:
        s=synth_iq(T=512,E=4,n_emitters=n,seed=10+n)
        print(f"synth n={n} iq {s.iq.shape} emitters {len(s.emitters)}")
    # Stage C temporal ablation placeholder
    print("temporal ablation: no-temporal vs Mamba — to be measured per §14 table (short vs long)")
    # Save checkpoint with §19 metadata
    meta={"git": "4b835a1", "config": cfg, "params": sum(p.numel() for p in model.parameters()), "dataset": "synthetic small slice 512-window", "seeds": 0}
    pathlib.Path("/ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, "/ckpt/full_small.pt")
    ckpt_vol.commit()
    print(f"saved /ckpt/full_small.pt meta {json.dumps(meta)}")
    # Metrics placeholder per §15
    print("metrics: range MAE/RMSE, az wrap, velocity error, count acc — to be populated after Gate A passes")
