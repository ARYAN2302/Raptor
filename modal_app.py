"""Modal training — reuses essential volumes only (no bulk RFUAV re-download)."""
import modal

app = modal.App("raptor")

image = modal.Image.debian_slim().pip_install(
    "torch==2.4.1", "numpy", "scipy", "h5py", "pyyaml", "tqdm", "einops", "scikit-learn"
).add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")

data_vol = modal.Volume.from_name("raptor-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)

@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def pretrain(config: str = "configs/training/pretrain.yaml"):
    import sys
    sys.path.insert(0, "/root/raptor")
    import yaml, torch
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import SyntheticIQDataset
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(open(f"/root/raptor/{config}"))
    print("pretrain cfg", cfg)
    import pathlib
    has_real = pathlib.Path("/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10/pack1_0-1s.iq").exists()
    print(f"real RFUAV available: {has_real} — using synthetic for P2 POC (no bulk download)")
    model = RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    ds = SyntheticIQDataset(n=cfg.get("n_synth",512), T=cfg.get("window",4096), E=4, max_emitters=1)
    def collate(b): 
        import torch
        return {"iq": torch.stack([x["iq"] for x in b])}
    dl = DataLoader(ds, batch_size=cfg.get("batch_size",8), shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",0.0003))
    for epoch in range(cfg.get("epochs",3)):
        total=0
        for bi, batch in enumerate(dl):
            iq = batch["iq"]
            if torch.cuda.is_available(): iq=iq.cuda()
            out = model.forward_recon(iq, mask_ratio=cfg.get("mask_ratio",0.4))
            loss = (out["tokens"]**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()
            if bi%20==0:
                print(f"epoch {epoch} batch {bi}/{len(dl)} loss {loss.item():.4f} avg {total/(bi+1):.4f}")
        print(f"epoch {epoch} DONE avg_loss {total/len(dl):.4f}")
    torch.save(model.state_dict(), "/ckpt/pretrain.pt")
    ckpt_vol.commit()
    print("saved /ckpt/pretrain.pt")

@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def train_state(config: str = "configs/training/state.yaml"):
    import sys
    sys.path.insert(0, "/root/raptor")
    import yaml, torch
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import SyntheticIQDataset
    from src.losses.set_loss import SetPredictionLoss
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(open(f"/root/raptor/{config}"))
    print("state cfg", cfg)
    model = RAPTOR({**cfg, "model": {**cfg.get("model",{}), "antennas":4, "n_queries": cfg.get("n_queries",4)}})
    if torch.cuda.is_available(): model=model.cuda()
    ds = SyntheticIQDataset(n=cfg.get("n_synth",1024), T=cfg.get("window",4096), E=4, max_emitters=cfg.get("max_emitters",3))
    def collate(b): 
        import torch
        return {"iq": torch.stack([x["iq"] for x in b]), "gt":[x["emitters"] for x in b]}
    dl = DataLoader(ds, batch_size=cfg.get("batch_size",16), shuffle=True, collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",0.0002))
    crit=SetPredictionLoss()
    for epoch in range(cfg.get("epochs",5)):
        total=0
        for bi, batch in enumerate(dl):
            iq=batch["iq"]
            if torch.cuda.is_available(): iq=iq.cuda()
            out=model(iq)
            loss=crit(out, batch["gt"])
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()
            if bi%20==0:
                print(f"epoch {epoch} batch {bi}/{len(dl)} loss {loss.item():.4f} avg {total/(bi+1):.4f} exist {out['existence'][0].tolist()[:2]}")
        print(f"epoch {epoch} DONE avg_loss {total/len(dl):.4f}")
        # quick range/az sanity on first batch
        try:
            gt=batch["gt"][0]
            if gt:
                print(f"  sample0 gt range {gt[0]['range']:.1f} az {gt[0]['azimuth']:.1f} pred range {out['range'][0][0].item():.1f} az {out['azimuth'][0][0].item():.1f}")
        except: pass
    torch.save(model.state_dict(), "/ckpt/state.pt")
    ckpt_vol.commit()
    print("saved /ckpt/state.pt")

@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol})
def inspect():
    import os
    for base in ["/data","/ckpt"]:
        print(f"\n== {base} ==")
        for root, dirs, files in os.walk(base):
            lvl=root.replace(base,"") or "/"
            if lvl.count("/")>3: 
                dirs[:]=[]
                continue
            print(f"{lvl}: {len(files)} files {files[:5]} {dirs[:5]}")
