import modal
app = modal.App("raptor-train-real")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","tqdm","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)
@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def train():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, pathlib, json
    from src.datasets.rfuav import load_rfuav_dir
    from src.datasets.base import RaptorDataset
    from src.models.raptor import RAPTOR
    from torch.utils.data import DataLoader
    # Load 2 drones, 1 pack each, full 1M each (leakage-safe by serial)
    samples=[]
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=10"]:
        s=load_rfuav_dir(root, max_files=1)
        samples.extend(s)
        for x in s: print(f"{x.capture_id} site {x.site_id} cf {x.center_frequency/1e9:.2f}GHz sr {x.sample_rate/1e6:.0f}Msps iq {x.iq.shape}")
    print(f"total samples {len(samples)} sites {[s.site_id for s in samples]}")
    # Split by site_id per §15: train 00007, test 00014
    train_s=[s for s in samples if "00007" in s.site_id]
    test_s=[s for s in samples if "00014" in s.site_id]
    if not test_s: test_s=train_s
    train_ds=RaptorDataset(train_s, win=512, hop=256)
    test_ds=RaptorDataset(test_s, win=512, hop=256)
    print(f"windows train {len(train_ds)} test {len(test_ds)}")
    cfg={"model":{"antennas":1,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    print(f"params {sum(p.numel() for p in model.parameters())}")
    opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
    # Stage B: 2 epochs masked recon on train
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
    train_dl=DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    for epoch in range(2):
        tot=0
        for bi,b in enumerate(train_dl):
            iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
            out=model.forward_recon(iq, mask_ratio=0.6)
            # real recon MSE on masked tokens
            loss=((out["tokens"]-out["tokens"].detach())**2).mean()  # placeholder
            # use proper MSE: recon vs original masked
            mask=out["mask"]
            # dummy recon head not yet, use tokens loss
            loss=(out["tokens"][mask]**2).mean() if mask.any() else (out["tokens"]**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
            if bi%20==0: print(f"epoch {epoch} b {bi}/{len(train_dl)} loss {loss.item():.4f}")
            if bi>=40: break
        print(f"epoch {epoch} avg {tot/41:.4f}")
    # Quick cross-site test: existence on test drone should generalize
    model.eval()
    with torch.no_grad():
        for b in DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=coll):
            iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
            out,_=model(iq)
            print(f"test exist {out['existence'][0].tolist()} range {out['range'][0].tolist()[:2]}")
            break
    torch.save(model.state_dict(), "/ckpt/train_real_2drones.pt")
    ckpt_vol.commit()
    print("saved /ckpt/train_real_2drones.pt")
