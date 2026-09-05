import modal
app = modal.App("raptor-final-all2")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)
@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def run():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch
    from src.models.raptor import RAPTOR
    from src.datasets.rfuav import load_rfuav_dir
    from src.datasets.base import RaptorDataset
    from src.preprocessing.normalize import normalize_iq
    from src.datasets.synthetic import synth_iq
    from src.evaluation.metrics import range_rmse, az_rmse
    from torch.utils.data import DataLoader
    import pathlib
    print("=== FINAL ALL (4 drones, honest) ===")
    samples=[]
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=10", "/data/rfuav/FLYSKY FS I6X", "/data/rfuav/FRSKY X9DP2019"]:
        s=load_rfuav_dir(root, max_files=1)
        samples.extend(s)
        for x in s: print(f"{x.capture_id} site {x.site_id} cf {x.center_frequency/1e9:.2f} E={x.iq.shape[1]}")
    print(f"total {len(samples)} sites {set(s.site_id for s in samples)}")
    for temporal in ["gru","mamba"]:
        print(f"\n--- temporal={temporal} ---")
        cfg={"model":{"antennas":4,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal": temporal}}
        model=RAPTOR(cfg)
        if torch.cuda.is_available(): model=model.cuda()
        print(f"params {sum(p.numel() for p in model.parameters())}")
        # Stage B: 1 epoch recon on 4 drones (E=1 each, but model max 4 handles variable)
        sites=list(set(s.site_id for s in samples))
        train_s=[s for s in samples if s.site_id in sites[:2]]
        val_s=[s for s in samples if s.site_id in sites[2:]]
        train_ds=RaptorDataset(train_s, win=512, hop=256)
        val_ds=RaptorDataset(val_s, win=512, hop=256)
        print(f"windows train {len(train_ds)} val {len(val_ds)}")
        def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
        train_dl=DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=coll)
        opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
        for bi,b in enumerate(train_dl):
            iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
            ant=torch.zeros(iq.shape[0],1,3)
            if torch.cuda.is_available(): ant=ant.cuda()
            out=model.forward_recon(iq, mask_ratio=0.6)
            loss=(out["tokens"][out["mask"]]**2).mean() if out["mask"].any() else (out["tokens"]**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if bi%10==0: print(f"recon {temporal} b {bi} loss {loss.item():.4f}")
            if bi>=15: break
        print(f"recon {temporal} done")
        # Stage range continuous with E=4 synthetic (now consistent max 4)
        def gen_cont(seed, n_w=4):
            import numpy as np
            rng=np.random.default_rng(seed)
            c=3e8; lam=c/2.4e9
            pos=np.zeros((4,3)); pos[:,0]=np.arange(4)*lam*0.5
            r0=rng.uniform(100,800); az0=rng.uniform(0,360); el0=rng.uniform(5,20)
            vel=rng.uniform(-10,10,3)
            def sph2cart(r,az,el):
                azr=np.deg2rad(az); elr=np.deg2rad(el)
                return np.array([r*np.cos(elr)*np.sin(azr), r*np.cos(elr)*np.cos(azr), r*np.sin(elr)])
            p0=sph2cart(r0,az0,el0)
            wins=[]; gts=[]
            for w in range(n_w):
                p=p0+vel*0.1*w
                r=np.linalg.norm(p); az=np.rad2deg(np.arctan2(p[0],p[1]))%360; el=np.rad2deg(np.arctan2(p[2], np.hypot(p[0],p[1])))
                kvec=p/np.linalg.norm(p)
                delay=pos.dot(kvec)/c
                phase=2*np.pi*2.4e9*delay
                t=np.arange(512)/100e6
                fd=np.dot(vel,kvec)/lam
                base=np.exp(1j*2*np.pi*fd*t)
                iq=np.zeros((512,4,2), dtype=np.float32)
                for e in range(4):
                    sig=base*np.exp(1j*phase[e])
                    iq[:,e,0]+=sig.real; iq[:,e,1]+=sig.imag
                iq+= np.random.randn(*iq.shape).astype(np.float32)*0.05
                iq_n=normalize_iq(iq)
                wins.append(torch.from_numpy(iq_n))
                gts.append({"range": r, "azimuth": az})
            return torch.stack(wins), gts
        opt=torch.optim.AdamW(model.parameters(), lr=0.0002)
        for epoch in range(1):
            for bi in range(20):
                wins,gts=gen_cont(bi,4)
                state=None
                for w in range(4):
                    iq=wins[w].unsqueeze(0)
                    if torch.cuda.is_available(): iq=iq.cuda()
                    ant=torch.zeros(1,4,3)
                    ant[:,:,0]=torch.arange(4).float()*0.06
                    if torch.cuda.is_available(): ant=ant.cuda()
                    out, state = model(iq, antenna_positions=ant, state=state)
                    if w==3:
                        loss=(out["range"][0,0]-gts[w]["range"])**2 * 1e-6
                        opt.zero_grad(); loss.backward(); opt.step()
                if bi%10==0: print(f"range {temporal} b {bi} loss {loss.item():.6f}")
        pr=[]; gt=[]
        import torch as t2
        with t2.no_grad():
            for i in range(32):
                wins,gts=gen_cont(5000+i,4)
                state=None
                for w in range(4):
                    iq=wins[w].unsqueeze(0)
                    if torch.cuda.is_available(): iq=iq.cuda()
                    ant=torch.zeros(1,4,3)
                    ant[:,:,0]=torch.arange(4).float()*0.06
                    if torch.cuda.is_available(): ant=ant.cuda()
                    out, state = model(iq, antenna_positions=ant, state=state)
                pr.append(out["range"][0,0].item()); gt.append(gts[3]["range"])
        print(f"{temporal} continuous range RMSE {range_rmse(pr,gt):.1f} m")
    print("final all done")

