import modal
app = modal.App("raptor-step12")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","tqdm","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)
@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def step12():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, pathlib, json
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import SyntheticIQDataset
    from src.losses.set_loss import SetPredictionLoss
    from src.evaluation.metrics import range_rmse, az_rmse
    from torch.utils.data import DataLoader
    cfg={"model":{"antennas":4,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":1,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    # One emitter LOS controlled
    ds=SyntheticIQDataset(n=512,T=512,E=4,max_emitters=1)
    # force 1 emitter for step 12
    import numpy as np
    from src.datasets.synthetic import synth_iq
    from src.preprocessing.normalize import normalize_iq
    # build 1-emitter only dataset
    class OneDS(torch.utils.data.Dataset):
        def __len__(self): return 256
        def __getitem__(self,i):
            s=synth_iq(T=512,E=4,n_emitters=1,seed=i)
            iq=normalize_iq(s.iq)
            return {"iq": torch.from_numpy(iq), "range": s.emitters[0]["range"], "az": s.emitters[0]["azimuth"], "el": s.emitters[0]["elevation"]}
    ds=OneDS()
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b]), "range": torch.tensor([x["range"] for x in b]), "az": torch.tensor([x["az"] for x in b])}
    dl=DataLoader(ds,batch_size=16,shuffle=True,collate_fn=coll)
    opt=torch.optim.AdamW(model.parameters(), lr=0.0002)
    for epoch in range(3):
        tot=0
        for bi,b in enumerate(dl):
            iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
            out,_=model(iq)
            # existence + range + az loss (deterministic start per §12)
            loss=(out["range"][:,0]-b["range"].cuda().float()).abs().mean()*0.001 + (1-out["existence"][:,0]).mean()*0.5
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
            if bi%10==0: print(f"ep {epoch} b {bi} loss {loss.item():.4f}")
            if bi>=20: break
        print(f"epoch {epoch} avg {tot/21:.4f}")
        # quick val RMSE on 32 samples
        with torch.no_grad():
            pr=[]; gt=[]
            for i in range(32):
                s=synth_iq(T=512,E=4,n_emitters=1,seed=1000+i)
                iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                if torch.cuda.is_available(): iq=iq.cuda()
                out,_=model(iq)
                pr.append(out["range"][0,0].item()); gt.append(s.emitters[0]["range"])
            print(f"val range RMSE {range_rmse(pr,gt):.1f} m")
    pathlib.Path("/ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "/ckpt/step12_one.pt")
    ckpt_vol.commit()
    print("saved /ckpt/step12_one.pt")

