import modal
app = modal.App("raptor-train-full")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","tqdm","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)

@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=7200)
def train():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, json, pathlib
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import synth_iq
    from src.preprocessing.normalize import normalize_iq
    from src.evaluation.metrics import range_rmse, az_rmse
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    cfg={"model":{"antennas":4,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    print(f"params {sum(p.numel() for p in model.parameters())}")
    class FullDS(Dataset):
        def __init__(self,n,T,E): self.n=n; self.T=T; self.E=E
        def __len__(self): return self.n
        def __getitem__(self,i):
            import random
            n_emit = random.choice([0,1,1,1,2])  # bias to 1-2
            s=synth_iq(T=self.T,E=self.E,n_emitters=n_emit,seed=i)
            iq=normalize_iq(s.iq)
            # targets
            r=torch.zeros(4); az=torch.zeros(4); ex=torch.zeros(4)
            for k,e in enumerate(s.emitters[:4]):
                ex[k]=1; r[k]=e["range"]/1500.0; az[k]=e["azimuth"]/180.0
            return {"iq": torch.from_numpy(iq), "exist": ex, "r": r, "az": az, "n": n_emit}
    ds=FullDS(1024,512,4)
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b]), "exist": torch.stack([x["exist"] for x in b]), "r": torch.stack([x["r"] for x in b]), "az": torch.stack([x["az"] for x in b])}
    dl=DataLoader(ds,batch_size=16,shuffle=True,collate_fn=coll)
    opt=torch.optim.AdamW(model.parameters(), lr=0.0002)
    for epoch in range(8):
        tot=0
        for bi,b in enumerate(dl):
            iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
            out,_=model(iq)
            # Hungarian placeholder: BCE existence + L1 range/az (periodic)
            loss_ex=nn.functional.binary_cross_entropy(out["existence"], b["exist"].cuda().float())
            loss_r=(out["range"]/1500.0 - b["r"].cuda().float()).abs().mean()*0.5
            # az wrap: (pred-az)*180
            az_pred=out["azimuth"]/180.0
            az_wrap=(az_pred - b["az"].cuda().float() + 1) % 2 - 1
            loss_az=(az_wrap.abs().mean())*0.2
            loss=loss_ex + loss_r + loss_az
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
            if bi%20==0: print(f"ep {epoch} b {bi} loss {loss.item():.4f} ex {loss_ex.item():.3f} r {loss_r.item():.3f} az {loss_az.item():.3f}")
        print(f"epoch {epoch} avg {tot/len(dl):.4f}")
        # val
        with torch.no_grad():
            pr_r=[]; gt_r=[]; pr_az=[]; gt_az=[]; correct=0; total=0
            for i in range(64):
                s=synth_iq(T=512,E=4,n_emitters=1,seed=5000+i)
                iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                if torch.cuda.is_available(): iq=iq.cuda()
                out,_=model(iq)
                pr_r.append(out["range"][0,0].item()); gt_r.append(s.emitters[0]["range"])
                pr_az.append(out["azimuth"][0,0].item()); gt_az.append(s.emitters[0]["azimuth"])
                pred_n=(out["existence"][0]>0.5).sum().item()
                if pred_n==1: correct+=1
                total+=1
            print(f"val range RMSE {range_rmse(pr_r,gt_r):.1f} az RMSE {az_rmse(pr_az,gt_az):.1f} count acc {correct}/{total}")
    pathlib.Path("/ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "/ckpt/full_8ep.pt")
    ckpt_vol.commit()
    print("saved /ckpt/full_8ep.pt")

