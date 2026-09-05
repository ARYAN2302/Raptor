import modal
app = modal.App("raptor-step13")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","tqdm","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)
@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def step13():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import SyntheticIQDataset, synth_iq
    from src.preprocessing.normalize import normalize_iq
    from torch.utils.data import DataLoader
    cfg={"model":{"antennas":4,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    # 0-2 emitters
    class MixDS(torch.utils.data.Dataset):
        def __len__(self): return 256
        def __getitem__(self,i):
            n=i%3
            s=synth_iq(T=512,E=4,n_emitters=n,seed=i)
            iq=normalize_iq(s.iq)
            return {"iq": torch.from_numpy(iq), "count": n}
    ds=MixDS()
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b]), "count": torch.tensor([x["count"] for x in b])}
    dl=DataLoader(ds,batch_size=16,shuffle=True,collate_fn=coll)
    opt=torch.optim.AdamW(model.parameters(), lr=0.0002)
    for epoch in range(2):
        for bi,b in enumerate(dl):
            iq=b["iq"].cuda() if torch.cuda.is_available() else b["iq"]
            out,_=model(iq)
            # count via existence sum vs gt
            pred_count=(out["existence"]>0.5).sum(dim=1).float()
            loss=(pred_count - b["count"].float().cuda().float()).abs().mean()*0.1 + (1-out["existence"].mean())*0.1
            # dummy
            loss=loss.mean() if hasattr(loss,'mean') else loss
            opt.zero_grad(); loss.backward(); opt.step()
            if bi%10==0: print(f"ep {epoch} b {bi} count loss {loss.item():.4f} pred {pred_count[:3].tolist()} gt {b['count'][:3].tolist()}")
            if bi>=15: break
        # val 0-2
        with torch.no_grad():
            correct=0
            for n in [0,1,2]:
                s=synth_iq(T=512,E=4,n_emitters=n,seed=2000+n)
                iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                if torch.cuda.is_available(): iq=iq.cuda()
                out,_=model(iq)
                pred=(out["existence"][0]>0.5).sum().item()
                print(f"val n_true {n} pred_count {pred} exist {out['existence'][0].tolist()}")
                if pred==n: correct+=1
            print(f"epoch {epoch} count acc {correct}/3")
    print("step13 0-2 done")

