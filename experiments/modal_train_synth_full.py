import modal
app = modal.App("raptor-synth-full")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)
@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def train():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, random, pathlib, json
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import synth_iq
    from src.preprocessing.normalize import normalize_iq
    from src.losses.set_loss import SetPredictionLoss
    from src.evaluation.metrics import range_rmse, az_rmse
    # Full graph config per §4/§18
    cfg={"model":{"antennas":4,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    print(f"params {sum(p.numel() for p in model.parameters())}")
    # Controlled 1 emitter -> 2 -> 3
    for stage, n_max in [(1,1),(2,2),(3,3)]:
        print(f"\n=== Stage {stage}: {n_max} emitters ===")
        # dataset 256 samples
        def make_batch(B=16, n_max=n_max):
            iqs=[]; gts=[]
            for _ in range(B):
                n = random.randint(0, n_max) if n_max>0 else 1
                # ensure 1 emitter for stage 1
                if stage==1: n=1
                s=synth_iq(T=512,E=4,n_emitters=n,seed=random.randint(0,100000))
                iq=normalize_iq(s.iq)
                iqs.append(torch.from_numpy(iq))
                # gt list of dicts
                gts.append(s.emitters)
            iqs=torch.stack(iqs)
            return iqs, gts
        opt=torch.optim.AdamW(model.parameters(), lr=0.0002)
        crit=SetPredictionLoss()
        for epoch in range(2):
            tot=0
            for bi in range(10):
                iqs, gts = make_batch()
                if torch.cuda.is_available(): iqs=iqs.cuda()
                # need antenna positions for array_encoder (use synthetic pos)
                # synthetic pos is [E,3] ULA, expand B
                ant_pos=torch.zeros(iqs.shape[0],4,3)
                ant_pos[:,:,0]=torch.arange(4).float()*0.06  # 0.06m ~ 0.5 lambda at 2.4GHz
                if torch.cuda.is_available(): ant_pos=ant_pos.cuda()
                # forward with state None for single window
                out, _ = model(iqs, antenna_positions=ant_pos)
                loss=crit(out, gts)
                opt.zero_grad(); loss.backward(); opt.step()
                tot+=loss.item()
                if bi%5==0: print(f"stage {stage} ep {epoch} b {bi} loss {loss.item():.4f}")
            print(f"stage {stage} ep {epoch} avg {tot/10:.4f}")
            # quick val
            with torch.no_grad():
                iqs,gts=make_batch(B=16,n_max=n_max)
                if torch.cuda.is_available(): iqs=iqs.cuda()
                ant_pos=torch.zeros(iqs.shape[0],4,3)
                ant_pos[:,:,0]=torch.arange(4).float()*0.06
                if torch.cuda.is_available(): ant_pos=ant_pos.cuda()
                out,_=model(iqs, antenna_positions=ant_pos)
                # count acc
                correct=0
                for b in range(16):
                    n_true=len(gts[b])
                    pred=(out["existence"][b]>0.5).sum().item()
                    if pred==n_true: correct+=1
                print(f"val stage {stage} count acc {correct}/16")
        # save stage ckpt per §19
        pathlib.Path("/ckpt").mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg, "stage": stage}, f"/ckpt/synth_stage{stage}.pt")
        ckpt_vol.commit()
        print(f"saved stage {stage}")
    print("synthetic train 1->3 done")

