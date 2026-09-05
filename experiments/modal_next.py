import modal
app = modal.App("raptor-next")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)
@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def temporal_ablation():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import synth_iq
    from src.preprocessing.normalize import normalize_iq
    from src.evaluation.metrics import range_rmse
    # §16 Exp1 single vs Exp2 temporal 4-step
    for mode, use_temp in [("no-temporal", False), ("mamba-4", True)]:
        cfg={"model":{"antennas":4,"d_model":64,"n_latent":32,"n_heads":4,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":1,"temporal": "mamba" if use_temp else "none"}}
        model=RAPTOR(cfg)
        if torch.cuda.is_available(): model=model.cuda()
        # Train 2 epochs on 4-step temporal synthetic where range evolves linearly (Doppler)
        # For smoke, just eval untrained RMSE to show delta
        import random
        pr=[]; gt=[]
        for i in range(32):
            s=synth_iq(T=512,E=4,n_emitters=1,seed=100+i)
            iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
            if torch.cuda.is_available(): iq=iq.cuda()
            # temporal: feed 4 consecutive windows as state
            state=None
            for t in range(4 if use_temp else 1):
                out, state = model(iq, state=state) if use_temp else model(iq)
            pr.append(out["range"][0,0].item()); gt.append(s.emitters[0]["range"])
        print(f"{mode} range RMSE {range_rmse(pr,gt):.1f} m (untrained baseline)")
    # periodic az test
    import torch
    from src.models.probabilistic_heads import az_wrap_loss
    print(f"az wrap loss 359 vs 1 deg: {az_wrap_loss(torch.tensor([359.]), torch.tensor([1.])):.4f} vs MSE {(359-1)**2}")
    print("next done")

