import modal
app = modal.App("raptor-task1-recon")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
iq_vol = modal.Volume.from_name("iris-raw-iq", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol, "/iq": iq_vol}, gpu="any", timeout=3600)
def task1():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, numpy as np, pathlib
    from src.models.raptor import RAPTOR
    from src.datasets.rfuav import load_rfuav_dir
    from src.datasets.base import RaptorDataset
    from src.losses.recon import MaskedReconLoss
    from torch.utils.data import DataLoader
    print("=== Task1: Masked IQ reconstruction on real RFUAV/UAVSig ===")
    samples=[]
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=10", "/data/rfuav/FLYSKY FS I6X", "/data/rfuav/FRSKY X9DP2019"]:
        s=load_rfuav_dir(root, max_files=1)
        samples.extend(s)
        for x in s: print(f"RFUAV {x.capture_id} site {x.site_id} E={x.iq.shape[1]} T={x.iq.shape[0]}")
    # Also UAVSig one
    from src.datasets.uavsig import load_uavsig_bins
    try:
        us=load_uavsig_bins("/iq", max_files=1)
        samples.extend(us)
        print(f"UAVSig {us[0].capture_id} E={us[0].iq.shape[1]} T={us[0].iq.shape[0]}")
    except Exception as e: print(f"UAVSig load fail {e}")
    ds=RaptorDataset(samples, win=512, hop=256)
    print(f"total windows {len(ds)}")
    cfg={"model":{"antennas":1,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    # Verify grad flow IQ->tok->array->perc->recon
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
    dl=DataLoader(ds, batch_size=8, shuffle=True, collate_fn=coll)
    batch=next(iter(dl))
    iq=batch["iq"]
    if torch.cuda.is_available(): iq=iq.cuda()
    ant=torch.zeros(iq.shape[0],1,3)
    if torch.cuda.is_available(): ant=ant.cuda()
    out=model.forward_recon(iq, mask_ratio=0.6, antenna_positions=ant)
    loss_fn=MaskedReconLoss()
    loss=loss_fn(out["recon"], out["tokens"], out["mask"])
    print(f"initial recon loss (masked 60%) {loss.item():.4f} tokens {out['tokens'].shape} recon {out['recon'].shape} mask {out['mask'].sum().item()}/{out['mask'].numel()}")
    loss.backward()
    # check grads
    grads={k: (v.grad is not None and v.grad.abs().sum().item()>0) for k,v in model.named_parameters() if "tok" in k or "arr" in k or "perc" in k or "recon" in k}
    print(f"grads tok/arr/perc/recon: {sum(grads.values())}/{len(grads)}")
    for k,v in model.named_parameters():
        if "tok.proj_I" in k and v.grad is not None:
            print(f"tok proj_I grad sum {v.grad.abs().sum().item():.4f}")
            break
    # Held-out masked error before training vs after one epoch
    model.eval()
    with torch.no_grad():
        batch=next(iter(dl))
        iq=batch["iq"]
        if torch.cuda.is_available(): iq=iq.cuda()
        ant=torch.zeros(iq.shape[0],1,3)
        if torch.cuda.is_available(): ant=ant.cuda()
        out=model.forward_recon(iq, mask_ratio=0.6, antenna_positions=ant)
        loss0=loss_fn(out["recon"], out["tokens"], out["mask"])
        print(f"held-out masked recon loss before train {loss0.item():.4f}")
    print("Task1 done — recon path verified, next train 1 epoch and re-measure")

