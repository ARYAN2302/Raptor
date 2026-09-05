import modal
app = modal.App("raptor-real-through")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
iq_vol = modal.Volume.from_name("iris-raw-iq", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol, "/iq": iq_vol}, gpu="any", timeout=3600)
def real():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch
    from src.datasets.rfuav import load_rfuav_dir
    from src.datasets.uavsig import load_uavsig_bins
    from src.datasets.base import RaptorDataset
    from src.models.raptor import RAPTOR
    import numpy as np
    print("=== RFUAV small subset through RAPTOR ===")
    samples=[]
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=10"]:
        s=load_rfuav_dir(root, max_files=1)
        samples.extend(s)
        for x in s:
            print(f"RFUAV {x.capture_id} E={x.iq.shape[1]} T={x.iq.shape[0]} sr={x.sample_rate} cf={x.center_frequency} bw={x.bandwidth} site={x.site_id} valid labels: emitter_count={x.emitter_count} pos_valid=False")
    from src.datasets.base import RaptorDataset
    ds=RaptorDataset(samples, win=512, hop=256)
    print(f"windows {len(ds)}")
    # RAPTOR expects [B,T,E,2] with E=1
    cfg={"model":{"antennas":1,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4,"temporal":"mamba"}}
    model=RAPTOR(cfg)
    import torch
    if torch.cuda.is_available(): model=model.cuda()
    from torch.utils.data import DataLoader
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
    dl=DataLoader(ds, batch_size=4, shuffle=False, collate_fn=coll)
    batch=next(iter(dl))
    iq=batch["iq"]
    print(f"batch iq {iq.shape} dtype {iq.dtype}")
    # need antenna_positions [B,E,3]
    B=iq.shape[0]; E=1
    ant_pos=torch.zeros(B,E,3)
    if torch.cuda.is_available():
        iq=iq.cuda(); ant_pos=ant_pos.cuda(); model=model.cuda()
    out, state = model(iq, antenna_positions=ant_pos)
    print(f"RFUAV through RAPTOR ok: tokens {out['tokens'].shape} latents {out['latents'].shape} exist {out['existence'].shape} state {state.shape}")
    # check preprocessing: normalize
    from src.preprocessing.normalize import normalize_iq
    print(f"iq mean {iq.float().mean().item():.6f} std {iq.float().std().item():.4f}")
    # UAVSig
    print("\n=== UAVSig small subset through RAPTOR ===")
    try:
        us=load_uavsig_bins("/iq", max_files=1)
        print(f"UAVSig {us[0].capture_id} iq {us[0].iq.shape} sr {us[0].sample_rate} site {us[0].site_id}")
        ds2=RaptorDataset(us, win=512, hop=256)
        batch2=next(iter(DataLoader(ds2, batch_size=2, collate_fn=coll)))
        iq2=batch2["iq"]
        print(f"UAVSig batch {iq2.shape}")
        if iq2.shape[0]>0:
            ant2=torch.zeros(iq2.shape[0],1,3)
            if torch.cuda.is_available(): iq2=iq2.cuda(); ant2=ant2.cuda()
            out2,_=model(iq2, antenna_positions=ant2)
            print(f"UAVSig through RAPTOR ok: tokens {out2['tokens'].shape} exist {out2['existence'][0].tolist()}")
            print(f"fix: UAVSig int16 scaled /32768, RFUAV complex float 1M cap, both normalized per_window")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"UAVSig fail {e}")

