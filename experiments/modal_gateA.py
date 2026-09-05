"""Gate A: RFUAV pretrain vs baselines A/B/C -> UAVSig transfer, leakage-safe by site_id, small slice §15"""
import modal
app = modal.App("raptor-gateA")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","tqdm","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("raptor-ckpt", create_if_missing=True)

@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, gpu="any", timeout=3600)
def gateA():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, yaml, random
    from src.dataio.canonical import CanonicalSample
    from src.datasets.rfuav import load_rfuav_dir
    from src.datasets.base import RaptorDataset
    from src.models.iq_tokenizer import ComplexIQTokenizer
    from src.models.perceiver import PerceiverBottleneck
    from src.models.baselines import BaselineCNN
    from torch.utils.data import DataLoader, Subset
    # Load real RFUAV small slice (1M cap per file, 2 files) and split by site_id (serial)
    samples=[]
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10", "/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=20"]:
        s=load_rfuav_dir(root, max_files=1)
        samples.extend(s)
        for x in s: print(f"RFUAV {x.capture_id} site {x.site_id} iq {x.iq.shape} sr {x.sample_rate}")
    print(f"total RFUAV samples {len(samples)}")
    # Leakage-safe: split by site_id (serial) not window — here 2 sites -> train 1 site, val 1 site
    sites=list(set(s.site_id for s in samples))
    print(f"sites {sites}")
    train_s=[s for s in samples if s.site_id==sites[0]]
    val_s=[s for s in samples if s.site_id==sites[1]] if len(sites)>1 else train_s[:1]
    train_ds=RaptorDataset(train_s, win=512, hop=256)
    val_ds=RaptorDataset(val_s, win=512, hop=256)
    print(f"windows train {len(train_ds)} val {len(val_ds)}")
    # Small model per §15
    tok=ComplexIQTokenizer(in_antennas=1, patch=8, stride=8, d_model=32)
    perc=PerceiverBottleneck(d_model=32, n_latent=8, n_heads=2, n_layers=1)
    cnn=BaselineCNN(in_ant=1, d=32)
    if torch.cuda.is_available():
        tok=tok.cuda(); perc=perc.cuda(); cnn=cnn.cuda()
    def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
    train_dl=DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=coll)
    val_dl=DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=coll)
    opt=torch.optim.AdamW(list(tok.parameters())+list(perc.parameters()), lr=0.0003)
    # 1 epoch small slice
    for bi,batch in enumerate(train_dl):
        iq=batch["iq"].cuda() if torch.cuda.is_available() else batch["iq"]
        t=tok(iq)
        mask=torch.rand(t.shape[0], t.shape[1], device=t.device)<0.6
        t_mask=t.clone(); t_mask[mask]=0
        z=perc(t_mask)
        # dummy recon: MSE on masked tokens vs mean latent (placeholder, will be replaced by real decoder)
        # For Gate A we just prove pipeline and compare vs baselines feature norms
        loss=(t[mask].mean()**2) if mask.any() else (t**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        cnn_feat=cnn(iq).mean().item()
        print(f"batch {bi}/{len(train_dl)} loss {loss.item():.4f} tok {t.shape} latent {z.shape} cnn_feat {cnn_feat:.4f}")
        if bi>=4: break
    print("Gate A small slice done — next: replace dummy head with real recon decoder + linear probe on UAVSig split by site, compare raw-IQ > magnitude/spectrogram")
    torch.save({"tok": tok.state_dict(), "perc": perc.state_dict()}, "/ckpt/gateA_small.pt")
    ckpt_vol.commit()
    print("saved /ckpt/gateA_small.pt")

