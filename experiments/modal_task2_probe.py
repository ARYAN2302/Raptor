import modal
app = modal.App("raptor-task2-probe")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, gpu="any", timeout=3600)
def task2():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, numpy as np, pathlib
    from src.models.raptor import RAPTOR
    from src.datasets.rfuav import load_rfuav_dir
    from src.datasets.base import RaptorDataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    print("=== Task2: Leakage-safe cross-serial probe ===")
    # Load 2 serials per model
    samples=[]
    mapping={}  # site -> label
    for root, label in [("/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10",0), ("/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=20",0), ("/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=10",1), ("/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=20",1), ("/data/rfuav/FLYSKY FS I6X",2), ("/data/rfuav/FRSKY X9DP2019",3)]:
        s=load_rfuav_dir(root, max_files=1)
        for x in s:
            x.extra["label"]=label
            print(f"{x.capture_id} site {x.site_id} label {label} cf {x.center_frequency/1e9:.2f}")
        samples.extend(s)
    # Build RAPTOR and extract latents
    cfg={"model":{"antennas":1,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    model.eval()
    def get_latents_for_samples(samp_list, max_w=200):
        from src.datasets.base import RaptorDataset
        from torch.utils.data import DataLoader
        ds=RaptorDataset(samp_list, win=512, hop=256)
        def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
        dl=DataLoader(ds, batch_size=16, shuffle=False, collate_fn=coll)
        lats=[]; labs=[]
        with torch.no_grad():
            for b in dl:
                iq=b["iq"]
                if torch.cuda.is_available(): iq=iq.cuda()
                ant=torch.zeros(iq.shape[0],1,3)
                if torch.cuda.is_available(): ant=ant.cuda()
                t=model.tok(iq, ant)
                t=model.arr(ant, t)
                z=model.perc(t)
                lats.append(z.mean(dim=1).cpu().numpy())
                # labels from samp_list via ds idx? For now assign by site
                # we need to map windows to labels: each window inherits sample label
                # Use ds.samples and idx
                # Instead just use first samp label for all windows in batch (approx)
                # For accurate, we should track via idx
                # Simplify: for now use first sample label
                # But we have 6 samples, each with different label, and windows intermix
                # Let's approximate: for this probe, we will create separate datasets per serial
                if len(lats)*16 >= max_w: break
        return np.concatenate(lats,axis=0)[:max_w], np.array([0]*len(lats)*16)[:max_w]  # placeholder
    # Better: do per-serial latents
    def latents_for_serial(serial_samples, max_w=100):
        ds=RaptorDataset(serial_samples, win=512, hop=256)
        from torch.utils.data import DataLoader
        def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
        dl=DataLoader(ds, batch_size=16, shuffle=False, collate_fn=coll)
        lats=[]
        with torch.no_grad():
            for b in dl:
                iq=b["iq"]
                if torch.cuda.is_available(): iq=iq.cuda()
                ant=torch.zeros(iq.shape[0],1,3)
                if torch.cuda.is_available(): ant=ant.cuda()
                t=model.tok(iq, ant)
                t=model.arr(ant, t)
                z=model.perc(t)
                lats.append(z.mean(dim=1).cpu().numpy())
                if len(lats)*16 >= max_w: break
        return np.concatenate(lats,axis=0)[:max_w]
    # Get per-serial
    # Find samples per serial
    from collections import defaultdict
    by_site=defaultdict(list)
    for s in samples: by_site[s.site_id].append(s)
    for site, lst in by_site.items():
        print(f"site {site} label {lst[0].extra['label']} windows {len(RaptorDataset(lst, win=512, hop=256))}")
    # Leakage-safe: train on 00007 (FPV) test on 00008 (FPV same model)
    site_to_label={"00007":0,"00008":0,"00014":1,"00015":1,"00017":2,"00019":3}
    # FPV same-model
    train_fp07=latents_for_serial(by_site["00007"], 100)
    test_fp08=latents_for_serial(by_site["00008"], 100)
    # MINI4 same-model
    train_m14=latents_for_serial(by_site["00014"], 100)
    test_m15=latents_for_serial(by_site["00015"], 100)
    # For 2-class FPV vs MINI4 same-model probe: train on one serial per model, test on other serial
    X_train=np.concatenate([train_fp07, train_m14],axis=0)
    y_train=np.array([0]*len(train_fp07) + [1]*len(train_m14))
    X_test=np.concatenate([test_fp08, test_m15],axis=0)
    y_test=np.array([0]*len(test_fp08) + [1]*len(test_m15))
    clf=LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    pred=clf.predict(X_test)
    acc=accuracy_score(y_test, pred)
    cm=confusion_matrix(y_test, pred)
    print(f"FPV vs MINI4 same-model leakage-safe (train 00007/00014 -> test 00008/00015): acc {acc:.3f} (chance 0.5)")
    print(f"confusion:\n{cm}")
    # Cross-model: train FPV (both serials) -> test MINI4 (both)
    train_fpv=np.concatenate([train_fp07, latents_for_serial(by_site["00008"],100)],axis=0)
    test_mini=np.concatenate([train_m14, test_m15],axis=0)
    X_train2=train_fpv; y_train2=np.array([0]*len(train_fpv))
    # For cross-model, we need binary FPV vs MINI4, but train has only FPV label 0, test only MINI4 label 1 -> can't train 2-class
    # Instead do train FPV (0) vs MINI4 (1) both in train, test on held-out windows from same captures but different windows (still leaky) — not ideal
    # Instead do train FPV (00007) -> test MINI4 (00014): 2-class but train has only FPV, test only MINI4 -> also single class
    # So for cross-model, we need to report that leakage-safe cross-model is ill-posed with single capture per class
    # We will report same-model as above and also all 4-class leaky for reference
    # All 4-class leaky (random windows, not leakage-safe) for structure
    from src.datasets.base import RaptorDataset
    from torch.utils.data import DataLoader
    # Pooled 4 drones 200 windows each
    all_samples=samples
    ds_all=RaptorDataset(all_samples, win=512, hop=256)
    # Get latents for all
    def get_all_latents(ds, max_w=400):
        from torch.utils.data import DataLoader
        def coll(b): return {"iq": torch.stack([x["iq"] for x in b])}
        dl=DataLoader(ds, batch_size=16, shuffle=True, collate_fn=coll)
        lats=[]; labs=[]
        # need to map windows to labels via ds idx
        # For pooled, we will just use ds.samples and idx to get label
        for b in dl:
            iq=b["iq"]
            if torch.cuda.is_available(): iq=iq.cuda()
            ant=torch.zeros(iq.shape[0],1,3)
            if torch.cuda.is_available(): ant=ant.cuda()
            with torch.no_grad():
                t=model.tok(iq, ant)
                t=model.arr(ant, t)
                z=model.perc(t)
                lats.append(z.mean(dim=1).cpu().numpy())
            # assign labels by nearest sample (approx)
            # For pooled 4-class, we can assign based on ds loop, but for smoke use random labels
            # Instead get labels from ds.samples via idx
            # Simplify: use first 4 labels cyclically
            labs.extend([0,1,2,3]*4)
            if len(lats)*16 >= max_w: break
        return np.concatenate(lats,axis=0)[:max_w], np.array(labs)[:max_w]
    # Actually use proper labels via ds idx
    # Let's do proper: iterate over ds idx and get label per window
    lats_pooled=[]; labs_pooled=[]
    for i in range(min(400, len(ds_all))):
        item=ds_all[i]
        # need to find which sample this window came from
        si, start = ds_all.idx[i]
        lab=samples[si].extra["label"]
        # get latent
        iq=item["iq"].unsqueeze(0)
        if torch.cuda.is_available(): iq=iq.cuda()
        ant=torch.zeros(1,1,3)
        if torch.cuda.is_available(): ant=ant.cuda()
        with torch.no_grad():
            t=model.tok(iq, ant)
            t=model.arr(ant, t)
            z=model.perc(t)
            lats_pooled.append(z.mean(dim=1).cpu().numpy()[0])
            labs_pooled.append(lab)
    lats_pooled=np.array(lats_pooled); labs_pooled=np.array(labs_pooled)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    skf=StratifiedKFold(n_splits=3)
    accs=[]
    for tr,te in skf.split(lats_pooled, labs_pooled):
        clf2=LogisticRegression(max_iter=500)
        clf2.fit(lats_pooled[tr], labs_pooled[tr])
        accs.append(accuracy_score(labs_pooled[te], clf2.predict(lats_pooled[te])))
    print(f"4-class pooled leaky (random windows) acc {np.mean(accs):.3f} ±{np.std(accs):.3f} (chance 0.25)")
    print("Task2 done")

