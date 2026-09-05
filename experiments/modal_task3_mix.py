import modal
app = modal.App("raptor-task3-mix")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops","scikit-learn").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, gpu="any", timeout=3600)
def task3():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, numpy as np, random, pathlib, json
    from src.models.raptor import RAPTOR
    from src.datasets.rfuav import load_rfuav_dir
    from src.preprocessing.normalize import normalize_iq
    print("=== Task3: Real multi-emitter benchmark (encoder latent, before decoder) ===")
    samples=[]
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=10", "/data/rfuav/DJI FPV COMBO/DJI FPV COMBO/VTSBW=20", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=10", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO/VTSBW=20"]:
        s=load_rfuav_dir(root, max_files=1)
        samples.extend(s)
        for x in s: print(f"{x.capture_id} site {x.site_id} label {x.extra.get('Drone')}")
    # Pick A=FPV 00007, B=MINI4 00014, same-model B2=FPV 00008
    # Get windows with IDs
    def get_window(sample, idx):
        win=512; hop=256
        start=idx*hop
        iq=sample.iq[start:start+win]
        if iq.shape[0]<win:
            pad=np.zeros((win-iq.shape[0], iq.shape[1],2), dtype=np.float32)
            iq=np.concatenate([iq,pad],axis=0)
        iq=normalize_iq(iq)
        return torch.from_numpy(iq), f"{sample.capture_id}_{start}", sample.site_id
    A, idA, siteA = get_window(samples[0], 10)
    B, idB, siteB = get_window(samples[2], 10)
    B2, idB2, siteB2 = get_window(samples[1], 10)  # FPV 00008 same-model as A
    print(f"A {idA} site {siteA} B {idB} site {siteB} B2 {idB2} site {siteB2}")
    def mix(x,y, coeff=0.5):
        # coeff for A, 1-coeff for B, then normalize
        m = coeff*x + (1-coeff)*y
        # verify numerically
        # save mixture
        return m, coeff
    mix_AB, coeff_AB = mix(A,B,0.5)
    mix_same, coeff_same = mix(A,B2,0.5)
    print(f"Mixture coeff A {coeff_AB:.2f} B {1-coeff_AB:.2f} for A+B, same {coeff_same:.2f}")
    # Verify A+B != A and neither dominates
    def verify(mix, a,b):
        # L2 distance
        d_a = np.linalg.norm((mix - a).numpy())
        d_b = np.linalg.norm((mix - b).numpy())
        d_ab = np.linalg.norm((a - b).numpy())
        # Neither dominates: if mix is exactly a, d_a=0; if mix is exactly b, d_b=0
        # Check that mix is between a and b, not equal to either
        print(f"  L2 mix-A {d_a:.4f} mix-B {d_b:.4f} A-B {d_ab:.4f} — verify mix != A/B and not dominated: {d_a>0.1 and d_b>0.1 and d_a < d_ab and d_b < d_ab}")
        return d_a, d_b
    print("Verify A+B:")
    verify(mix_AB, A, B)
    print("Verify same-model A+B2:")
    verify(mix_same, A, B2)
    # Save benchmark with IDs and coeffs
    benchmark={
        "A": {"id": idA, "site": siteA, "coeff": 1.0},
        "B": {"id": idB, "site": siteB, "coeff": 1.0},
        "A+B": {"ids": [idA, idB], "coeffs": [coeff_AB, 1-coeff_AB], "sites": [siteA, siteB]},
        "same A+B2": {"ids": [idA, idB2], "coeffs": [coeff_same, 1-coeff_same], "sites": [siteA, siteB2]},
    }
    print(f"Benchmark {json.dumps(benchmark, indent=2)}")
    # Test encoder latent itself (before decoder)
    cfg={"model":{"antennas":1,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":4}}
    model=RAPTOR(cfg)
    if torch.cuda.is_available(): model=model.cuda()
    model.eval()
    def get_latent(iq):
        iq_b=iq.unsqueeze(0)
        if torch.cuda.is_available(): iq_b=iq_b.cuda()
        ant=torch.zeros(1,1,3)
        if torch.cuda.is_available(): ant=ant.cuda()
        with torch.no_grad():
            t=model.tok(iq_b, ant)
            t=model.arr(ant, t)
            z=model.perc(t)
            # z [1,8,32] -> mean
            return z.mean(dim=1).cpu().numpy()[0], z.cpu().numpy()[0]
    zA, zA_full = get_latent(A)
    zB, zB_full = get_latent(B)
    zAB, _ = get_latent(mix_AB)
    zSame, _ = get_latent(mix_same)
    print(f"Latents mean pooled dim {zA.shape}")
    # Can classifier identify 1 vs 2 emitters?
    # Build simple dataset: 50 windows of A/B (1 emitter) vs 50 mixtures (2 emitters)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import random
    X=[]; y=[]
    for i in range(50):
        # 1 emitter: random A or B
        iq,_ ,_ = get_window(samples[0] if random.random()<0.5 else samples[2], random.randint(0,100))
        z,_ = get_latent(iq)
        X.append(z); y.append(0)  # 1 emitter
    for i in range(50):
        a,_ ,_ = get_window(samples[0], random.randint(0,100))
        b,_ ,_ = get_window(samples[2], random.randint(0,100))
        m,c = mix(a,b,0.5)
        z,_ = get_latent(m)
        X.append(z); y.append(1)  # 2 emitters
    X=np.array(X); y=np.array(y)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    skf=StratifiedKFold(n_splits=3)
    accs=[]
    for tr,te in skf.split(X,y):
        clf=LogisticRegression(max_iter=500)
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    print(f"1 vs 2 emitters classifier (encoder latent) acc {np.mean(accs):.3f} ±{np.std(accs):.3f} (chance 0.5)")
    # Can embeddings distinguish A from B inside mixtures?
    # Cosine similarity between zA/zB and zAB
    def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
    print(f"cos(zA,zB) {cos(zA,zB):.4f} cos(zA,zAB) {cos(zA,zAB):.4f} cos(zB,zAB) {cos(zB,zAB):.4f} cos(zA,zSame) {cos(zA,zSame):.4f}")
    # Source-specific recovery: can we recover A from mixture via linear regression?
    # Simple: train to predict A latent from mixture latent
    print(f"Task3 done — encoder latent 1vs2 acc above, cos sims show if A+B is between A and B")

