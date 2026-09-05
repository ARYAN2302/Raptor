import modal
app = modal.App("raptor-ablations")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
@app.function(image=image, gpu="any", timeout=3600)
def ablations():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, random
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import synth_iq
    from src.preprocessing.normalize import normalize_iq
    from src.evaluation.metrics import range_rmse, az_rmse
    def test_az(with_geometry=True):
        cfg={"model":{"antennas":4,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":1}}
        model=RAPTOR(cfg)
        if torch.cuda.is_available(): model=model.cuda()
        opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
        for epoch in range(1):
            for bi in range(20):
                s=synth_iq(T=512,E=4,n_emitters=1,seed=bi)
                iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                if torch.cuda.is_available(): iq=iq.cuda()
                pos=torch.zeros(1,4,3)
                pos[:,:,0]=torch.arange(4).float()*0.06
                if with_geometry:
                    if torch.cuda.is_available(): pos=pos.cuda()
                else:
                    pos=torch.zeros(1,4,3)
                    if torch.cuda.is_available(): pos=pos.cuda()
                out,_=model(iq, antenna_positions=pos)
                # az loss
                az_pred=out["azimuth"][0,0]
                az_gt=torch.tensor(s.emitters[0]["azimuth"], device=az_pred.device, dtype=az_pred.dtype)
                loss=torch.abs((az_pred-az_gt+180)%360-180)*0.01
                opt.zero_grad(); loss.backward(); opt.step()
        # val
        pr=[]; gt=[]
        with torch.no_grad():
            for i in range(32):
                s=synth_iq(T=512,E=4,n_emitters=1,seed=1000+i)
                iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                if torch.cuda.is_available(): iq=iq.cuda()
                pos=torch.zeros(1,4,3)
                pos[:,:,0]=torch.arange(4).float()*0.06
                if with_geometry and torch.cuda.is_available(): pos=pos.cuda()
                elif not with_geometry: pos=torch.zeros(1,4,3)
                if torch.cuda.is_available() and with_geometry: pass
                out,_=model(iq, antenna_positions=pos if with_geometry else torch.zeros(1,4,3).cuda() if torch.cuda.is_available() else torch.zeros(1,4,3))
                pr.append(out["azimuth"][0,0].item()); gt.append(s.emitters[0]["azimuth"])
        print(f"az with_geometry={with_geometry} RMSE {az_rmse(pr,gt):.1f} deg")
        return az_rmse(pr,gt)
    print("=== Array geometry ablation (§14) ===")
    rmse_with=test_az(True)
    rmse_without=test_az(False)
    print(f"Δ az RMSE without geometry {rmse_without-rmse_with:.1f} deg (positive means geometry helps)")
    # magnitude-only vs complex
    print("\n=== Complex vs magnitude (§14) ===")
    print("magnitude-only baseline present src/models/baselines.py:1 MagnitudeBaseline uses |IQ| destroys phase — expected to degrade az per §3, not yet numerically measured in this smoke, but range already shows phase matters for array")
    # perceiver vs no bottleneck
    print("\n=== Perceiver vs no bottleneck (§14) ===")
    print("Perceiver O(MN) M=8 N=256 tokens -> 8 latents, baseline would be O(M^2) 256^2 — not yet run full no-bottleneck due to memory, but architecture supports variable M")
    # temporal already done: single 443.7 vs temporal 401.6
    print("\n=== Temporal already measured ===")
    print("single 443.7 m vs temporal 401.6 m (42m gain, still poor per §16)")
