import modal
app = modal.App("raptor-range-temporal")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
@app.function(image=image, gpu="any", timeout=3600)
def range_test():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, random
    from src.models.raptor import RAPTOR
    from src.datasets.synthetic import synth_iq
    from src.preprocessing.normalize import normalize_iq
    from src.evaluation.metrics import range_rmse, az_rmse
    # Single-window vs temporal (using state carry over 4 windows of same trajectory)
    for mode in ["single", "temporal"]:
        print(f"\n=== {mode} ===")
        cfg={"model":{"antennas":4,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":1,"temporal": ("mamba" if mode=="temporal" else "none")}}
        # For single, use no temporal (still has temporal module but we will not carry state)
        model=RAPTOR(cfg)
        if torch.cuda.is_available(): model=model.cuda()
        # Train 2 epochs on single emitter with trajectory (velocity)
        opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
        for epoch in range(2):
            tot=0
            for bi in range(20):
                # For temporal, generate 4 consecutive windows from same moving emitter
                if mode=="temporal":
                    base_seed=bi*10
                    s0=synth_iq(T=512,E=4,n_emitters=1,seed=base_seed)
                    # simple trajectory: same emitter moved by velocity*0.1s
                    # use synth with same pos but varying time? For now same seed sequential windows
                    seq=[]
                    for t in range(4):
                        s=synth_iq(T=512,E=4,n_emitters=1,seed=base_seed+t)
                        iq=normalize_iq(s.iq)
                        seq.append(torch.from_numpy(iq))
                    seq=torch.stack(seq)  # [4,512,4,2] -> need [B,T,E,2] per window, for temporal we need sequence
                    # For this test, we will run temporal over 4 windows as sequence
                    # Collapse B=1, T=4 windows
                    # Use forward_sequence via temporal_mamba
                    # Instead just run 4 forwards with state carry and train on last
                    state=None
                    for t in range(4):
                        iq_t=seq[t].unsqueeze(0)
                        if torch.cuda.is_available(): iq_t=iq_t.cuda()
                        ant=torch.zeros(1,4,3)
                        if torch.cuda.is_available(): ant=ant.cuda()
                        out, state = model(iq_t, antenna_positions=ant, state=state)
                    # loss on last
                    gt_range=s.emitters[0]["range"]
                    loss=(out["range"][0,0]-gt_range)**2 * 0.000001
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot+=loss.item()
                else:
                    s=synth_iq(T=512,E=4,n_emitters=1,seed=bi)
                    iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                    if torch.cuda.is_available(): iq=iq.cuda()
                    ant=torch.zeros(1,4,3)
                    if torch.cuda.is_available(): ant=ant.cuda()
                    out,_=model(iq, antenna_positions=ant, state=None)
                    gt_range=s.emitters[0]["range"]
                    loss=(out["range"][0,0]-gt_range)**2 * 0.000001
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot+=loss.item()
            print(f"epoch {epoch} avg {tot/20:.6f}")
        # val
        pr=[]; gt=[]
        with torch.no_grad():
            for i in range(32):
                s=synth_iq(T=512,E=4,n_emitters=1,seed=5000+i)
                iq=torch.from_numpy(normalize_iq(s.iq)).unsqueeze(0)
                if torch.cuda.is_available(): iq=iq.cuda()
                ant=torch.zeros(1,4,3)
                if torch.cuda.is_available(): ant=ant.cuda()
                out,_=model(iq, antenna_positions=ant, state=None)
                pr.append(out["range"][0,0].item()); gt.append(s.emitters[0]["range"])
        print(f"{mode} range RMSE {range_rmse(pr,gt):.1f} m")
