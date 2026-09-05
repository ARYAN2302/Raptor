import modal
app = modal.App("raptor-range-continuous")
image = modal.Image.debian_slim().pip_install("torch","numpy","scipy","pyyaml","einops").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
@app.function(image=image, gpu="any", timeout=3600)
def test():
    import sys; sys.path.insert(0, "/root/raptor")
    import torch, numpy as np, random
    from src.models.raptor import RAPTOR
    from src.preprocessing.normalize import normalize_iq
    from src.evaluation.metrics import range_rmse
    # Continuous trajectory generator: one emitter moving with velocity, 4 windows dt=0.1s
    def gen_continuous(seed=0, n_windows=4, dt=0.1, E=4, T=512, sr=100e6, cf=2.4e9):
        rng=np.random.default_rng(seed)
        c=3e8; lam=c/cf
        pos=np.zeros((E,3)); pos[:,0]=np.arange(E)*lam*0.5
        # initial pos
        r0=rng.uniform(100,800); az0=rng.uniform(0,360); el0=rng.uniform(5,20)
        # velocity 5-15 m/s random direction
        vel=rng.uniform(-10,10,3)
        # convert initial spherical to cartesian
        def sph2cart(r,az,el):
            azr=np.deg2rad(az); elr=np.deg2rad(el)
            x=r*np.cos(elr)*np.sin(azr); y=r*np.cos(elr)*np.cos(azr); z=r*np.sin(elr)
            return np.array([x,y,z])
        p0=sph2cart(r0,az0,el0)
        windows=[]
        gts=[]
        for w in range(n_windows):
            p = p0 + vel*dt*w
            r=np.linalg.norm(p)
            az=np.rad2deg(np.arctan2(p[0], p[1]))%360
            el=np.rad2deg(np.arctan2(p[2], np.hypot(p[0],p[1])))
            # generate IQ for this window with correct phase for p
            # Use synth logic but fixed pos per window
            # For smoke, we use plane wave phase from p direction
            kvec=p/np.linalg.norm(p)
            delay=pos.dot(kvec)/c
            phase=2*np.pi*cf*delay
            t=np.arange(T)/sr
            kvec_norm=kvec
            fd=np.dot(vel, kvec)/lam
            base=np.exp(1j*2*np.pi*fd*t)
            iq=np.zeros((T,E,2), dtype=np.float32)
            for e in range(E):
                sig=base*np.exp(1j*phase[e])
                iq[:,e,0]+=sig.real; iq[:,e,1]+=sig.imag
            # add noise
            sig_p=np.mean(iq**2); noise_p=sig_p/(10**(10/10))
            iq+= np.random.randn(*iq.shape).astype(np.float32)*np.sqrt(noise_p)*0.5
            iq_n=normalize_iq(iq)
            windows.append(torch.from_numpy(iq_n))
            gts.append({"range": r, "azimuth": az, "elevation": el, "velocity_xyz": vel.tolist()})
        return torch.stack(windows), gts  # [4,512,4,2], gts 4
    # Test single vs temporal with SAME trajectories
    for mode in ["single","temporal"]:
        print(f"\n=== {mode} continuous trajectory ===")
        cfg={"model":{"antennas":4,"d_model":32,"n_latent":8,"n_heads":2,"patch":8,"stride":8,"perceiver_layers":1,"n_queries":1,"temporal": ("mamba" if mode=="temporal" else "gru")}}
        # Actually use gru for single (no state), mamba for temporal
        from src.models.raptor import RAPTOR
        model=RAPTOR(cfg)
        if torch.cuda.is_available(): model=model.cuda()
        opt=torch.optim.AdamW(model.parameters(), lr=0.0003)
        # train on continuous trajectories
        for epoch in range(2):
            tot=0
            for bi in range(20):
                windows, gts = gen_continuous(seed=bi, n_windows=4)
                if mode=="temporal":
                    # carry state across 4 windows
                    state=None
                    for w in range(4):
                        iq=windows[w].unsqueeze(0)
                        if torch.cuda.is_available(): iq=iq.cuda()
                        ant=torch.zeros(1,4,3)
                        ant[:,:,0]=torch.arange(4).float()*0.06
                        if torch.cuda.is_available(): ant=ant.cuda()
                        out, state = model(iq, antenna_positions=ant, state=state)
                        if w==3:
                            loss=(out["range"][0,0]-gts[w]["range"])**2 * 1e-6
                            opt.zero_grad(); loss.backward(); opt.step()
                            tot+=loss.item()
                else:
                    # single: only last window, no state
                    iq=windows[3].unsqueeze(0)
                    if torch.cuda.is_available(): iq=iq.cuda()
                    ant=torch.zeros(1,4,3)
                    ant[:,:,0]=torch.arange(4).float()*0.06
                    if torch.cuda.is_available(): ant=ant.cuda()
                    out,_=model(iq, antenna_positions=ant, state=None)
                    loss=(out["range"][0,0]-gts[3]["range"])**2 * 1e-6
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot+=loss.item()
            print(f"epoch {epoch} avg {tot/20:.6f}")
        # val on continuous trajectories
        pr=[]; gt=[]
        with torch.no_grad():
            for i in range(32):
                windows, gts = gen_continuous(seed=5000+i, n_windows=4)
                if mode=="temporal":
                    state=None
                    for w in range(4):
                        iq=windows[w].unsqueeze(0)
                        if torch.cuda.is_available(): iq=iq.cuda()
                        ant=torch.zeros(1,4,3)
                        ant[:,:,0]=torch.arange(4).float()*0.06
                        if torch.cuda.is_available(): ant=ant.cuda()
                        out, state = model(iq, antenna_positions=ant, state=state)
                    pr.append(out["range"][0,0].item()); gt.append(gts[3]["range"])
                else:
                    iq=windows[3].unsqueeze(0)
                    if torch.cuda.is_available(): iq=iq.cuda()
                    ant=torch.zeros(1,4,3)
                    ant[:,:,0]=torch.arange(4).float()*0.06
                    if torch.cuda.is_available(): ant=ant.cuda()
                    out,_=model(iq, antenna_positions=ant, state=None)
                    pr.append(out["range"][0,0].item()); gt.append(gts[3]["range"])
        print(f"{mode} continuous range RMSE {range_rmse(pr,gt):.1f} m (same emitter, 4 windows)")
        print(f"NOTE: this is honest continuous, previous 443→401 was invalid (independent seeds)")

