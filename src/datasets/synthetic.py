import numpy as np, torch
from ..dataio.canonical import CanonicalSample
def synth_iq(T=4096,E=1,sample_rate=100e6,center_freq=2.4e9,n_emitters=1,snr_db=10,seed=0):
    rng=np.random.default_rng(seed)
    c=3e8; lam=c/center_freq; pos=np.zeros((E,3)); pos[:,0]=np.arange(E)*lam*0.5
    iq=np.zeros((T,E,2),dtype=np.float32)
    emitters=[]
    for k in range(n_emitters):
        r=rng.uniform(50,1500); az=rng.uniform(0,360); el=rng.uniform(2,25)
        kvec=np.array([np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az)), np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az)), np.sin(np.deg2rad(el))])
        phase=2*np.pi*center_freq*pos.dot(kvec)/c
        vel=rng.uniform(-15,15,3)
        fd=float(np.dot(vel,kvec)/lam)
        t=np.arange(T)/sample_rate
        base=np.exp(1j*2*np.pi*fd*t)
        for e in range(E):
            sig=base*np.exp(1j*phase[e])
            iq[:,e,0]+=sig.real; iq[:,e,1]+=sig.imag
        emitters.append(dict(id=k, range=r, azimuth=az, elevation=el, velocity_xyz=vel.tolist()))
    sig_p=np.mean(iq**2); noise_p=sig_p/(10**(snr_db/10)) if n_emitters>0 else 1e-3
    iq+=rng.normal(0, np.sqrt(noise_p), size=iq.shape).astype(np.float32)
    return CanonicalSample(iq=iq, sample_rate=sample_rate, center_frequency=center_freq, bandwidth=sample_rate, antenna_positions=pos, emitter_count=n_emitters, emitters=emitters, source_dataset="synthetic", capture_id=f"synth_{seed}")
class SyntheticIQDataset(torch.utils.data.Dataset):
    def __init__(self,n=256,T=4096,E=1,max_emitters=1):
        self.n=n; self.T=T; self.E=E; self.max_emitters=max_emitters
    def __len__(self): return self.n
    def __getitem__(self,i):
        n_emit=int(torch.randint(0,self.max_emitters+1,(1,)).item())
        s=synth_iq(T=self.T,E=self.E,n_emitters=n_emit,snr_db=float(np.random.uniform(0,20)),seed=int(i))
        from ..preprocessing.normalize import normalize_iq
        iq=normalize_iq(s.iq)
        return {"iq": torch.from_numpy(iq), "label": n_emit, "site_id": f"synth_{i%4}"}
