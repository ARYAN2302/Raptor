import numpy as np, sys
sys.path.insert(0,"src")
from src.dataio.canonical import CanonicalSample
from src.preprocessing.normalize import normalize_iq
from src.datasets.synthetic import synth_iq
from src.models.raptor import RAPTOR

def test_canonical():
    s=synth_iq(T=1024,E=4,n_emitters=1,seed=0)
    assert s.iq.shape==(1024,4,2)
    q=normalize_iq(s.iq)
    assert q.shape==s.iq.shape

def test_tokenizer():
    m=RAPTOR({"model":{"antennas":4,"d_model":64,"n_latent":16,"n_queries":2}})
    import torch
    s=synth_iq(T=4096,E=4,n_emitters=1,seed=1)
    iq=torch.from_numpy(s.iq).unsqueeze(0)
    out=m(iq)
    assert out["range"].shape==(1,2)
    assert out["azimuth"].shape==(1,2)

def test_temporal_ablation():
    from src.models.temporal import TemporalStateModel
    import torch
    t=TemporalStateModel(d_model=64)
    x=torch.randn(2,16,64)
    y=t(x, seq_len=1)
    assert y.shape==x.shape

def test_mixture_counting():
    from src.datasets.synthetic import synth_iq
    for n in [0,1,2,3]:
        s=synth_iq(T=1024,E=4,n_emitters=n,seed=n)
        assert s.emitter_count==n
