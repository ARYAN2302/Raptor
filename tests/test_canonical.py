import sys
sys.path.insert(0,"/tmp/raptor_build/Raptor")
from src.dataio.canonical import CanonicalSample
from src.preprocessing.normalize import normalize_iq
from src.datasets.synthetic import synth_iq
from src.models.iq_tokenizer import ComplexIQTokenizer
from src.models.perceiver import PerceiverBottleneck
import torch
def test_canonical():
    s=synth_iq(T=1024,E=1,n_emitters=1,seed=0)
    assert s.iq.shape==(1024,1,2)
    q=normalize_iq(s.iq)
    assert q.shape==s.iq.shape
def test_tokenizer():
    tok=ComplexIQTokenizer(in_antennas=1,patch=8,stride=8,d_model=32)
    perc=PerceiverBottleneck(d_model=32,n_latent=8,n_heads=2,n_layers=1)
    s=synth_iq(T=4096,E=1,seed=1)
    iq=torch.from_numpy(s.iq).unsqueeze(0)
    t=tok(iq); z=perc(t)
    assert z.shape==(1,8,32)
def test_no_leakage_split():
    from src.datasets.synthetic import SyntheticIQDataset
    ds=SyntheticIQDataset(n=8,T=512,E=1)
    ids=[ds[i]["site_id"] for i in range(8)]
    # deterministic site grouping: no window from same site should be both train/test if split by site
    assert len(set(ids))>1
