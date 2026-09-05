import torch
from src.models.temporal_mamba import TemporalMamba
def test_streaming():
    tm=TemporalMamba(d_model=32)
    lat=torch.randn(2,8,32)
    out1, s1 = tm(lat, None)
    out2, _ = tm(lat, None)
    # with state, output should differ
    out3, _ = tm(lat, s1)
    assert not torch.allclose(out1, out3), "state should affect output"
    # batched state
    lat2=torch.randn(2,8,32)
    out4, s4 = tm(lat2, s1)
    assert s4.shape==(2,8,32)
    # sequence
    seq=torch.randn(2,4,8,32)
    outs, sf = tm.forward_sequence(seq, None)
    assert outs.shape==(2,4,8,32)
    # reset
    r=tm.reset(2,8,lat.device)
    assert r.shape==(2,8,32) and r.abs().sum()==0

