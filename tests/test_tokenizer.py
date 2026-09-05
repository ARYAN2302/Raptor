import torch
from src.models.iq_tokenizer import ComplexIQTokenizer
from src.models.array_encoder import ArrayEncoder
def test_tokenizer_preserves_eq():
    tok=ComplexIQTokenizer(patch=8,stride=8,d_model=32,max_antennas=4)
    for E in [1,2,4]:
        iq=torch.randn(2,512,E,2)
        out=tok(iq)
        assert out.shape==(2,E*64,32), f"E={E} got {out.shape}"
        # varying T
        iq2=torch.randn(2,1024,E,2)
        out2=tok(iq2)
        assert out2.shape[1]==E*128
def test_antenna_identity():
    tok=ComplexIQTokenizer(patch=8,stride=8,d_model=32,max_antennas=4)
    iq=torch.randn(1,512,2,2)
    out1=tok(iq)
    # swapping antennas should change output (identity preserved, not collapsed)
    iq_swapped=iq[:,:, [1,0], :]
    out2=tok(iq_swapped)
    assert not torch.allclose(out1, out2), "antenna swap should change tokens"
def test_array_encoder_geometry():
    enc=ArrayEncoder(d_model=32)
    tok=ComplexIQTokenizer(patch=8,stride=8,d_model=32,max_antennas=2)
    iq=torch.randn(1,512,2,2)
    tokens=tok(iq)
    pos1=torch.tensor([[[0,0,0],[0.5,0,0]]], dtype=torch.float32)
    pos2=torch.tensor([[[0,0,0],[5.0,0,0]]], dtype=torch.float32)
    out1=enc(pos1, tokens)
    out2=enc(pos2, tokens)
    assert not torch.allclose(out1, out2), "geometry change should change representation"
    # arbitrary E
    for E in [1,3,4]:
        pos=torch.randn(1,E,3)
        iq=torch.randn(1,512,E,2)
        tok2=ComplexIQTokenizer(patch=8,stride=8,d_model=32,max_antennas=8)
        t=tok2(iq)
        enc2=ArrayEncoder(d_model=32)
        o=enc2(pos, t)
        assert o.shape==t.shape

