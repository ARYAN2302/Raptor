import torch
from src.models.set_decoder import SetDecoder
from src.models.perceiver import PerceiverBottleneck
def test_variable_cardinality():
    dec=SetDecoder(d_model=32,n_queries=4,n_heads=2)
    for B in [1,2]:
        lat=torch.randn(B,8,32)
        out=dec(lat)
        assert out["existence"].shape==(B,4)
        assert out["range"].shape==(B,4)
        assert out["identity"].shape==(B,4,32)
        assert out["logits"].shape==(B,4,4)

