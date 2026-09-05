import torch
from src.models.perceiver import PerceiverBottleneck
def test_variable_input():
    perc=PerceiverBottleneck(d_model=32,n_latent=8,n_heads=2,n_layers=1)
    for M in [64,128,256]:
        t=torch.randn(2,M,32)
        z=perc(t)
        assert z.shape==(2,8,32)
    # gradient flow
    t=torch.randn(2,128,32, requires_grad=True)
    z=perc(t)
    loss=z.mean()
    loss.backward()
    assert t.grad is not None and t.grad.abs().sum()>0

