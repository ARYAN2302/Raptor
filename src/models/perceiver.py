"""Perceiver bottleneck §2.2 — O(MN) not O(M^2); latents decouple depth."""
import torch, torch.nn as nn
class PerceiverBottleneck(nn.Module):
    def __init__(self, d_model=64, n_latent=32, n_heads=4, n_layers=1):
        super().__init__()
        self.latents=nn.Parameter(torch.randn(1,n_latent,d_model)*0.02)
        self.layers=nn.ModuleList([nn.ModuleDict({
            "cross": nn.MultiheadAttention(d_model,n_heads,batch_first=True),
            "self": nn.MultiheadAttention(d_model,n_heads,batch_first=True),
            "ffn": nn.Sequential(nn.Linear(d_model,d_model*4), nn.GELU(), nn.Linear(d_model*4,d_model)),
            "n1": nn.LayerNorm(d_model), "n2": nn.LayerNorm(d_model), "n3": nn.LayerNorm(d_model),
        }) for _ in range(n_layers)])
    def forward(self, tokens):
        B=tokens.shape[0]
        z=self.latents.expand(B,-1,-1)
        for lyr in self.layers:
            z2,_=lyr["cross"](lyr["n1"](z), tokens, tokens)
            z=z+z2
            z2,_=lyr["self"](lyr["n2"](z), z, z)
            z=z+z2
            z=z+lyr["ffn"](lyr["n3"](z))
        return z
