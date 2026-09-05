"""DETR set decoder §4.6 — K queries, existence/range/az/el/velocity/uncertainty, Hungarian."""
import torch, torch.nn as nn
class SetDecoder(nn.Module):
    def __init__(self, d_model=128, n_queries=4, n_heads=4, n_layers=2):
        super().__init__()
        self.n_queries=n_queries
        self.queries=nn.Parameter(torch.randn(1,n_queries,d_model)*0.02)
        self.layers=nn.ModuleList([nn.TransformerDecoderLayer(d_model,n_heads,dim_feedforward=d_model*4,batch_first=True) for _ in range(n_layers)])
        self.exist=nn.Linear(d_model,1)
        self.range=nn.Linear(d_model,1)
        self.az=nn.Linear(d_model,1)
        self.el=nn.Linear(d_model,1)
        self.vel=nn.Linear(d_model,3)
        self.logvar=nn.Linear(d_model,5)
    def forward(self, latents):
        B=latents.shape[0]
        q=self.queries.expand(B,-1,-1)
        x=q
        for lyr in self.layers: x=lyr(x, latents)
        return {
            "existence": torch.sigmoid(self.exist(x).squeeze(-1)),
            "range": torch.nn.functional.softplus(self.range(x).squeeze(-1))*1500,
            "azimuth": torch.tanh(self.az(x).squeeze(-1))*180,
            "elevation": torch.tanh(self.el(x).squeeze(-1))*45,
            "velocity": self.vel(x),
            "logvar": self.logvar(x),
        }
