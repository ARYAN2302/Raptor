"""DETR-style set decoder §2.4 — variable cardinality via learned queries + no-object."""
import torch, torch.nn as nn
import math

class SetDecoder(nn.Module):
    def __init__(self, d_model=128, n_queries=4, n_heads=4, n_layers=2):
        super().__init__()
        self.n_queries=n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model)*0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        # heads
        self.exist = nn.Linear(d_model, 1)
        self.range = nn.Linear(d_model, 1)
        self.az = nn.Linear(d_model, 1)  # will use sin/cos internally
        self.el = nn.Linear(d_model, 1)
        self.vel = nn.Linear(d_model, 3)
        self.logvar = nn.Linear(d_model, 5)  # uncertainty per continuous field
        self.cls = nn.Linear(d_model, 8)  # optional identity embedding/class

    def forward(self, latents):
        # latents: [B,L,D] as memory
        B = latents.shape[0]
        q = self.queries.expand(B,-1,-1)
        x = q
        for lyr in self.layers:
            x = lyr(x, latents)
        out = {
            "existence": torch.sigmoid(self.exist(x).squeeze(-1)),  # [B,Q]
            "range": torch.nn.functional.softplus(self.range(x).squeeze(-1))*1500,  # positive
            "azimuth": torch.tanh(self.az(x).squeeze(-1))*180,  # [-180,180] unwrap later
            "elevation": torch.tanh(self.el(x).squeeze(-1))*45,
            "velocity": self.vel(x),  # [B,Q,3]
            "logvar": self.logvar(x),  # [B,Q,5]
            "logits": self.cls(x),
        }
        return out
