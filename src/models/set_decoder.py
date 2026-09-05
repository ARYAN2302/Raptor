"""DETR set decoder §4.6 — K queries, existence/range/az/el/velocity/uncertainty/identity, Hungarian."""
import torch, torch.nn as nn
class SetDecoder(nn.Module):
    def __init__(self, d_model=64, n_queries=4, n_heads=4, n_layers=2, id_dim=32, n_classes=4):
        super().__init__()
        self.n_queries=n_queries
        self.queries=nn.Parameter(torch.randn(1,n_queries,d_model)*0.02)
        self.layers=nn.ModuleList([nn.TransformerDecoderLayer(d_model,n_heads,dim_feedforward=d_model*4,batch_first=True) for _ in range(n_layers)])
        self.exist=nn.Linear(d_model,1)
        self.range=nn.Linear(d_model,1)
        self.az=nn.Linear(d_model,1)
        self.el=nn.Linear(d_model,1)
        self.vel=nn.Linear(d_model,3)
        self.logvar=nn.Linear(d_model,5)  # range, az, el, vel(3) shared or 5
        self.id_emb=nn.Linear(d_model, id_dim)
        self.cls=nn.Linear(d_model, n_classes)
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
            "velocity": self.vel(x),  # [B,K,3] ENU
            "logvar": self.logvar(x),  # [B,K,5] heteroscedastic
            "identity": torch.nn.functional.normalize(self.id_emb(x), dim=-1),  # [B,K,id_dim]
            "logits": self.cls(x),  # [B,K,n_classes]
        }
