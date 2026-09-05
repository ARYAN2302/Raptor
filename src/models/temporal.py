"""Temporal state §2.3 — Mamba/SSM placeholder (real Mamba via mamba-ssm if available)."""
import torch, torch.nn as nn

class TemporalStateModel(nn.Module):
    """Chunk-level temporal. In single-window mode acts as identity; in streaming mode uses SSM."""
    def __init__(self, d_model=128, use_mamba=False, n_layers=2):
        super().__init__()
        self.use_mamba=use_mamba
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                self.mamba = nn.ModuleList([Mamba(d_model) for _ in range(n_layers)])
                self.has_mamba=True
            except Exception:
                self.has_mamba=False
                self.mamba=None
        if not use_mamba or not getattr(self,'has_mamba',False):
            # lightweight causal conv + attention fallback
            self.fallback = nn.ModuleList([
                nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())
                for _ in range(n_layers)
            ])
            self.has_mamba=False

    def forward(self, latents, seq_len=1):
        # latents: [B,L,D] for single window -> [B, seq, L, D] if seq>1
        if seq_len==1:
            return latents
        # seq >1: temporal over seq dimension
        # naive: mean-pool latents per step, run SSM, broadcast back (POC)
        # expected input [B, S, L, D]
        if latents.dim()==4:
            B,S,L,D = latents.shape
            pooled = latents.mean(dim=2)  # [B,S,D]
            if self.has_mamba:
                for m in self.mamba:
                    pooled = pooled + m(pooled)
            else:
                # conv over S
                x = pooled.transpose(1,2)  # [B,D,S]
                for conv in self.fallback:
                    x = x + conv(x)
                pooled = x.transpose(1,2)
            # broadcast temporal context back
            pooled = pooled.unsqueeze(2).expand(-1,-1,L,-1)
            return latents + 0.1*pooled
        return latents
