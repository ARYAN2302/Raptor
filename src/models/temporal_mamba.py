"""Temporal state §4.5 — state_t = F(state_{t-1}, latent_t), streaming."""
import torch, torch.nn as nn
class TemporalMamba(nn.Module):
    def __init__(self, d_model=128, use_mamba=False):
        super().__init__()
        self.use_mamba=use_mamba
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                self.mamba=Mamba(d_model)
                self.has=True
            except: self.has=False
        if not use_mamba or not getattr(self,'has',False):
            self.fallback=nn.Sequential(nn.Conv1d(d_model,d_model,3,padding=1), nn.GELU())
            self.has=False
    def forward(self, latents, state=None):
        # latents [B,M,D] -> persistent state
        if state is None: return latents, latents
        # streaming: combine
        if self.has:
            out=self.mamba(latents+state)
        else:
            x=(latents+state).transpose(1,2)
            x=x+self.fallback(x)
            out=x.transpose(1,2)
        return out, out
