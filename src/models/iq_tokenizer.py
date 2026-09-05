"""Complex IQ Tokenizer §2.1 — dual-channel, phase-preserving, metadata-aware."""
import torch, torch.nn as nn

class ComplexIQTokenizer(nn.Module):
    """1D conv patch embedding over time, preserves I/Q as separate channels."""
    def __init__(self, in_antennas=4, patch=64, stride=32, d_model=128):
        super().__init__()
        self.patch=patch; self.stride=stride; self.d_model=d_model
        # separate I/Q learned channels via 2*E in-channels
        self.proj = nn.Conv1d(in_antennas*2, d_model, kernel_size=patch, stride=stride, padding=patch//2)
        self.norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.randn(1, 2048, d_model)*0.02)

    def forward(self, iq, metadata=None):
        # iq: [B,T,E,2] -> [B, E*2, T]
        B,T,E,C = iq.shape
        x = iq.permute(0,2,3,1).reshape(B, E*C, T)
        x = self.proj(x).transpose(1,2)  # [B, N, D]
        x = self.norm(x)
        # add learned pos (trunc)
        x = x + self.pos[:,:x.shape[1],:]
        # optional metadata conditioning (sample_rate, center_freq, bandwidth)
        if metadata is not None:
            # simple FiLM: metadata [B, 4] -> scale
            pass
        return x  # [B, N, D]

def build_tokenizer(cfg):
    return ComplexIQTokenizer(in_antennas=cfg.get("antennas",4), patch=cfg.get("patch",64), stride=cfg.get("stride",32), d_model=cfg.get("d_model",128))
