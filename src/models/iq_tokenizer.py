"""Radio-FM-inspired dual-channel tokenizer §2.1 — I/Q separate, phase-preserving."""
import torch, torch.nn as nn
class ComplexIQTokenizer(nn.Module):
    def __init__(self, in_antennas=1, patch=8, stride=8, d_model=64):
        super().__init__()
        self.patch=patch; self.stride=stride
        self.proj=nn.Conv1d(in_antennas*2, d_model, kernel_size=patch, stride=stride, padding=0)
        self.norm=nn.LayerNorm(d_model)
        self.pos=nn.Parameter(torch.randn(1,2048,d_model)*0.02)
    def forward(self, iq):
        # iq [B,T,E,2] -> [B, E*2, T]
        B,T,E,C=iq.shape
        x=iq.permute(0,2,3,1).reshape(B,E*C,T)
        # Conv1d handles variable N via patch; RoPE-like via learned pos truncated
        x=self.proj(x).transpose(1,2)  # [B,L,D]
        x=self.norm(x)
        x=x+self.pos[:,:x.shape[1],:]
        return x
