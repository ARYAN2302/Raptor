"""Baselines §10 A-C for Gate A."""
import torch, torch.nn as nn
class BaselineCNN(nn.Module):
    def __init__(self, in_ant=1, d=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(in_ant*2,64,7,padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64,d))
    def forward(self,iq):
        B,T,E,C=iq.shape
        x=iq.permute(0,2,3,1).reshape(B,E*C,T)
        return self.net(x)
class MagnitudeBaseline(nn.Module):
    # B: uses |IQ| only, destroys phase (§11 ablation)
    def forward(self,iq): return torch.sqrt((iq**2).sum(-1))  # [B,T,E]
class SpectrogramBaseline(nn.Module):
    def forward(self,iq):
        # magnitude STFT on I channel mean antenna
        x=iq[:,:,0,0].float()  # [B,T]
        return torch.stft(x, n_fft=64, return_complex=True).abs()
