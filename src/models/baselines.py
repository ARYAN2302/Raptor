"""Baselines §10 A-F."""
import torch, torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, in_ant=4, d=64, n_queries=4):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(in_ant*2, 64, 7, padding=3), nn.ReLU(), nn.Conv1d(64,64,7,padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head=nn.Linear(64, n_queries*7)
        self.nq=n_queries
    def forward(self,iq):
        B,T,E,C=iq.shape
        x=iq.permute(0,2,3,1).reshape(B,E*C,T)
        f=self.net(x).squeeze(-1)
        o=self.head(f).view(B,self.nq,7)
        return o

class MagnitudeBaseline(nn.Module):
    def forward(self,iq): return torch.sqrt((iq**2).sum(-1))  # magnitude-only for ablation

class SpectrogramBaseline(nn.Module):
    def forward(self,iq):
        # magnitude STFT placeholder
        return torch.stft(iq[...,0].mean(-1), n_fft=256, return_complex=True)
