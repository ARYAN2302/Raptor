"""Radio-FM-inspired tokenizer §2.1/4.2 — preserves I/Q, time, antenna identity+geometry."""
import torch, torch.nn as nn
class ComplexIQTokenizer(nn.Module):
    def __init__(self, patch=8, stride=8, d_model=64, max_antennas=8):
        super().__init__()
        self.patch=patch; self.stride=stride; self.d_model=d_model
        # separate I/Q conv per antenna (shared weights, per-element tokens)
        self.proj_I = nn.Conv1d(1, d_model//2, kernel_size=patch, stride=stride, padding=0)
        self.proj_Q = nn.Conv1d(1, d_model//2, kernel_size=patch, stride=stride, padding=0)
        self.norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.randn(1,16384,d_model)*0.02)
        self.ant_emb = nn.Parameter(torch.randn(1,max_antennas,1,d_model)*0.02)
    def forward(self, iq, antenna_positions=None):
        # iq [B,T,E,2] -> preserves E
        B,T,E,C = iq.shape
        assert C==2, "last dim must be I/Q"
        # per-antenna I/Q separate
        iq_e = iq.permute(0,2,3,1)  # [B,E,2,T]
        I = iq_e[:,:,0,:]  # [B,E,T]
        Q = iq_e[:,:,1,:]  # [B,E,T]
        # apply conv per antenna via reshape B*E
        I_t = self.proj_I(I.reshape(B*E,1,T)).transpose(1,2)  # [B*E,L,D/2]
        Q_t = self.proj_Q(Q.reshape(B*E,1,T)).transpose(1,2)  # [B*E,L,D/2]
        x = torch.cat([I_t, Q_t], dim=-1).view(B, E, -1, self.d_model)  # [B,E,L,D]
        L = x.shape[2]
        x = x.reshape(B, E*L, self.d_model)
        x = self.norm(x)
        # time pos (shared)
        x = x + self.pos[:,:E*L,:]
        # antenna identity (per-element, not collapsed)
        # self.ant_emb [1,max_ant,1,D] -> [E,D] -> [E*L,D] -> [B,E*L,D]
        ant_base = self.ant_emb[0, :E, 0, :]  # [E,D]
        ant = ant_base.repeat_interleave(L, dim=0)  # [E*L,D]
        x = x + ant.unsqueeze(0).expand(B, -1, -1)
        # optional geometry conditioning will be added by ArrayEncoder, but identity preserved here
        return x  # [B, E*L, D]
