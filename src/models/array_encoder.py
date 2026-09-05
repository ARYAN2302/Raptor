"""Array geometry encoding §4.4 — per-element explicit, E arbitrary."""
import torch, torch.nn as nn
class ArrayEncoder(nn.Module):
    def __init__(self, d_model=64, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, d_model))
    def forward(self, antenna_positions, tokens):
        # tokens [B, E*L, D] from tokenizer; antenna_positions [B,E,3] or None
        if antenna_positions is None:
            return tokens
        B, EL, D = tokens.shape
        E = antenna_positions.shape[1]
        L = EL // E
        # per-element geometry [B,E,D]
        geo = self.mlp(antenna_positions)  # [B,E,D]
        # broadcast to L tokens per element
        geo_exp = geo.repeat_interleave(L, dim=1)  # [B, E*L, D] (repeats each element L times in order)
        # tokenizer order is E*L with E major, L minor (reshape B,E,L,D -> B,E*L) so repeat_interleave matches
        return tokens + geo_exp
