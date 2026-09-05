"""Array geometry encoding §4.4 — element coords -> embeddings, consistent identity."""
import torch, torch.nn as nn
class ArrayEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.proj=nn.Linear(3, d_model)
    def forward(self, antenna_positions, tokens):
        # antenna_positions [B,E,3] -> [B,E,D] -> broadcast to tokens per antenna?
        # Simple: add mean geometry emb to latents (full impl conditions cross-attn)
        if antenna_positions is None: return tokens
        geo=self.proj(antenna_positions).mean(dim=1, keepdim=True)  # [B,1,D]
        return tokens+geo
