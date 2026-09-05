"""Reconstruction decoder §7 Task1 — Perceiver latent → masked token reconstruction."""
import torch, torch.nn as nn
class ReconDecoder(nn.Module):
    def __init__(self, d_model=32, n_heads=2, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, latents, masked_tokens, mask):
        # latents [B,M,D], masked_tokens [B,L,D] with zeros at masked positions, mask [B,L] bool
        # Use learnable mask tokens + cross-attend to latents
        B,L,D = masked_tokens.shape
        # simple: decode via cross-attention where queries are masked positions
        # For smoke, we reconstruct all tokens from latents via decoder
        q = masked_tokens  # [B,L,D] with zeros at masked
        x = q
        for lyr in self.layers:
            x = lyr(x, latents)
        recon = self.proj(x)  # [B,L,D]
        return recon
