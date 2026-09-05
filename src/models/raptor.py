"""RAPTOR full forward §2."""
import torch, torch.nn as nn
from .iq_tokenizer import ComplexIQTokenizer
from .perceiver import PerceiverBottleneck
from .temporal import TemporalStateModel
from .set_decoder import SetDecoder

class RAPTOR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m=cfg.get("model", cfg)
        self.tokenizer = ComplexIQTokenizer(in_antennas=m.get("antennas",4), patch=m.get("patch",64), stride=m.get("stride",32), d_model=m.get("d_model",128))
        self.perceiver = PerceiverBottleneck(d_model=m.get("d_model",128), n_latent=m.get("n_latent",64), n_heads=m.get("n_heads",4), n_layers=m.get("perceiver_layers",2))
        self.temporal = TemporalStateModel(d_model=m.get("d_model",128), use_mamba=m.get("use_mamba",False))
        self.decoder = SetDecoder(d_model=m.get("d_model",128), n_queries=m.get("n_queries",4), n_heads=m.get("n_heads",4))
        # recon head for masked pretraining
        self.recon = nn.Linear(m.get("d_model",128), m.get("antennas",4)*2*m.get("patch",64))

    def forward(self, iq, antenna_geom=None, seq_len=1):
        # iq: [B,T,E,2]
        tok = self.tokenizer(iq)  # [B,N,D]
        lat = self.perceiver(tok, antenna_geom)
        # temporal expects [B, S, L, D] if seq>1; here single step
        lat = self.temporal(lat, seq_len=seq_len)
        dec = self.decoder(lat)
        return {"tokens": tok, "latents": lat, **dec}

    def forward_recon(self, iq, mask_ratio=0.4):
        tok = self.tokenizer(iq)
        B,N,D=tok.shape
        mask = torch.rand(B,N, device=tok.device) < mask_ratio
        tok_masked = tok.clone()
        tok_masked[mask]=0
        lat = self.perceiver(tok_masked)
        # reconstruct masked tokens via linear head on latents (broadcast)
        # simple: average latent -> project -> MSE vs original masked tokens
        # for POC, decode from first latent slot
        recon = self.recon(lat[:,0,:])  # [B, E*2*patch] -> not aligned; use MLP to predict token MSE
        # Instead predict token-level recon by cross-attn? simplified: use lat->token proj
        # We'll compute loss against masked tokens via small MLP
        return {"mask": mask, "tokens": tok, "latents": lat, "recon": recon}
