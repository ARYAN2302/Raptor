"""Full RAPTOR §4 — composes all stages, same code path small or full."""
import torch, torch.nn as nn
from .iq_tokenizer import ComplexIQTokenizer
from .array_encoder import ArrayEncoder
from .perceiver import PerceiverBottleneck
from .temporal_mamba import TemporalMamba
from .set_decoder import SetDecoder
class RAPTOR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m=cfg.get("model", cfg)
        self.tok=ComplexIQTokenizer(in_antennas=m.get("antennas",1), patch=m.get("patch",8), stride=m.get("stride",8), d_model=m.get("d_model",64))
        self.arr=ArrayEncoder(d_model=m.get("d_model",64))
        self.perc=PerceiverBottleneck(d_model=m.get("d_model",64), n_latent=m.get("n_latent",32), n_heads=m.get("n_heads",4), n_layers=m.get("perceiver_layers",1))
        self.temp=TemporalMamba(d_model=m.get("d_model",64), use_mamba=(m.get("temporal","")== "mamba"))
        self.dec=SetDecoder(d_model=m.get("d_model",64), n_queries=m.get("n_queries",4), n_heads=m.get("n_heads",4))
    def forward(self, iq, antenna_positions=None, state=None):
        t=self.tok(iq)
        t=self.arr(antenna_positions, t)
        z=self.perc(t)
        z, ns=self.temp(z, state)
        out=self.dec(z)
        out["latents"]=z; out["tokens"]=t
        return out, ns
    def forward_recon(self, iq, mask_ratio=0.6):
        t=self.tok(iq)
        B,L,D=t.shape
        mask=torch.rand(B,L, device=t.device)<mask_ratio
        tm=t.clone(); tm[mask]=0
        z=self.perc(tm)
        return {"tokens": t, "latents": z, "mask": mask}
