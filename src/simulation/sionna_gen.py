"""Sionna bridge §6-7 — controlled GT scenes. Falls back to numpy synthetic if sionna unavailable."""
from dataclasses import dataclass
import numpy as np
from ..dataio.canonical import CanonicalSample

@dataclass
class SionnaConfig:
    carrier_freq: float=2.4e9
    sample_rate: float=100e6
    bandwidth: float=20e6
    n_ant: int=4
    max_emitters: int=2

def generate_scene(cfg: SionnaConfig, seed=0, n_emitters=1):
    # Try Sionna RT if available, else numpy steering model (same as synthetic)
    try:
        import sionna
        # placeholder: would build scene, ray trace, return IQ
        raise ImportError
    except Exception:
        from ..datasets.synthetic import synth_iq
        return synth_iq(T=4096, E=cfg.n_ant, sample_rate=cfg.sample_rate, center_freq=cfg.carrier_freq, n_emitters=n_emitters, seed=seed)
