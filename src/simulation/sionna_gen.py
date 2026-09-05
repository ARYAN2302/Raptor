"""Sionna bridge §8 — delegates to sionna_rt.py for honest provenance."""
from .sionna_rt import generate_sionna_scene, HAS_SIONNA
from dataclasses import dataclass
import numpy as np
@dataclass
class SionnaConfig:
    carrier_freq: float=2.4e9
    sample_rate: float=100e6
    bandwidth: float=20e6
    n_ant: int=4
def generate_scene(cfg: SionnaConfig, seed=0, n_emitters=1, with_multipath=False):
    from .sionna_rt import generate_sionna_scene
    return generate_sionna_scene(carrier_freq=cfg.carrier_freq, sample_rate=cfg.sample_rate, n_ant=cfg.n_ant, n_emitters=n_emitters, seed=seed, with_multipath=with_multipath)
