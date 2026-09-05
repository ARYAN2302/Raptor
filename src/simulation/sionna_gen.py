"""Sionna bridge §8 — exact GT schema."""
from dataclasses import dataclass
import numpy as np
from ..dataio.canonical import CanonicalSample
@dataclass
class SionnaConfig:
    carrier_freq: float=2.4e9
    sample_rate: float=100e6
    bandwidth: float=20e6
    n_ant: int=4
@dataclass
class Scene:
    sample_id: str
    site_id: str
    array_geometry: np.ndarray
    iq: np.ndarray
    emitters: list
def generate_scene(cfg: SionnaConfig, seed=0, n_emitters=1):
    from ..datasets.synthetic import synth_iq
    s=synth_iq(T=512,E=cfg.n_ant,sample_rate=cfg.sample_rate,center_freq=cfg.carrier_freq,n_emitters=n_emitters,seed=seed)
    return Scene(sample_id=f"syn_{seed}", site_id=f"site_{seed%3}", array_geometry=s.antenna_positions, iq=s.iq, emitters=s.emitters)
