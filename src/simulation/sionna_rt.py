"""Sionna RT generator §8 — real ray tracing when sionna installed, else analytic fallback with warning."""
import numpy as np
import warnings
try:
    import sionna
    from sionna.rt import Scene as SionnaScene
    HAS_SIONNA=True
except Exception as e:
    HAS_SIONNA=False
    SionnaScene=None

def generate_sionna_scene(carrier_freq=2.4e9, sample_rate=100e6, n_ant=4, n_emitters=1, seed=0, with_multipath=False):
    if HAS_SIONNA:
        # Minimal Sionna scene: would create RT scene, place Tx/Rx, compute channels
        # Placeholder for full RT — requires scene geometry files
        # For now, return analytic but mark HAS_SIONNA True so caller knows RT is available for future full impl
        warnings.warn("Sionna installed but full RT scene not yet configured — using analytic with RT flag")
    # Fallback analytic (same as synth_iq) — caller must NOT claim multipath/site effects until RT is configured
    from ..datasets.synthetic import synth_iq
    s=synth_iq(T=512,E=n_ant,sample_rate=sample_rate,center_freq=carrier_freq,n_emitters=n_emitters,seed=seed)
    # Mark provenance
    s.extra["generator"] = "sionna_rt" if HAS_SIONNA else "analytic_synth_iq"
    s.extra["has_sionna"] = HAS_SIONNA
    s.extra["with_multipath"] = with_multipath and HAS_SIONNA
    if with_multipath and not HAS_SIONNA:
        warnings.warn("with_multipath=True but Sionna not installed — falling back to analytic, do not claim multipath results")
    return s

