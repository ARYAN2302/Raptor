import numpy as np
def normalize_iq(iq: np.ndarray, mode="per_window", eps=1e-8):
    """iq [T,E,2] -> normalized; preserves phase. Never silently resample."""
    if mode=="per_window":
        # normalize by RMS per antenna, same gain for I/Q
        rms=np.sqrt((iq**2).mean(axis=(0,2), keepdims=True))+eps  # [1,E,1]
        return iq/rms
    elif mode=="none": return iq
    else: raise ValueError(mode)

def estimate_snr_db(iq):
    sig=np.mean(iq**2)
    return 10*np.log10(sig+1e-12)
