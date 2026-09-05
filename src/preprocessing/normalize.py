import numpy as np
def normalize_iq(iq, mode="per_window", eps=1e-8):
    if mode=="per_window":
        rms=np.sqrt((iq**2).mean(axis=(0,2), keepdims=True))+eps
        return iq/rms
    elif mode=="none": return iq
    else: raise ValueError(mode)
