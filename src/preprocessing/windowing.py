import numpy as np
def window_iq(iq: np.ndarray, win: int, hop: int):
    T,E,C=iq.shape
    out=[]
    for s in range(0, T-win+1, hop):
        out.append(iq[s:s+win])
    return np.stack(out) if out else np.zeros((0,win,E,C), dtype=iq.dtype)
