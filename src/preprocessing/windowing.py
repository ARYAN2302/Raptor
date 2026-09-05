import numpy as np
def window_iq(iq, win, hop):
    T,E,C=iq.shape
    out=[]
    for s in range(0, max(1,T-win+1), hop):
        w=iq[s:s+win]
        if w.shape[0]<win:
            pad=np.zeros((win-w.shape[0],E,C),dtype=iq.dtype)
            w=np.concatenate([w,pad],axis=0)
        out.append(w)
    return np.stack(out) if out else np.zeros((0,win,E,C),dtype=iq.dtype)
