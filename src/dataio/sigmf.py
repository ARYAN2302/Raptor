"""SigMF helpers — read .sigmf-meta + .sigmf-data coherently."""
import json, pathlib
import numpy as np
def read_sigmf(prefix: str):
    p=pathlib.Path(prefix)
    meta=json.loads(open(str(p)+".sigmf-meta").read())
    data=np.fromfile(str(p)+".sigmf-data", dtype=np.complex64)
    sr=meta.get("global",{}).get("core:sample_rate", 1e6)
    cf=meta.get("captures",[{}])[0].get("core:frequency", 0)
    return data, meta, sr, cf
