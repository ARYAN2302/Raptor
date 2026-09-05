"""Canonical IQ §7 — preserves metadata, SigMF compatible, no silent resample."""
from __future__ import annotations
import json, hashlib, pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

@dataclass
class CanonicalSample:
    iq: np.ndarray  # [T,E,2] float32 I/Q
    sample_rate: float
    center_frequency: float
    bandwidth: float
    antenna_positions: Optional[np.ndarray]=None  # [E,3]
    antenna_orientations: Optional[np.ndarray]=None
    receiver_pose: Optional[np.ndarray]=None
    source_dataset: str="synthetic"
    capture_id: str=""
    site_id: str="site0"
    snr_estimate: Optional[float]=None
    timestamp_start: float=0.0
    emitter_count: int=0
    emitters: list=field(default_factory=list)
    extra: dict=field(default_factory=dict)
    def hash(self)->str:
        h=hashlib.sha256()
        h.update(self.iq.tobytes()[:1<<20])
        h.update(str(self.sample_rate).encode())
        return h.hexdigest()[:12]

def load_iq(path: str, sample_rate=100e6, center_freq=2.4e9, bandwidth=100e6)->CanonicalSample:
    p=pathlib.Path(path)
    if p.suffix==".npy":
        arr=np.load(str(p))
        if np.iscomplexobj(arr):
            iq=np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)
        else:
            iq=arr.astype(np.float32)
            if iq.ndim==2 and iq.shape[-1]==2: pass
            elif iq.ndim==1: iq=np.stack([iq[::2], iq[1::2]], axis=-1)
    elif p.suffix==".bin":
        raw=np.fromfile(str(p), dtype=np.int16)
        iq=np.stack([raw[0::2], raw[1::2]], axis=-1).astype(np.float32)/32768.0
        iq=iq[:,None,:]
    elif p.suffix==".iq":
        raw=np.fromfile(str(p), dtype=np.complex64)
        iq=np.stack([raw.real, raw.imag], axis=-1).astype(np.float32)[:,None,:]
    else:
        raise ValueError(f"unknown suffix {p.suffix}")
    if iq.ndim==2: iq=iq[:,None,:]
    return CanonicalSample(iq=iq, sample_rate=sample_rate, center_frequency=center_freq, bandwidth=bandwidth, source_dataset=p.parent.name, capture_id=p.stem)

def save_manifest(samples: list[CanonicalSample], path: str):
    import pathlib, json, hashlib
    out=[]
    for s in samples:
        d=asdict(s)
        d["iq"]=f"{s.iq.shape} {s.iq.dtype}"
        d["iq_hash"]=s.hash()
        for k in ["antenna_positions","antenna_orientations","receiver_pose"]:
            if d[k] is not None:
                try: d[k]=np.asarray(d[k]).tolist()
                except: pass
        out.append(d)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f: json.dump(out,f,indent=2)
