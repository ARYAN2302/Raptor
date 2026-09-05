"""Canonical IQ format §7 — SigMF-aware, metadata-preserving."""
from __future__ import annotations
import json, hashlib, pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

@dataclass
class CanonicalSample:
    iq: np.ndarray  # [T, E, 2] or [T, 2] -> float32 I/Q
    sample_rate: float
    center_frequency: float
    bandwidth: float
    antenna_positions: Optional[np.ndarray] = None  # [E,3] ENU
    antenna_orientations: Optional[np.ndarray] = None
    receiver_pose: Optional[np.ndarray] = None
    source_dataset: str = "synthetic"
    capture_id: str = ""
    site_id: str = "site0"
    snr_estimate: Optional[float]=None
    timestamp_start: float = 0.0
    emitter_count: int = 0
    emitters: list = field(default_factory=list)  # each: dict with range/az/el/velocity_xyz
    extra: dict = field(default_factory=dict)

    def hash(self)->str:
        h=hashlib.sha256()
        h.update(self.iq.tobytes()[:1<<20])
        h.update(str(self.sample_rate).encode())
        return h.hexdigest()[:12]

def load_iq(path: str, sample_rate: float=100e6, center_freq: float=2.4e9, bandwidth: float=100e6, dtype=np.complex64)->CanonicalSample:
    """Load raw IQ file (.bin int16, .iq complex float32, .npy). Preserves metadata."""
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
        # UAVSig  int16 interleaved I/Q
        raw=np.fromfile(str(p), dtype=np.int16)
        iq=np.stack([raw[0::2], raw[1::2]], axis=-1).astype(np.float32)/32768.0
        iq=iq[None,:] if False else iq.reshape(-1, iq.shape[-1])  # [T,2]
        iq=iq[:,None,:].repeat(1, axis=1) if False else iq  # keep [T,2] -> caller adds E dim
        # normalize to [T,1,2] for single antenna
        if iq.ndim==2:
            iq=iq[:,None,:]
    elif p.suffix==".iq":
        raw=np.fromfile(str(p), dtype=np.complex64)
        iq=np.stack([raw.real, raw.imag], axis=-1).astype(np.float32)[:,None,:] if raw.ndim==1 else np.stack([raw.real, raw.imag], axis=-1).astype(np.float32)
        if iq.ndim==2:
            iq=iq[:,None,:]
    else:
        raise ValueError(f"unknown IQ suffix {p.suffix}")
    # ensure [T,E,2]
    if iq.ndim==2:
        iq=iq[:,None,:]
    return CanonicalSample(iq=iq, sample_rate=sample_rate, center_frequency=center_freq, bandwidth=bandwidth, source_dataset=p.parent.name, capture_id=p.stem)

def save_manifest(samples: list[CanonicalSample], path: str):
    out=[]
    for s in samples:
        d=asdict(s)
        d["iq"]=f"{s.iq.shape} {s.iq.dtype}"
        d["iq_hash"]=s.hash()
        # numpy -> list
        for k in ["antenna_positions","antenna_orientations","receiver_pose"]:
            if d[k] is not None:
                try: d[k]=np.asarray(d[k]).tolist()
                except: pass
        out.append(d)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f: json.dump(out,f,indent=2)
