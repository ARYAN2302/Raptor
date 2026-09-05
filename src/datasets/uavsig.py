import pathlib, numpy as np
from ..dataio.canonical import CanonicalSample
def load_uavsig_bins(root: str, max_files=2):
    root=pathlib.Path(root)
    samples=[]
    for p in sorted(root.glob("*.bin"))[:max_files]:
        raw=np.fromfile(str(p), dtype=np.int16)
        iq=np.stack([raw[0::2], raw[1::2]], axis=-1).astype(np.float32)/32768.0
        iq=iq[:1_000_000][:,None,:].astype(np.float32)
        samples.append(CanonicalSample(iq=iq, sample_rate=56e6, center_frequency=2.4e9, bandwidth=20e6, source_dataset="uavsig", capture_id=p.stem, site_id=p.stem))
    return samples
