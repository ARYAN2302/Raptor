"""RFUAV loader — reads existing raptor-data:/rfuav (essential only, no bulk re-download)."""
import pathlib, xml.etree.ElementTree as ET
import numpy as np
from ..dataio.canonical import CanonicalSample

def parse_xml(xml_path):
    t=ET.parse(xml_path).getroot()
    def txt(tag): 
        e=t.find(tag); return e.text if e is not None else None
    return dict(sample_rate=float(txt("SampleRate") or 100e6), center_frequency=float(txt("CenterFrequency") or 2.4e9), bandwidth=float(txt("IFBandwidth") or 100e6), drone=txt("Drone"), serial=txt("SerialNumber"))

def load_rfuav_dir(root: str, max_files=8):
    root=pathlib.Path(root)
    samples=[]
    for xml in sorted(root.rglob("*.xml"))[:max_files]:
        meta=parse_xml(str(xml))
        # associated .iq shards
        base=xml.parent
        for iq_path in sorted(base.glob("*.iq"))[:2]:
            raw=np.fromfile(str(iq_path), dtype=np.complex64)
            if raw.size>1_000_000:
                raw=raw[:1_000_000]  # cap for POC
            iq=np.stack([raw.real, raw.imag], axis=-1).astype(np.float32)[:,None,:]
            iq=np.repeat(iq, 1, axis=1)  # single antenna for RFUAV (preserve call)
            # RFUAV is single-antenna; we keep E=1
            samples.append(CanonicalSample(iq=iq, sample_rate=meta["sample_rate"], center_frequency=meta["center_frequency"], bandwidth=meta["bandwidth"], source_dataset="rfuav", capture_id=f"{meta['drone']}_{iq_path.stem}", site_id=meta["serial"] or "rfuav", extra=meta))
    return samples
