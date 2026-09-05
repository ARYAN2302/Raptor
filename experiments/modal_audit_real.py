import modal
app = modal.App("raptor-audit-real")
image = modal.Image.debian_slim().pip_install("numpy","scipy").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
iq_vol = modal.Volume.from_name("iris-raw-iq", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol, "/iq": iq_vol}, timeout=3600)
def audit():
    import pathlib, numpy as np, xml.etree.ElementTree as ET, os
    print("=== RFUAV actual files ===")
    for root in ["/data/rfuav/DJI FPV COMBO/DJI FPV COMBO", "/data/rfuav/DJI MINI4 PRO/DJI MINI4 PRO"]:
        p=pathlib.Path(root)
        print(f"\n-- {root} exists {p.exists()}")
        if not p.exists(): continue
        for xml in sorted(p.rglob("*.xml"))[:2]:
            print(f"xml {xml}")
            t=ET.parse(xml).getroot()
            for tag in ["DeviceType","Drone","SerialNumber","DataType","CenterFrequency","SampleRate","IFBandwidth","ScaleFactor","SampleCount"]:
                e=t.find(tag)
                print(f"  {tag}: {e.text if e is not None else None}")
            # check iq file
            iq_path = list(xml.parent.glob("*.iq"))[:1]
            if iq_path:
                iq_path=iq_path[0]
                raw=np.fromfile(str(iq_path), dtype=np.complex64)
                print(f"  iq {iq_path.name} {raw.shape} dtype {raw.dtype} first 2 {raw[:2]}")
                # check channels: RFUAV is single antenna per file
                print(f"  channels/elements: 1 (per file), sample_rate from xml, bandwidth, center freq")
                # timestamps: not in xml, infer from SampleCount / SampleRate
                try:
                    sr=float(t.find('SampleRate').text)
                    sc=float(t.find('SampleCount').text)
                    print(f"  duration {sc/sr:.2f}s")
                except: pass
        # count total .iq
        iqs=list(p.rglob("*.iq"))
        print(f"total .iq under {root}: {len(iqs)} total size {sum(x.stat().st_size for x in iqs)/1e9:.2f} GB")
    print("\n=== UAVSig actual files ===")
    p=pathlib.Path("/iq")
    for f in sorted(p.glob("*.bin"))[:3]:
        print(f"bin {f.name} size {f.stat().st_size/1e6:.1f} MB")
        raw=np.fromfile(str(f), dtype=np.int16)
        print(f"  int16 {raw.shape} -> IQ pairs {raw.shape[0]//2} first I {raw[0:5]} Q {raw[1:10:2][:5]}")
        # check for gaps: file is continuous int16, gaps not visible without labels
        print(f"  sample_rate: not in file, need dataverse metadata (WHIRLS labels separate)")
    print("\n=== UAVSig labels (if any on volume) ===")
    for f in p.rglob("*.csv"):
        print(f)
    print("\n=== AERPAW check (no raw IQ expected) ===")
    print("AERPAW-28 Dryad not on Modal volumes, need web inventory per audit")
    # Check antenna geometry: RFUAV single channel, no array — need Sionna for multi-antenna
    print("\n=== Antenna geometry ===")
    print("RFUAV: E=1 per file (no array), UAVSig: E=1 (single B205mini)")
    print("Multi-antenna requires Sionna synthetic per §8")
