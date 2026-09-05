#!/usr/bin/env python3
"""Phase 0-1 dataset audit — respects essential-only download (§5)."""
import pathlib, sys
sys.path.insert(0,"src")
from src.datasets.rfuav import load_rfuav_dir
from src.datasets.uavsig import load_uavsig_bins
from src.datasets.synthetic import synth_iq
from src.dataio.canonical import save_manifest

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--rfuav-root", default="/data/rfuav/DJI FPV COMBO/DJI FPV COMBO")
    p.add_argument("--iq-root", default="/iq")
    p.add_argument("--out", default="data/manifests/audit.json")
    a=p.parse_args()
    print("=== RAPTOR Dataset Audit (essential only) ===")
    # synthetic sanity
    s=synth_iq(T=4096,E=4,n_emitters=2,seed=0)
    print(f"synthetic: {s.iq.shape} emitters={s.emitters}")
    # try real if mounted
    try:
        rfuav=load_rfuav_dir(a.rfuav_root, max_files=2)
        print(f"rfuav: {len(rfuav)} samples from {a.rfuav_root}")
        for x in rfuav[:1]: print(f"  {x.capture_id} {x.iq.shape} sr={x.sample_rate}")
    except Exception as e: print(f"rfuav not mounted ({e}) — using synthetic for QoL")
    try:
        uavsig=load_uavsig_bins(a.iq_root, max_files=2)
        print(f"uavsig: {len(uavsig)} from {a.iq_root}")
    except Exception as e: print(f"uavsig not mounted ({e})")
    print("audit ok — no bulk download triggered (raptor-data rars stay archived)")

if __name__=="__main__": main()
