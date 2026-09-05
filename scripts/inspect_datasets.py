#!/usr/bin/env python3
import sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.datasets.rfuav import load_rfuav_dir
from src.datasets.uavsig import load_uavsig_bins
from src.datasets.synthetic import synth_iq
print("synthetic", synth_iq(T=512,E=1,seed=0).iq.shape)
try:
    r=load_rfuav_dir("/tmp/raptor_build/Raptor/data_manifest_dummy", max_files=0)
    print("rfuav dummy", len(r))
except Exception as e: print(e)
print("inspect ok — full inventory requires mounted /data per reports/dataset_audit.md")
