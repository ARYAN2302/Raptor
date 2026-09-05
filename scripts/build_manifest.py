#!/usr/bin/env python3
import sys
sys.path.insert(0,"src")
from src.datasets.synthetic import synth_iq
from src.dataio.canonical import save_manifest
import numpy as np
def main():
    samples=[synth_iq(T=4096,E=4,n_emitters=int(np.random.randint(0,3)), seed=i) for i in range(32)]
    save_manifest(samples, "data/manifests/synth.json")
    print("manifest saved 32 synthetic")
if __name__=="__main__": main()
