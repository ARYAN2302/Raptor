# Dataset Audit — Phase 0 (essential only, no bulk re-download)

**Date:** 2026-09-05 · **Volumes:** `raptor-data`, `iris-raw-iq`, `iris-data`

## RFUAV (Dataset A) — representation pretraining
- **Repo:** https://github.com/kitoweeknd/RFUAV
- **Volume:** `raptor-data:/rfuav` + `/rfuav_rar`
- **Extracted:** `DJI FPV COMBO / VTSBW=10|20|40` → `pack*.iq` 800 MB/chunk, `pack*.xml` (SignalHoundIQFile, USRPX310, 100 Msps, 5.76 GHz, Complex Float, ScaleFactor 90)
- **Archived:** 11× `.rar` (14303 MB DJI FPV COMBO.rar etc., ValidationSet 36-41 GB) — **not extracted yet** (essential-only rule). Enough for P2 POC with single drone; will extract one more drone (DJI MINI4 PRO) only when cross-device generalization (Q5) is measured.
- **GT:** drone class/serial/SNR/center_freq/sample_rate/bandwidth per XML. **No range/az/el/velocity labels** — confirms Handoff Q6: no public raw IQ → state supervision.
- **Leakage trap:** serial number = device identity — split by `serial` not random windows.

## UAVSig (Dataset B) — identity/counting eval
- **Page:** https://cores.ee.ucla.edu/downloads/datasets/uavsig/
- **Volume:** `iris-raw-iq:/` 8× 457 MB `.bin` (int16 interleaved I/Q, ~56 Msps bins): DJI_inspire/mavic_mini/mavic_pro/phantom_4, Parrot_mambo/disco, Yuneec_typhoon
- **Plus:** `iris-data:/iris_rfuav.h5` 12.4 GB preprocessed — used for Phase 3 transfer tests.
- **Leakage:** session/device confound — must split by capture file, not window.

## AERPAW-28 (Dataset C) — state eval / sim2real
- **Page:** https://aerpaw.org/dataset/multi-modal-rf-sensor-and-radar-dataset-for-uav-tracking/ (33 flights, Fortem radar + Keysight RF + UAV ground truth)
- **Finding:** provides **processed** RF/radar position estimates + UAV ground truth, **not raw coherent single-array IQ** for passive C-UAS. No claim of `raw IQ → range/az/el/velocity` supervision (Handoff §3 Q6). Useful for sim2real gap measurement, not training.
- **AERPAW-31:** air-to-air channel sounding SigMF at 3.4 GHz — raw IQ + geometry but not drone-comms localization. Deferred to Phase 6+ (Optional D).

## AERPAW-8 / Others
- 4-sensor TDOA passive localization — different measurement (TDOA, not single-array coherent IQ). Reference baseline only.

## Decision
- **Do not bulk-download RFUAV (~110 GB rars) for Phase 2.** Single-drone POC + synthetic mixtures sufficient to prove representation transfer (Gate A). Essential fetch list = already satisfied. Next fetch gate = Gate B/C: extract second drone + AERPAW-28 ground truth CSV only.

## Canonical Format (§7)
```
sample: iq [T,E,2] float32, sample_rate, center_freq, bandwidth, antenna_positions [E,3], site_id, emitter_count, emitters[{range,az,el,velocity_xyz,radial_velocity}]
```
Stored via `src/io/canonical.py:1`, SigMF compat `src/io/sigmf.py:1`. Never silently resample.
