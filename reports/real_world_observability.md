# Real-World Observability — Passive RF IQ → Range/Az/El/Velocity

## Table per Instruction

| Quantity  | Real data used | Ground truth available | Result | Verdict |
|---|---|---|---|---|
| Azimuth   | RFUAV 2-channel? No — RFUAV E=1, UAVSig E=1, AERPAW 12 dual-channel B210 2 channels per SigMF but coherence NOT verified (manifest aerpaw12), AERPAW 19 single-channel | UAV lat/lon/alt via GPS, receiver LW1-LW5 fixed, but B210 phase coherence not verified → no reliable multi-element phase | Az RMSE 94.5° (with geometry) vs 99.8° without (synthetic E=4, 32 val, 20 iters) — sanity not real, real multi-element data not yet phase-coherent verified | **INCONCLUSIVE** — no real coherent multi-element IQ with verified array geometry + ground truth yet |
| Elevation | Same as azimuth — requires 3D geometry, AERPAW 12 has 3 altitudes (40/70/100m) but single B210 dual-channel not verified for elevation (needs 2D array) | 3D position via GPS, but elevation from single B210 not demonstrable without 2D array | Elevation not yet measured on real coherent data (synthetic el error not yet run) | **INCONCLUSIVE** — no real 3D array data with verified elevation |
| Range     | AERPAW 12 has 5 nodes LW1-LW5 + UAV trajectory, but no direct range label — must compute from receiver + UAV pos per Step2; AERPAW 19 has tx/rx lat/lon/alt + velocity per capture, 244MB not yet downloaded due to Anubis 403, synthetic range RMSE single 443m vs temporal 401m (invalid) → honest continuous 245m vs 202m (synthetic) | Range must be computed ENU per src/utils/coordinates.py, not provided directly; real range not yet trained on AERPAW 12 (150GB not yet downloaded, Anubis) | Synthetic range RMSE 202m (temporal) on 100-800m uniform (15% error) — not real passive RF, real range from IQ over time not yet demonstrated on AERPAW 12/19 | **INCONCLUSIVE** — no real passive RF range with temporal/multipath/environment proven; synthetic shows 200m error even with temporal |
| Velocity  | AERPAW 19 has velocity X/Y/Z per capture + timestamps, AERPAW 12 has GPS 1Hz + radio 20ms/100ms interpolated to mX,mY,mZ per example.py | Velocity available per capture in 19, and via compute_velocity in 12 | Velocity not yet measured on real RF (synthetic Doppler fd used, but real RF velocity not yet trained) | **INCONCLUSIVE** — velocity requires consecutive time windows + accurate trajectory, not yet run on real |

## Final Conclusion

**Which have actually been demonstrated from real passive RF observations?**

- **Azimuth:** INCONCLUSIVE — no real coherent multi-element IQ with verified phase coherence + ground truth yet (RFUAV/UAVSig E=1, AERPAW 12 B210 dual-channel not verified).
- **Elevation:** INCONCLUSIVE — same as azimuth, plus needs 3D array.
- **Range:** INCONCLUSIVE — synthetic 200m error even with temporal, real AERPAW 12/19 range from IQ over time not yet trained due to 150GB download + Anubis block; no evidence for passive single-snapshot range.
- **Velocity:** INCONCLUSIVE — synthetic Doppler used, but real consecutive windows + trajectory velocity not yet measured on AERPAW 19/12.

**All four remain unproven from real passive RF IQ with the required E≥2 coherent array + synchronized ground truth.** Next is to download AERPAW 12/19 via manual browser (Anubis) and train Step3 one-emitter on real dual-channel with verified coherence per Step1 manifest.

