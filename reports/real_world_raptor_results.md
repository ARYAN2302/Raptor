# Real-World RAPTOR Results — AERPAW 12/19/31/32 Supervised

## Table per Step10

| Quantity        | Dataset | Test setup | Error | Baseline | Verdict |
|---|---|---|---|---:|---:|---|
| Range           | AERPAW 12 (intended) | Train LW1-3 → Test LW4-5, flight/trajectory split, 2-channel B210 SigMF 5/10/20MHz, 1.25/2.5/5MHz BW, 40/70/100m alt (150GB not yet downloaded, Anubis 403) — synthetic placeholder E=2, 512-T, 1 emitter, 32 val | Synthetic range RMSE single 245m → temporal 202m (continuous trajectory, 4 windows, 100-800m) | Trivial prior 430m (uniform) | **INCONCLUSIVE** (real not yet trained, synthetic 200m) |
| Azimuth         | AERPAW 12 (intended) | Same as range, dual-channel, need phase-coherent array | Synthetic az RMSE with geometry 94.5° vs without 99.8° (20 iters, 32 val) — real B210 coherence NOT verified | Prior 103° (uniform) | **INCONCLUSIVE** (real coherence not verified) |
| Elevation       | AERPAW 12/19 (intended) | 3D pos via GPS + receiver, need 3D array | Not yet measured on real (synthetic el not run, single B210 not 2D) | — | **INCONCLUSIVE** |
| Radial velocity | AERPAW 19 (intended) | 9 flights, 56MHz Zadoff-Chu, tx/rx lat/lon/alt + velocity X/Y/Z per capture, GNSSDO PPS sync | Not yet measured on real (synthetic fd used) | — | **INCONCLUSIVE** |
| 3D velocity     | AERPAW 19 (intended) | Same as radial, compute via compute_velocity | Not yet measured | — | **INCONCLUSIVE** |

## Final Conclusion

**Can real passive RF IQ support recovery of range, azimuth, elevation, and velocity?**

- **No quantity has been demonstrated from real passive RF IQ with the required coherent multi-channel + synchronized ground truth.** All results above are synthetic placeholders (range 202m, az 94.5°) or unverified (B210 dual-channel coherence not assumed per Step1 manifest). AERPAW 12 (150GB, 5 nodes, 2 channels, 20ms/100ms), 19 (244MB, 9 flights, velocity), 31/32 (air-to-air/air-to-ground 3.4GHz) are audited in `data/manifests/aerpaw*.json` with IQ↔telemetry sync verified via example.py interpolation, but **IQ files not yet downloaded due to Dryad Anubis 403 + 150GB size** — requires manual browser download per Step1.

**Next:** Manual download of AERPAW 12/19 via Dryad (bypass Anubis) + verify B210 phase coherence via SigMF `core:hw_info` before claiming spatial inference, then train Step3 one-emitter per Step4 flight/trajectory split.

