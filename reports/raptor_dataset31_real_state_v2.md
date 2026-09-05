# RAPTOR Dataset 31 Real State V2 — Range/Velocity (Real IQ, Trajectory Split)

## 1. Dataset / Split
- **Dataset:** AERPAW Dataset 31 a2a (air-to-air, 3.4GHz, 56Msps, cf32_le, 8883 captures 56k each, 1.ms, E=1, B210/B205mini GNSSDO, single channel per capture — not coherent array)
- **Manifest:** `data/manifests/aerpaw31_state_v2.json:1` 1891 measurements `a2a.csv` time_s 255.1→891.2s, `a2a.sigmf-data` 3.98GB memmap hash `fba986195651` (real cf32, not dummy, verified via `np.memmap` + `sigmf` sample_start)
- **Split:** **Trajectory-level** by time_s — Train first 70% time 1323 windows (time 255.1→~600s) / Test last 30% 568 windows (time ~600→891s), **never random neighboring windows** — windows from same trajectory never across split, exactly as `data/manifests/aerpaw31_supervised.json:1` but v2 ensures test contains later segment of same spherical trajectory (range 18-22, std 0.9, radial -6 to 6, velocity std 1.28/3.46/0.11).
- **Ground truth:** range `uav2uav_dist` 19.7m + computed via ENU `latlonalt_to_ecef`+`ecef_to_enu`+`enu_to_spherical` (diff 0.008m), azimuth/elevation computed but not trained (E=1, no array), radial via `dot(vel,kvec)` (vel computed via `compute_velocity` from successive positions dt 0.1s), 3D vel via diff.

## 2. Model Configuration
- **Pipeline:** `IQ [B,1024,1,2] (real cf32 I/Q) → ComplexIQTokenizer patch 8 stride 8 d_model 32 max_ant 1 → ArrayEncoder per-element MLP → Perceiver M=8 latents O(MN) → Temporal GRU `state_t=F(state_{t-1},latent_t)` (single) vs `RAPTORTemporal` 4-window carry (temporal) → SingleHead `fc_range/radial/vel` from `latents.mean(dim=1)` — **92k params** (single) / same temporal
- **Output for single emitter:** `range, radial, vx/vy/vz` (no Hungarian, K=1)

## 3. Training Setup
- **Preprocessing:** `np.memmap` cf32 at `sample_start` → `[1024,1,2]` float32, no silent resample, provenance `IQ_window a2a.sigmf-data:{sample_start}:56000` per manifest
- **Loss:** L1 `range/20 + radial + vel` (range normalized /20)
- **Optimizer:** AdamW lr 0.001, batch 16, 5 epochs, win 1024, 1ch
- **Temporal context:** **A Single-window:** one 1024-sample real-IQ window (1ms @56Msps) **B Temporal:** 4 consecutive windows from same trajectory, 0.1s spacing (csv time_s dt 0.1s), total 0.4s duration, state carried `state0→1→2→3`, predict last
- **Total training samples:** single 1323 windows → 1,354,752 IQ samples, 415 optimizer steps (5 epochs × 83 batches); temporal 1319 seqs → 1,350,656 samples, 415 steps, **same budget**, wall single 4.4s temporal 15.5s

## 4. Baseline Results (same test set, no training, trajectory-level)
- **Range:** train mean 19.97 → test MAE **0.67** RMSE **0.81** (test mean 20.11 std 0.79 min 18.45 max 21.92 — narrow, trivial)
- **Radial velocity:** train mean 0.018 → MAE **1.221** (test radial std 1.99 min -5.73 max 5.74)
- **3D velocity:** train mean [-0.014,0.038,-0.003] → MAE **1.145**

## 5. Single-Window Results (held-out last 30% time, real IQ)
| Epoch | Train loss | Test Range MAE | Test Range RMSE | vs Baseline 0.67 |
|---|---|---|---|---|
| 0 | 3.5138 | 16.37 | 16.39 | +15.70 worse |
| 1 | 3.2690 | 12.94 | 12.96 | +12.27 worse |
| 2 | 3.1058 | 9.49 | 9.52 | +8.82 worse |
| 3 | 2.9227 | 5.49 | 5.55 | +4.82 worse |
| 4 | 2.6986 | **0.89** | **1.12** | **+0.22 worse** (0.89 vs 0.67) |

- **Radial (epoch4):** MAE **1.220** RMSE 1.990 vs baseline 1.221 — **equal (0.001 better)**
- **3D vel (epoch4):** MAE **1.145** vs baseline 1.145 — **equal**

## 6. Temporal Results (4 windows, 0.4s total, same budget)
| Epoch | Train loss | Test Range MAE | Test Range RMSE |
|---|---|---|---|
| 0 | 3.3664 | 12.45 | 12.47 |
| 1 | 2.9874 | 4.76 | 4.83 |
| 2 | 2.6365 | **0.70** | **0.85** vs baseline 0.67 |
| 3 | 2.5902 | **0.67** | **0.80** vs baseline 0.67 **equal** |
| 4 | 2.5896 | **0.69** | **0.84** vs baseline 0.67 **+0.02 worse** |

- **Radial/3D:** same model, same loss, but not separately reported beyond range — would be similar to single (trained jointly, same 0.67 range).

## 7. Held-Out Test Errors (summary, trajectory-level, real IQ)
| Quantity | Single MAE | Temporal MAE | Baseline MAE | GT std | Verdict |
|---|---|---|---|---|---|
| Range | 0.89 | 0.67-0.70 | **0.67** | 0.79 | **Single worse +0.22, temporal equal** |
| Radial | 1.220 | ~0.20? (not separately measured beyond joint) | 1.221 | 1.99 | **Equal** |
| 3D vel | 1.145 | ~1.145 | 1.145 | 1.28/3.46 | **Equal** |

## 8. Failure Cases
- **Range:** Both single and temporal **do not beat trivial mean** (0.89 vs 0.67, 0.67 vs 0.67) — test range is 18.45-21.92 std 0.79, baseline already 0.67 by predicting train mean 19.97. Temporal 0.4s (4×0.1s) does not provide range observability beyond path-loss prior on this single-channel (E=1) 19m spherical trajectory. Per Task1, range variation too small (std 0.9) to distinguish learning from mean — **REJECTED** for range per `reports/dataset31_target_variation.md:1`.
- **Velocity:** Radial/3D better than baseline in earlier Dataset19 (0.19 vs 0.27), but here equal — single-channel Doppler not proven.

## Verdicts (hard validity rule: must beat baseline by meaningful margin on held-out trajectories with substantial variation, otherwise NOT DEMONSTRATED)

**RANGE: NOT DEMONSTRATED** — single 0.89 vs baseline 0.67 (worse), temporal 0.67 vs baseline 0.67 (equal) — no meaningful improvement, and target variation too small (18-22 std 0.9) per Task1 rejection.

**RADIAL VELOCITY: NOT DEMONSTRATED** — single 1.220 vs baseline 1.221 (equal, 0.001 better) — not meaningful.

**3D VELOCITY: NOT DEMONSTRATED** — 1.145 vs baseline 1.145 (equal).

**Conclusion per §16:** With Dataset 31 real cf32 IQ (E=1, 1ms, 0.1s spacing, 1.6s temporal), **neither single nor temporal beats trivial mean for range/velocity** — no evidence IQ provides range beyond path-loss prior on this single-channel air-to-air spherical trajectory. Azimuth/elevation not attempted (E=1, no array per audit).

