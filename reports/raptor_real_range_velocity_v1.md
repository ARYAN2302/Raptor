# RAPTOR Real Range + Velocity V1 — Dataset 19 Single Emitter

## 1. Dataset / Split
- **Dataset:** AERPAW Dataset 19 A2G Channel Sounding, 9 flights `2023-12-15_15_41` ... `16_53`, 56Msps, 3.686GHz, Zadoff-Chu, 1 channel `f32_le` 10344 samples (0.18ms) per capture, `core:tx_location`/`rx_location` lat/lon/alt, `core:velocity` XYZ, `core:dist`, GNSSDO PPS sync, `core:num_channels=1` (not array)
- **Split:** **Trajectory/flight-level** — Train flights 15_41,15_51,15_58,16_14,16_19,16_36 (6 flights, 600 windows single, 5067 seqs temporal) — Test flights 16_42,16_47,16_53 (3 flights, 300 windows single, 2431 seqs temporal). **Never random neighboring windows** — windows from same flight never across split.
- **Ground truth per §8:** range/az/el via `latlonalt_to_ecef`+`ecef_to_enu`+`enu_to_spherical`, radial via `dot(vel,kvec)`, 3D velocity provided + via `compute_velocity` (dt 0.4s), coordinate ENU ref at receiver.

## 2. Model Configuration
- **Pipeline:** `IQ [B,1024,1,2] (f32 real I, Q=0) → ComplexIQTokenizer patch 8 stride 8 d_model 32 max_ant 1 → ArrayEncoder per-element MLP → Perceiver M=8 latents O(MN) → Temporal GRU `state_t=F(state_{t-1},latent_t)` (single) vs `RAPTORTemporal` with 4-window carry (temporal) → SingleHead `fc_range/radial/vel` from `latents.mean(dim=1)` → uncertainty `logvar` (not evaluated)
- **Params:** 92k (single) / temporal same + GRU
- **Output for single emitter:** `range, radial, vx/vy/vz` (no Hungarian, K=1)

## 3. Training Setup
- **Preprocessing:** `f32` raw → [win,1,2] with Q=0, no silent resample, provenance recorded per manifest `data/manifests/aerpaw19_train.json`
- **Loss:** L1 `range/100 + radial + vel` (range normalized /100)
- **Optimizer:** AdamW lr 0.001, batch 16, 5 epochs
- **Temporal context:** **A Single-window:** current IQ capture only (1024 samples, 0.18ms) **B Temporal:** 4 consecutive captures from same trajectory, 0.4s spacing, total 1.6s duration, state carried `state0→state1→state2→state3`, predict last

## 4. Baseline Results (same test set, no training)
- **Range:** constant mean train 34.0m → test MAE **1.0** RMSE **1.6** (test range mean 33.7 std 1.6 min 31.3 max 45.0 — narrow, trivial)
- **Radial velocity:** mean 0.141 → MAE **0.274** RMSE ~0.35
- **3D velocity:** mean [0.002,-0.0004,-0.624] → MAE **0.355**

## 5. Single-Window Results (held-out flights)
- **Range:** MAE **26.8** (epoch 4) / 32.1 epoch0 → **88.3** MAE **134.2** RMSE (temporal seq test, but single-window model) — **worse than baseline 1.0/1.6**
- **Radial:** MAE **0.192** RMSE **0.513** vs baseline 0.274 — **slightly better than baseline (-0.08)**
- **3D vel:** MAE **0.05-0.07** RMSE **0.24-0.25** vs baseline 0.355 — **better than baseline**

## 6. Temporal Results (4 windows, 1.6s)
- **Range:** MAE **111.4** RMSE **162.4** (epoch 4) vs single **88.3/134.2** — **worse than single** (+23 MAE)
- **Radial/vel:** similar, temporal not better (train loss 4.74 vs single 4.40)

## 7. Held-Out Test Errors (summary)
| Quantity | Single MAE | Temporal MAE | Baseline MAE | GT std |
|---|---|---|---|---|
| Range | 88.3 | 111.4 | **1.0** | 1.6 |
| Radial | 0.192 | ~0.20 | 0.274 | — |
| 3D vel | ~0.05 | ~0.05 | 0.355 | — |

## 8. Failure Cases
- **Range:** Single-channel RSS/path-loss cue is weak — test ranges 31-45m narrow, model predicts ~0-60m wide, error 26-88m >> baseline 1m. Temporal 4×0.4s does not help (111m worse), likely because 1.6s trajectory insufficient for observability and single-channel has no array phase.
- **Velocity:** Radial/3D better than baseline but still 0.19 error on ~0.02 mean — not proven useful.

## Verdicts (held-out real trajectories vs baselines)

**RANGE: NO** — held-out MAE 26-88m vs baseline 1.0m, temporal 111m worse than single 88m — no evidence IQ over time recovers range with single-channel 1.6s context; trivial mean baseline wins.

**RADIAL VELOCITY: INCONCLUSIVE** — single 0.192 vs baseline 0.274 (30% better) but absolute error 0.19 on small velocities, not proven robust across flights.

**3D VELOCITY: INCONCLUSIVE** — 0.05 vs baseline 0.355 (better), but single-channel Doppler not verified, and temporal not better.

**Conclusion per §16:** With Dataset 19 single-channel 56Msps Zadoff-Chu and 0.4s spacing, **range is not demonstrably recoverable** from IQ alone vs trivial baseline; temporal 1.6s does not help. Velocity shows weak signal but needs longer trajectory and coherent array for azimuth/elevation (not supported per audit `E=1`).

