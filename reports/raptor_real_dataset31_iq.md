# Dataset 31 REAL IQ — RAPTORSingle Range/Velocity (Trajectory Split, Real cf32)

## Verification that IQ is Real (not dummy)
- **File:** `/Users/adarshthakur/Desktop/DATASET/a2a.sigmf-data` 3.98GB, `a2a.sigmf-meta` 3.11MB, 8883 captures, `sample_start` per capture, `core:datatype cf32_le`, `sample_rate 56Msps`, `center 3.4GHz`
- **Hash:** first 1024 complex `fba986195651` vs dummy random `1f6a0f3e9e17` — differs, mean abs 0.0592 vs dummy 0.8
- **Meta:** `sample_start 0 datetime 2025-09-18T20:31:39.970727` sample_rate 56Msps
- **Manifest:** `data/manifests/aerpaw31_supervised.json:0` `IQ_window a2a.sigmf-data:0:56000` `sigmf_sample_start 0` — wired via `np.memmap(..., dtype=np.complex64)[sample_start:sample_start+1024]` → `[1024,1,2]` float32
- **Sanity:** model received `memmap cf32` not `np.random.randn` dummy (hash differs, stats differ)

## Training Setup (same as Dataset 19, §Task1-3)
- **Dataset:** 1891 measurements `a2a.csv` time_s 255.1→891.2s, 70% train 1323 / 30% test 568, **trajectory-level split** first 70% time → last 30% time (contiguous, never random windows)
- **Model:** `RAPTORSingle` `IQ [B,1024,1,2] (cf32 I/Q) → ComplexIQTokenizer P=8 d32 → ArrayEncoder per-element → Perceiver M=8 → Temporal GRU (single) vs `RAPTORTemporal` 4 windows 0.1s spacing (temporal)` → SingleHead `fc_range/radial/vel` from `latents.mean(dim=1)` — same `d32 M8` 92k params as Dataset 19, no Hungarian (K=1), `E=1` single channel per audit
- **Loss:** L1 `range/20 + radial + vel` (range normalized /20)
- **Budget:** 5 epochs, batch 16, AdamW 0.001, win 1024, 1ch

## Results vs Same Baselines (held-out last 30% time, trajectory-level)

**Baselines (mean train):**
- Range mean 19.97 → test MAE **0.67** RMSE 0.81 (test mean 20.11 std 0.79)
- Radial mean 0.018 → MAE **1.221** (test radial std 1.99)
- 3D vel mean [-0.014,0.038,-0.003] → MAE **1.145**

**Single-window (real IQ, 1×1024):**
- Range **0.69** RMSE 0.83 (epoch4) vs baseline 0.67 — **equal, not better** (0.02 worse)
- Radial: not separately reported in last run, but trained jointly — previous dummy 0.69 was dummy, real single 0.69 range only
- 3D vel: not separately reported

**Temporal (4 windows, 0.4s spacing, state carry, 1.6s total):**
- Range **0.67** RMSE 0.79 (epoch4) vs baseline 0.67 — **matches baseline** (0.00 diff), vs single 0.69 — **0.02 better than single, equal to baseline**
- Radial/3D: same model, same loss, but not separately reported — would be similar to range (trained jointly)

## Hard Final Answer (Dataset 31 REAL IQ, held-out trajectory, same baselines, no azimuth/elevation per E=1)

| Quantity | Result | Verdict |
|---|---|---|
| **Range** | Single 0.69 vs baseline 0.67 (equal), temporal 0.67 vs baseline 0.67 (equal) — no improvement over trivial mean | **INCONCLUSIVE** (not better than mean, temporal matches baseline only) |
| **Radial velocity** | Not separately measured for real IQ in this run (trained jointly, baseline 1.221) — previous single 0.69 was dummy IQ, real radial not yet | **INCONCLUSIVE** (no real IQ radial vs baseline reported) |
| **3D velocity** | Not separately measured (baseline 1.145) — same as radial | **INCONCLUSIVE** |

**Conclusion:** With **real Dataset 31 cf32 IQ (56Msps, 3.4GHz, E=1, 1ms → 1024 window)**, **neither single-window nor temporal (4×1.6s) beats the trivial mean range baseline** (0.69 vs 0.67, 0.67 vs 0.67) — **no evidence that passive RF IQ provides range beyond path-loss prior** on this single-channel air-to-air spherical trajectory. Velocity not proven. **Do not claim range/velocity from real IQ on this E=1 dataset.**

Artefacts: `scripts/train_aerpaw31_real_iq.py:1` real memmap cf32, hash fba986, `data/manifests/aerpaw31_supervised.json:0` sample_start 0, git 21dd4e5, Modal logs.
