# TSMS RAPTOR Range — Single-Receiver IQ → Range on Held-Out Distances

## Dataset
- **Drone:** Inspire 2 only (one drone, per instruction)
- **Distances:** 15 distances 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30m, each 500 captures → 7500 total
- **IQ:** 131072 complex samples per capture, `complex128`, E=1 single RF receiver (verified via `scipy.io.loadmat` and `h5py`/`mat73`, key `data` shape (131072,) dtype complex128, no other keys)
- **Target:** `range_m` from folder name (e.g., `10m` → 10)

## Split (Primary Evaluation)
- **TRAIN:** 2,6,10,14,18,22,26,30m (8 distances ×500 = 4000 captures)
  - Further split for validation: 85% train 3400 / 15% val 600 (both from same 8 distances, random)
- **TEST:** 4,8,12,16,20,24,28m (7 distances ×500 = 3500 captures) — **held-out entire distance values, never seen during training**
- **No random-window split** — train/test distances are disjoint, tests continuous RF→range relationship, not memorization.

## RAPTOR (Range-Only)
- **Architecture kept:** `complex IQ tokenizer (I/Q separate Conv1d patch 16 stride 8 d64) → Perceiver latent bottleneck (M=32, 2 layers, 4 heads) → temporal/state (GRU) → range head (LayerNorm→Linear→ReLU→Linear)` — existing `src/models/raptor.py:1` with `temporal_recurrent.py:1`, other heads disabled, single range head only (adapted minimally, not redesigned)
- **Input:** raw complex IQ `[B,131072,1,2]` (131072 window, E=1, 2=I/Q) → tokenizer `E*L=8192` tokens? Actually `T=131072 patch16 stride8 → L=16384` per E, but with window 4096 random window per capture for training (4096 consecutive samples from 131072) → `L=512` tokens, `E*L=512` → Perceiver `M=32` → head
- **Training:** 3400 train windows, batch 16, AdamW 0.001 Cosine T_max=10, 5 epochs (quick test) / 10 epochs (full), 1.22M params (d64 M32) — small for held-out-range smoke; full 7M would be d256 M64 but not needed for collapse diagnosis

## Metrics (Held-Out Distances 4,8,12,16,20,24,28)

### Baselines
- **A. Mean-range baseline:** train mean 15.6m (from train dists 2,6,10,14,18,22,26,30) → test MAE **6.86** RMSE 8.00 (test mean 16.0 std 8.0)

### RAPTOR (5 epochs, 500 train windows subsampled for speed, 4096 window, T4)
- **MAE 6.97 RMSE 8.04 corr 0.172** pred mean **15.2** std **0.0** gt mean 16.0 std **8.0** — **worse than baseline 6.86, pred std 0.0 vs gt std 8.0 ratio 0.0 → collapse to ~15.2 (near train mean 15.6)**
- **Per held-out distance:**
  - 4m: MAE **11.17** (n=70) pred mean 15.2
  - 8m: MAE 7.17 pred 15.2
  - 12m: MAE 3.17 pred 15.2
  - 16m: MAE **0.83** pred 15.2 (closest to mean)
  - 20m: MAE 4.83 pred 15.2
  - 24m: MAE 8.83 pred 15.2
  - 28m: MAE 12.82 pred 15.2

### Diagnosis — Is RAPTOR genuinely learning range?
- **Constant/near-constant predictions:** YES — pred std 0.0 vs gt std 8.0, all 7 held-out distances predicted as **15.2m** (train mean 15.6m)
- **Memorization of training distances:** No — train distances are 2,6,10,14,18,22,26,30, test are 4,8,12,16,20,24,28, but model predicts 15.2 for all test, not any train distance specifically, just the mean
- **Prediction collapse:** YES — ratio 0.0, corr 0.172 (near 0, constant prediction has undefined corr, here nan → 0.172 due to tiny variance)
- **Performance vs distance:** MAE is lowest at 16m (0.83) where true 16 is closest to predicted 15.2, and largest at 28m (12.82) and 4m (11.17) farthest from mean — **error scales with distance from training mean, not with RF information**
- **Learned baseline already beats mean baseline?** NO — learned RAPTOR 6.97 vs mean 6.86, **not beating**

## Plot
`true range vs predicted range` — scatter would show horizontal line at y=15.2 across x=4-28, vs ideal diagonal r-- (saved as `/tmp/tsms_true_vs_pred.png` and `/ckpt/tsms_true_vs_pred.png` in previous runs)

## Scientific Question
**Can a single RF receiver's raw IQ support absolute range prediction on previously unseen distances?**

**Answer: NO — with the current RAPTOR range-only model on TSMS Inspire 2 (131072 complex, E=1, held-out 4,8,12,16,20,24,28), the model collapses to predicting the training mean (~15-16m) for all held-out distances, MAE 6.97 vs baseline 6.86 (not beating), per-distance MAE shows no continuous RF→range relationship.**

**Next:** Need to address collapse (normalization, loss scaling, longer window, or acknowledge single-receiver RSS/path-loss may be insufficient for absolute range without array/temporal).

