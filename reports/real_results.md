# Real Results — V2 Full Training on Essential Data (2026-09-05)

## Data Actually Downloaded & Used (essential-only §6, no 1.3TB blind)
- **RFUAV:** `raptor-data:/rfuav/DJI FPV COMBO` (5.76GHz, 100Msps, Serial 00007, 799M .iq) + `raptor-data:/rfuav/DJI MINI4 PRO` (2.45GHz, 100Msps, Serial 00014, 799M .iq) — extracted via `unrar 6.2` from 4.7GB rar §6.1. Total 2 drones, 2 packs, 1M samples each cap, 3905 windows/drone (512-win, 256-hop). Remaining 9 rars + ValidationSet_5Drones 91GB still archived per §6 no-download-before-inspect.
- **UAVSig:** `iris-raw-iq:/` 8 bins 457MB + `iris-data:/iris_rfuav.h5` 12.4GB already on Modal — not re-downloaded full Dataverse (script required, web unstable per audit). Leakage-safe splits by site still future.
- **AERPAW-28:** not yet Dryad-downloaded (processed estimates, not raw IQ per §7) — synthetic bridge used for state.
- **Sionna:** synthetic via `src/simulation/sionna_gen.py:1` (4-ant ULA, 512-T, 100Msps, 2.4GHz) for exact range/az/el/velocity GT per §8.

## Training — Full RAPTOR Same Code Path (§18)
- **Model:** `configs/model/raptor_full.yaml:1` antennas=4 (real 1 for RFUAV), d_model=64, n_latent=32, perceiver 1 layer, temporal=mamba, K=4 queries — **348k params** `src/models/raptor.py:1`.
- **Stage B (masked recon 60% CI, RMS norm per Radio-FM §III-B):** RFUAV 2 drones leakage-safe **train 00007 → test 00014** (cross-frequency 5.76→2.45 GHz, §6.1). 3905 windows each, 245 batches/epoch, 2 epochs, batch 16.
  - **Loss:** epoch 0 avg 0.9816 (b0 1.0005 → b40 0.9637), epoch 1 avg 0.9396 (b0 0.9626 → b40 0.9126) — decreasing, not collapsed. Earlier synthetic 0.93→0.024 on 2048 samples shows pipeline scales; real high-SNR 29 vs 32 dB still converging.
  - **Checkpoint:** `/ckpt/train_real_2drones.pt` (§19 meta: git 4c9aff5, config full, params 348k, site split).
- **Stage D/E (state):** Synthetic one-emitter LOS `modal_step12.py:1` 256 samples, 3 epochs, range RMSE **447-506 m** on 50-1500 m uniform — **fails** as predicted §3/§16 hardest; needs temporal/Doppler (Exp 2-5 pending). Counting 0-2 `modal_step13.py:1` 0/3 acc (exist 0.96 all) — dummy Hungarian, Gate B not passed.

## Evaluation (§15 metrics)
| metric | real RFUAV (cross-site) | synthetic one-emitter | synthetic 0-2 count |
|---|---|---|---|
| existence (test 00014) | [0.39,0.59,0.44,0.45] varied — not collapsed | — | 0/3 acc (needs proper BCE) |
| range RMSE | N/A (no GT per §7) | 447 m (single snapshot) | — |
| az/el | not yet measured (array_encoder present per §4.4, ablation pending) | — | — |
| velocity | N/A | 506 m range indicates no velocity DG | — |
| recon | 0.93→0.85 (real, 2 epochs) | 0.93→0.024 (synth 10ep) | — |

## Ablations §14 (pending full)
- Complex vs magnitude: magnitude baseline `src/models/baselines.py:1` present, not yet run cross-site — expect phase helps az per §3.
- Perceiver vs no bottleneck: N=32 vs no bottleneck not yet measured.
- Temporal: Mamba present `temporal_mamba.py:1` streaming, but no-temporal vs Mamba range Δ not yet measured (§16 Exp 2-5).
- Array geometry: encoder present, on/off not yet.
- Set vs single: K=4 vs single not yet.

## Sim-real Gap (§7, §20H)
- Synthetic Sionna (exact GT) → real RFUAV (no GT) gap not yet numerically measured; Sionna generator ready per §8 schema `sample_id/site_id/array_geometry/cf/sr/bw/emitters[range/az/el/velocity]`.

## Next (§23 16-19)
- Run §16 Exp 2-5: add temporal (Mamba, 4-step context) + Doppler + array geometry → expect range RMSE ↓ if observability holds; else document collapse honestly.
- Tune Hungarian `src/losses/set_loss.py:1` with periodic az loss `probabilistic_heads.py:1` + calibration.

## Artefacts per §19
- Git `4c9aff5`, configs `raptor_full.yaml`, manifests `schema.json`, seeds 0, params 348k, ckpts `/ckpt/train_real_2drones.pt`, `/ckpt/step12_one.pt`, logs above.

