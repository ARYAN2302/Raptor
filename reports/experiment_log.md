# Experiment Log

## 001_iq_pretrain — P2 masked recon
- Config: `configs/training/pretrain.yaml` (synthetic 512, window 4096, mask 0.4)
- Baselines: CNN-A, magnitude-only-B, spectrogram-C, no-temporal-D all logged.
- Gate A: raw-IQ > magnitude/spectrogram on UAVSig transfer else stop.

## 002_identity / 003_counting / 004_temporal — Phases 3-5
- Synthetic mixtures 0-2 → 0-4, same-model overlapping, count acc + PR.

## 005_sionna_state — Phase 7
- Sionna scenes: 1 emitter LOS → multipath → multi-emitter → site variation. Targets: range/az/el/velocity RMSE + ECE.

## 006_sim2real — Phase 8
- Train Sionna, test AERPAW processed truth. Gap reported honestly.
