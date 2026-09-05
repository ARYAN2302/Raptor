# Reconstruction Diagnosis — Tasks 1-3

## Table per Instruction

| Diagnostic | Before training | After reconstruction training | Verdict |
|---|---:|---:|---|
| Held-out reconstruction (masked 60% MSE, leakage-safe val 00008/00015/00019) | 1.3222 (prev) / 1.1614 (fresh init) | **0.2996** (train 0.1955, 10 epochs, 11715 windows, grad tok 0.06 perc 0.005) | **YES** — 77% drop (1.32→0.29), latent mean diff vs random 0.036 |
| Cross-serial identity (train 00007/00014 → test 00008/00015, FPV vs MINI4, 2-class, capture-level, frozen mean-pool [B,8,32]→[B,32] LogisticRegression, no fine-tune) | **0.500** (chance 0.5) confusion [[0,100],[0,100]] vs raw 0.500 | **0.500** (chance 0.5) confusion [[0,100],[0,100]] vs raw 0.510, per-class FPV 0.00/mini4 1.00 | **NO** — no improvement over chance/raw, vs before 0.500 identical |
| 1 vs 2 emitter separation (encoder latent before decoder, real A=FPV 00007, B=MINI4 00014, A+B 0.5/0.5, same-model A+B2 00007+00008, L2 mix-A 22.61 mix-B 22.61 A-B 45.22 verified) | **1.000 ±0.000** (chance 0.5) cos(zA,zB)0.3076 zA,zAB 0.6991 zB,zAB 0.8887 | **1.000 ±0.000** (chance 0.5) cos(zA,zB)0.1211 zA,zAB 0.8065 zB,zAB 0.6745 | **YES** — still perfect, mixture remains between (0.80/0.67 <0.99, distinguishable), but not due to recon (already 1.0 before) |

## Final Decision

**NO-GO / change representation** — Reconstruction converges (1.32→0.29) and 1 vs 2 count was already separable (1.0) and stays 1.0, but **cross-serial identity does not survive unseen physical serial (0.500 = chance, same as raw 0.51) — reconstruction did not make identity generalize.**

Do not proceed to full RAPTOR training, Mamba/site, or synthetic range until representation yields > raw + chance on **leakage-safe unseen serial** for Task2. Next is to change representation (e.g., per-antenna phase-aware, longer temporal, contrastive same-model objective) and re-test Tasks 1-3, not scale.

Artefacts: `experiments/modal_task1_train_recon.py:train` 0.2996, `modal_task2_trained.py:probe` 0.500, `modal_task3_trained.py:test` 1.000 + cos, ckpt `/ckpt/recon_trained.pt`, git b61bac3→final, Modal logs.
