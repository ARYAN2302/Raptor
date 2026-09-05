# Real-RF Three Experiments — Stop Condition (§STOP)

## Exp1: Does real RF contain usable structure in RAPTOR latent? (frozen, no training, leakage-safe by capture)
- **Setup:** 4 drones (FPV 00007/00008 5.76GHz, MINI4 00014/00015 2.45GHz, FLYSKY 00017, FRSKY 00019) — 6 captures, 1M each, 512-win, 32-D latent mean `src/models/raptor.py:1` before decoder, random init (ckpt mismatch, strict=False). Leakage-safe by capture (train 2 captures → test 2 captures).
- **Metric:** 4-class drone/model linear probe (LogisticRegression, 3-fold CV, leaky windows for structure test; leakage-safe would be 0 due to single capture per class). Raw magnitude baseline `sqrt(I^2+Q^2).mean` same probe.
- **Results:**
  - RAPTOR latent 4-class (leaky) **0.795 ±0.032** (chance 0.25) — `experiments/modal_real_three.py:exp1`
  - RAPTOR latent earlier 4-class leaky 0.952 ±0.026 (4 drones, 200 windows)
  - Raw magnitude 4-class **0.500 ±0.003**
  - 2-class same-model (FPV vs MINI4, train 00007/00014 → test 00008/00015, leakage-safe) — not yet run due to single-sample per capture issue, but 4-class leaky shows latent **has non-random structure** (0.795 vs 0.25 chance, +0.295 over raw), but not proven leakage-safe.
- **Answer:** **Yes, latent contains structure vs random (0.79 > 0.25, > raw 0.50), but not proven leakage-safe for same-model — need 2 serials per model for honest same-model probe.**

## Exp2: Can real RF mixtures produce separable emitter representations?
- **Setup:** Real windows A (FPV 00007), B (MINI4 00014), A+B arithmetic, same-model A+B2 (FPV 00007 + FPV 00008 different serials) through `src/models/raptor.py:1` encoder, K=4 queries, existence>0.5 count, `experiments/modal_real_three.py:exp2`.
- **Results (real IQ, no training):**
  - A exist [0.37,0.39,0.36,0.57] count **1** (correct 1)
  - B exist [0.37,0.41,0.35,0.56] count **1** (correct 1)
  - A+B exist [0.40,0.43,0.39,0.58] count **1** (should 2) — **fail**
  - same A+B2 exist [0.40,0.43,0.39,0.58] count **1** (should 2) — **fail, identical to cross-model**
  - Raw mag A 1.40 B 1.36 A+B 0.37 same 0.37 — no trivial power cue (normalized)
- **Answer:** **No — real mixtures not separable with current encoder (predicts 1 for both single and mixture), same-model identical to cross-model. No evidence for separable emitter representations on real IQ.**

## Exp3: Does temporal state improve anything measurable on real RF?
- **Setup:** Single emitter DJI FPV COMBO 00007 consecutive windows hop 256 (10 windows, same capture/session), `src/models/temporal_recurrent.py:1` GRU state `state_t=F(state_{t-1},latent_t)` vs reset, measure cosine similarity between consecutive latents `experiments/modal_real_three.py:exp3`.
- **Results:**
  - Temporal carry **0.9997 ±0.0004**
  - Reset each window **0.9999 ±0.0001**
  - **Reset slightly more stable (+0.0002), no benefit**
- **Answer:** **No measurable benefit on real RF for this stability metric — consecutive windows already 0.9999 similar, state does not improve.**

## Stop Condition per instruction
- **1. Structure:** **Weak yes** (leaky 0.79 > raw 0.50, but not leakage-safe proven)
- **2. Separation:** **No**
- **3. Temporal:** **No**

**Decision:** **Stop and diagnose representation before full real-data training (§STOP).** Next is to fix representation (proper recon, Hungarian, per-antenna phase) and re-test Exp1-3 with 2 serials per model for honest same-model probe, not proceed to synthetic/range.

Artefacts: `experiments/modal_real_three.py:exp1` `exp1.json` (4-class 0.795), `exp2` counts above, `exp3` cos sim, git 632989d, Modal logs.
