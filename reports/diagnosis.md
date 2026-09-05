# Diagnosis — Frozen RAPTOR Representation (Stop Before Full Training)

## 1. Reconstruction Result (Task1)
- **Path:** `IQ [B,T=512,E=1,2] → tokenizer [B,E*L=64,32] → array_encoder per-element → Perceiver [B,M=8,32] → recon_decoder [B,L=64,32]` via cross-attn `src/models/recon_decoder.py:1` + `src/models/raptor.py:forward_recon`.
- **Real data:** RFUAV 4 drones (FPV 00007/00008, MINI4 00014/00015, FLYSKY 00017, FRSKY 00019) 1M each + UAVSig 1 bin, 15621 windows 512-win, 60% CI masking (Radio-FM Eq10).
- **Loss:** initial masked MSE **1.3599** (tokens 64×32, mask 330/512) → held-out **1.3222**, `experiments/modal_task1_recon.py:task1`. Gradients `tok/arr/perc/recon 50/51`, `tok proj_I` grad 2.44 — flow verified `IQ→tok→array→perc→recon`.
- **Status:** **Infrastructure works, not yet trained to convergence** (1.35→1.32, not 0.02 as synthetic 10ep). Needs 2-3 epochs on 4 drones to prove recon learns.

## 2. Leakage-Safe Cross-Serial Result (Task2)
- **Setup:** 6 captures (2 serials per FPV/MINI4) — train on one serial, test on unseen serial **same model** per §15, never neighboring windows, `experiments/modal_task2_probe.py:task2`.
- **Probe:** frozen latent mean [B,8,32]→[B,32], LogisticRegression.
- **Results:**
  - **Same-model leakage-safe (FPV 00007→00008 + MINI4 00014→00015, 2-class):** **0.500 acc (chance 0.5)** confusion `[[0,100],[0,100]]` — predicts all as MINI4, no generalization.
  - **4-class leaky (random windows, 3-fold):** **0.795 ±0.032** (chance 0.25) vs raw magnitude **0.500 ±0.003** — latent has structure but leaks via window adjacency.
  - **Cross-model train FPV→test MINI4:** ill-posed (disjoint labels) — not reportable, shows need for 2 serials per class for honest 4-class.
- **Answer to Q:** **No — frozen random-init latent does NOT represent emitter/model that survives unseen physical unit (0.5 = chance).**

## 3. Real-Mixture Representation/Separability (Task3, before decoder)
- **Benchmark:** Real windows A (FPV 00007_2560), B (MINI4 00014_2560), A+B 0.5/0.5, same-model A+B2 (FPV 00007+00008) — IDs saved `experiments/modal_task3_mix.py:task3`, verified `L2 mix-A 22.61 mix-B 22.61 A-B 45.22` and `31.28/62.57` — **mix != A/B and neither dominates** (both 22 <45).
- **Encoder latent (mean pooled 32-D) before set decoder:**
  - **1 vs 2 emitters classifier:** **1.000 ±0.000** (chance 0.5) — 50×1-emitter vs 50×2-emitter, 3-fold — **perfectly separable** for count.
  - **Embedding cos sims:** `cos(zA,zB) 0.3076` vs `cos(zA,zAB) 0.6991` `cos(zB,zAB) 0.8887` `cos(zA,zSame) 0.6477` — mixture lies between A and B (0.69/0.88 >0.30), B dominates slightly, same-model similar 0.64.
- **Answer:** **Count is separable (1.0 acc), but assignment/identity is not proven — mixture is between, not near either source, and same-model vs cross-model identical per Task2.**

## 4. Diagnosis — Most Likely Failure
- **Recon not yet learned:** 1.35→1.32, no training to convergence, so latents are near-random (hence 0.5 cross-serial, 0.795 leaky only from window leakage).
- **Per-antenna phase not exploited for identity:** RFUAV is E=1, so array_encoder has no spatial diversity to learn hardware fingerprint; identity must come from temporal/spectral alone, which random recon doesn't capture.
- **Set decoder not yet involved:** Task3 1.0 count on latent is promising for detection, but Task2 shows decoder would still fail on identity without trained recon.
- **Recommendation:** **Train recon to convergence on 4 drones (2 serials per model) with leakage-safe splits, then re-run Task2 same-model probe** — do not add site/Mamba/synthetic until recon + probe exceeds raw baseline + chance on **unseen serial**.

Artefacts: `experiments/modal_task1_recon.py:task1` loss 1.35, `modal_task2_probe.py:task2` 0.5, `modal_task3_mix.py:task3` 1.0 + cos sims, git 17c6c17, Modal logs.
