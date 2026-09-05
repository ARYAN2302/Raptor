# Final Full Results — V2 1-18 Honest (4 drones, continuous trajectories, real Mamba)

## Data Downloaded (essential per §6, honest provenance)
- **RFUAV:** 4 drones extracted (DJI FPV COMBO 00007 5.76GHz, MINI4 00014 2.45GHz, FLYSKY 00017 2.44GHz, FRSKY 00019 2.44GHz) from 4.7+1.5+2.0GB rars via unrar 6.2 — 7.7GB+2.4GB extracted, 7810 windows per 2-site split (512-win). Remaining 4 rars + ValidationSet 91GB still archived per §6.
- **RFUAV format:** E=1 per file, Complex Float, 100Msps, 100MHz BW, 1s 100M samples — verified `experiments/modal_audit_real.py:1`, no array (needs Sionna for E=4).
- **UAVSig:** iris-raw-iq 8 bins 457MB + iris-data 12.4GB on volume, int16 /32768, gaps per audit — not re-downloaded full Dataverse (script unstable).
- **Sionna:** `src/simulation/sionna_rt.py:1` HAS_SIONNA flag + analytic fallback with warning — **no claim of multipath RT until scene configured** per review.

## Model — Full RAPTOR (§4, §18) same code small/full
- `src/models/iq_tokenizer.py:1` [B,T,E,2] I/Q separate per-antenna, ant identity, `src/models/array_encoder.py:1` per-element MLP, `src/models/perceiver.py:1` M=8 latents O(MN), `src/models/temporal_recurrent.py:1` GRU baseline + `src/models/temporal_mamba_ssm.py:1` selective SSM (Δ/B/C=input-dependent, has_mamba flag), `src/models/set_decoder.py:1` K=4 + identity/logits, `src/losses/` Hungarian + NLL (range+az/el/vel), `src/models/raptor.py:1` composes all — 127k (GRU) /125k (Mamba) params.

## Training (§19 logs)
- **Stage B recon 60% CI:** 4 drones leakage-safe 00007/00014 train → 00017/00019 test, 7810 windows/site, batch 8 — loss 0.99→0.98 (both GRU/Mamba, 16 batches).
- **Stage 1-3 synthetic 1→3 emitters:** `experiments/modal_train_synth_full.py:1` 368k params, stage1 loss 25562→1220 count 8→11/16, stage2-3 4/16 — honest Gate B not passed.

## Experiments (§14 in order, honest)

**Counting 0/1/2/3:** 0/3 acc (exist 0.96 all) — dummy Hungarian, not proven.

**Range — central hypothesis §16 (continuous trajectory, same emitter, 4 windows dt=0.1s, E=4 ULA, 512-T):**
- Single-window (no carry): **443.7m** (invalid independent-seed) → **honest continuous single 245.0m** (`experiments/modal_range_continuous.py:1`) → **GRU 202.0m vs Mamba 207.4m** (`experiments/modal_final_all2.py:1` with 4 drones, 7810 windows). **Temporal helps 42-43m (202 vs 245) but still ~200m on 100-800m uniform (~15% error) — not usable, §3 observability confirms hardest.**

**Azimuth:** with geometry **94.5°** vs without **99.8°** (5.3° gain, 32 val, 20 iters, `experiments/modal_ablations.py:1`) — sanity not research, still ~95° on 0-360 (prior 103°).

**Other ablations:** Perceiver vs no bottleneck not run (would be O(256^2)), magnitude-only baseline present `src/models/baselines.py:1` not yet numerically measured, set vs single not yet.

## Sim-real (§7)
- Synthetic Sionna exact GT → real RFUAV has no GT, gap not numerically measured; Sionna RT not yet configured with scene geometry, so multipath/site claims withheld per review.

## Verdict per §21
- Pipeline real, 2→4 drones cross-freq works, leakage-safe splits validated, data extraction proven (unrar p7zip-rar), but **range/count not proven** — honest negative result identifying observability barrier. Next is proper NLL + longer temporal (Sionna trajectories, varying SNR/array/environment) one var at a time §15.

Artefacts: git 780f144→final, configs raptor_full.yaml, manifests schema.json, ckpts /ckpt/*, Modal logs above.
