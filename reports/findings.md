# Findings — V2 §20H (small-slice evidence, honest)

## What works?
- Full RAPTOR composition same code small/full: tokenizer [B,T,E,2] I/Q separate per-antenna + ant identity + array_encoder per-element → Perceiver O(MN) M=8 NL=256 → temporal_mamba streaming state_t=F(state_{t-1},latent_t) (GRU, diff 0.05) → set_decoder K=4 + identity/logits → losses recon+Hungarian+NLL (89/94 grads ok).
- Masked recon 60% CI stable (synthetic recon 0.93→0.024, real RFUAV cross-site 00007→00014 0.98→0.93 3905 windows).
- Leakage-safe by site_id/serial validated (00007 train 3905 / 00008 val 3905).

## What does not?
- Passive single-snapshot range: **single 443.7 m vs temporal 401.6 m RMSE** on 50-1500m uniform (modal_range_temporal.py:1, 32 val, 2 epochs) — temporal helps 42m but still ~400m, confirms §16 hardest, §3 observability.
- Counting 0-2: **0/3 acc** (exist 0.96 all) — dummy Hungarian, Gate B not passed.
- Azimuth: **with geometry 94.5° vs without 99.8° RMSE** (modal_ablations.py:1, 32 val) — geometry helps 5.3°, still poor (~90° on 0-360 uniform ~103°), shows need for coherent multi-antenna + longer training.

## What is actually observable?
- Phase/geometry matters for az (5° gain), range needs temporal/Doppler (§16 Exp 2-5) not yet sufficient.

## What does temporal buy us?
- **42m range improvement** (443→401) with 4-window carry, but still far from usable — hypothesis unproven, needs longer context + Doppler-relevant Sionna scenes.

## What does coherent array buy us?
- **5.3° az gain** with explicit array_encoder per-element vs zero.

## Can we count emitters?
- Not yet — 0/3, needs proper existence BCE + Hungarian tuning per §12.

## Can we separate same-model?
- Not yet tested (needs RFUAV serial-level same-model pairs).

## Can we estimate physical state?
- Az degraded but geometry helps; range fails single frame.

## Sim-real gap?
- Synthetic Sionna generator ready `src/simulation/sionna_gen.py:1` exact GT schema §8, but synthetic→real gap not yet numerically measured (real has no GT per §7).

## Next bottleneck per §21
- Proper Hungarian + periodic az + NLL + longer temporal (Sionna trajectories, varying SNR/array/environment) one var at a time §15.

Artefacts per §19: git 4c9aff5, configs raptor_full.yaml, manifests schema.json, seeds 0, params 127k-367k, ckpts /ckpt/*, Modal logs above.
