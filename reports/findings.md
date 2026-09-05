# Findings — V2 §20H (Steps 12-19, small-slice evidence, not final metrics)

## What works?
- Full RAPTOR composition runs end-to-end on same code small/full: tokenizer (I/Q separate P=8) → array_encoder (coord proj) → Perceiver (M=32, O(MN)) → temporal_mamba streaming → set_decoder K=4 → probabilistic heads (μ/logvar) per §4, verified `scripts/train_raptor.py:1` 121k params.
- Masked recon 60% CI on synthetic: pipeline stable (`modal_full.py:1` Stage B). On real RFUAV small slice leakage-safe 00007/00008 3905 windows: recon loss decreasing (smoke).
- Leakage-safe splits by site_id/serial implemented per §15, validated `modal_gateA.py:1`.

## What does not?
- Passive single-snapshot range: **447-506 m RMSE** on 50-1500 m uniform (Step12 `modal_step12.py:1` 256 samples, 3 epochs) — no better than prior (~430m). Confirms §3 observability and §16 hardest hypothesis: time-invariant single window insufficient.
- Counting 0-2: predicts 4/4 for 0/1/2 (existence ~0.96) — dummy Hungarian not tuned, Gate B not passed. Shows counting is non-trivial, same-model separation not yet tested.

## What is actually observable?
- Phase / array geometry not yet ablated — array_encoder present but not measured. Expect azimuth/elevation tied to phase per §3, range requires temporal/Doppler (§16).

## What does temporal state buy us?
- Mamba present `temporal_mamba.py:1` streaming state_t = F(state_{t-1}, latent_t), but ablation no-temporal vs Mamba not yet run with proper velocity/range metric (§14). Hypothesis remains unproven.

## What does coherent array information buy us?
- Not yet measured; array ablation pending (§14).

## Can we count emitters?
- Not with current dummy loss — 0/3 acc. Needs proper Hungarian + existence BCE per §12.

## Can we separate same-model emitters?
- Not yet tested; requires RFUAV serial-level same-model pairs per §11.

## Can we estimate physical state?
- Azimuth/elevation not yet evaluated; range fails single frame.

## How large is sim-to-real gap?
- Not yet measured; Sionna generator `src/simulation/sionna_gen.py:1` produces exact GT but real labels (AERPAW-28 processed) not yet aligned per §7.

## Next bottleneck
- Proper Hungarian + NLL training for range/az with periodic loss, and temporal ablation with Doppler-relevant Sionna scenes (§16 Exp 2-5). Scale one var at a time per §15.

All experiments logged with git d87f7e9..9a0d67f, configs `raptor_full.yaml:1`, manifests `data/manifests/schema.json:1` per §19.
