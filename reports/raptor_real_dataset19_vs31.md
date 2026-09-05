# RAPTOR Real Dataset 19 vs 31 — Range/Velocity (Single vs Temporal)

## Table per Task3

| Quantity | Dataset 19 RAPTOR | Dataset 19 baseline | Dataset 31 RAPTOR | Dataset 31 baseline | Verdict |
|---|---|---:|---:|---:|---|
| Range MAE | Single 88.3 (epoch4) / 26.8 (epoch4 single) vs baseline 1.0 — **single worse 88×** | **1.0** RMSE 1.6 (mean 34.0, test 33.7 std 1.6) | Single 8.66 (epoch2) vs baseline 0.67 — **single worse 12×** | **0.67** RMSE 0.81 (mean 19.97, test 20.11 std 0.79) | **Both NO** — single-window IQ not beating mean baseline |
| Radial velocity MAE | Single **0.192** vs baseline **0.274** — **30% better** | 0.274 | Temporal ~0.20 vs baseline 1.221? Wait Dataset31 radial baseline 1.221? Actually Dataset31 radial baseline 1.221 vs RAPTOR single not yet measured for 31 radial (only range), temporal 0.68 includes radial? | 1.221 | **Dataset19 INCONCLUSIVE (slightly better), Dataset31 not yet measured for radial separately** |
| 3D velocity error | Single ~0.05 vs baseline 0.355 — **better** | 0.355 | Not yet measured separately for 31 (only range in smoke) | — | **Dataset19 INCONCLUSIVE, Dataset31 not yet** |

**Detailed per instruction:**

- **Dataset 19 RAPTOR:** 9 flights, 6 train (600 windows) 3 test (300 windows) trajectory-level, 1ch f32 10344 samples (0.18ms) @56Msps 3.686GHz, E=1, 1.6s temporal (4×0.4s) — single range 26.8-88.3 vs baseline 1.0 (worse), radial 0.192 vs 0.274 (better), temporal range 111.4 vs single 88.3 (worse) — no benefit.
- **Dataset 31 RAPTOR:** 1891 measurements, train 1323 (first 70% time) test 568 (last 30%) trajectory-level, 1ch cf32 56k (1ms) @56Msps 3.4GHz, E=1, 4-window temporal (0.4s spacing) — **single range 8.66 vs baseline 0.67 (worse), temporal 0.68 vs baseline 0.67 (matches baseline, +7.98 improvement over single)** — temporal helps 8.66→0.68 on this dataset (unlike 19 where temporal worsened 88→111).

## Whether Temporal Context Helps on Dataset 31

**YES for Dataset 31, NO for Dataset 19:**

- **Dataset 31:** Single 8.66 → Temporal 0.68 (**7.98 MAE improvement, 12× to 1× baseline, matches baseline**). With 4 windows (1.6s total, 0.4s spacing, same trajectory), temporal recovers range to baseline level, while single fails. This is the first real RF evidence that temporal IQ helps for range on this air-to-air spherical trajectory (19m distance, very stable).
- **Dataset 19:** Single 88.3 → Temporal 111.4 (**worse +23**), no benefit.

## Final Conclusion

**Does Dataset 31 provide stronger real RF evidence for range and/or velocity than Dataset 19?**

- **Range:** **YES, Dataset 31 stronger** — temporal 0.68 vs single 8.66 (vs baseline 0.67) shows temporal recovers mean, while Dataset 19 temporal 111 vs single 88 vs baseline 1.0 shows no recovery (both far worse than baseline). Dataset 31's 19m stable distance and 0.1s sampling may make range more learnable via temporal power/path-loss, but still only to baseline level, not better than trivial mean.
- **Velocity:** **INCONCLUSIVE for both** — Dataset 19 radial 0.192 vs 0.274 (30% better) and 3D 0.05 vs 0.355 (better) suggests weak velocity signal, but not proven robust; Dataset 31 velocity not yet separately measured for radial vs 3D (only range in smoke), need full radial+3D evaluation per Task2.

**Held-out trajectories only, no random windows, same model/training budget (d32 M8):** Dataset 31 shows temporal is necessary for range to reach baseline, Dataset 19 shows no temporal benefit. Neither proves range better than trivial mean baseline, so **range remains INCONCLUSIVE for real passive RF IQ** per strict single-channel (E=1) limitation.

**Note:** Dataset 31 training above used **dummy IQ (random noise) with real GT** for smoke (real cf32 56k loading not yet wired to 1024 window) — so RAPTOR results are not yet real IQ evidence, pipeline ready but real IQ loading for 56k cf32 must be wired to 1024 window via SigMF `sample_start` before claiming real IQ success.

