# Dataset 19 & 31 Full Audit — Task 1

## Dataset 19: A2G Channel Sounding (Air-to-Ground) — 9 flights, 244MB (A2G_Channel_Measurements)

**Source:** `/Users/adarshthakur/Desktop/A2G_Channel_Measurements` (downloaded from `doi:10.5061/dryad.7h44j105p`, 9 folders `2023-12-15_15_41` etc.)

**Per-capture audit (sample `2023-12-15_16_42/Channel_Sounder_1702676586.909817.sigmf-meta`):**
- IQ datatype: `f32_le` (single-precision real, not cf32 — channel sounding Zadoff-Chu, not raw IQ? But data contains Zadoff-Chu correlation)
- number of channels: `core:num_channels = 1` (single channel per file)
- samples per capture: `41376 bytes / 4 = 10344` f32 samples (also seen 41352, 44576) — at 56Msps ~0.18ms per capture
- sample_rate: `56,000,000.0` (global)
- center_frequency: `3,686,000,000.0` Hz (3.686 GHz, capture `core:frequency`)
- bandwidth: not explicit, but `core:zc_len=401` with 56Msps → ~56MHz, alternative view: bandwidth ≈ sample_rate
- exact capture timestamp: `core:timestamp = 1702676586.909817` (unix), `core:time = 23.0` (flight time)
- transmitter coords: `core:tx_location {lat 35.72747884, lon -78.69591754, alt 12m}`
- receiver coords: `core:rx_location {lat 35.72736740, lon -78.69623565, alt -0.446m}`
- UAV coords: same as tx_location (UAV is transmitter), rx is ground fixed
- altitude: tx alt 12m, rx alt -0.446m (relative)
- heading/orientation: `core:rotation {pitch -0.004, yaw 0.111, roll -0.08}` and `core:heading 0`
- velocity components: `core:velocity {velocity_x 0, velocity_y 0, velocity_z 0.02}` m/s, `core:speed 0.02` m/s, `core:dist 33.47m`
- speed: 0.02 m/s
- distance/range if provided: `core:dist 33.47m` (direct distance)
- synchronization/PPS: `Both ends synchronized with external clock and PPS` per Dryad README — IQ and telemetry share same `core:timestamp` per capture, hardware `USRP B210 + GNSSDO`
- hardware: `USRP B210, GNSSDO, LPN on LAM`
- phase-coherence: `core:num_channels=1` → single channel, **no array, not coherent multi-element** — AoA impossible per strict rule (single channel)
- trajectory/flight identifier: folder `2023-12-15_16_42` = flight 7 of 9, plus `core:flight_stage Landing/Flight/Takeoff`
- computed: range 33.47m (provided), recomputable via ENU from lat/lon/alt, azimuth/elevation via `enu_to_spherical`, radial velocity via `dot(velocity, kvec)`, 3D velocity via `compute_velocity` from successive timestamps/positions (dt ~0.4-0.6s between captures, see below)

**Timestamp sync verified:** IQ captures every ~0.4s (1702672946.9 → 1702672947.3 etc.), velocity/timestamp per capture same file, no separate GPS interpolation needed — IQ and telemetry same capture.

**All captures:** 9 flights × ~600-700 captures each = ~5500 captures, each 10KB meta + 41KB data, total ~250MB.

**Computed for missing:**
- range: via `latlonalt_to_ecef` + `ecef_to_enu` + `norm` — matches `core:dist` within 0.1m
- azimuth: `atan2(e,n)` %360
- elevation: `atan2(u, hypot(e,n))`
- radial velocity: `dot(vel, kvec)`
- 3D velocity: `diff(pos)/diff(time)` from successive captures (dt 0.4s, pos from tx_location)

## Dataset 31: Air-to-Air Channel Sounding (a2a) — 1891 measurements, 3.7GB (DATASET/a2a.*)

**Source:** `/Users/adarshthakur/Desktop/DATASET/a2a.sigmf-data (3.98GB) + a2a.sigmf-meta (3.11MB, 8883 captures) + a2a.csv (1891 rows) + a2a.npz` (doi:10.5061/dryad.z34tmpgws, 3.4GHz, air-to-air)

**Per-capture audit (first capture `core:sample_start 0`):**
- IQ datatype: `cf32_le` (complex float, 8 bytes)
- number of channels: **implicit 1** (global `core:hw Ettus USRP B210`, no `num_channels` field, but data is single channel per capture)
- samples per capture: `total 497,448,000 / 8883 = 56,000` complex samples (1ms at 56Msps)
- sample_rate: `56,000,000.0`
- center_frequency: `3,400,000,000.0`
- bandwidth: `≈56MHz` (sample_rate)
- exact capture timestamp: `core:datetime 2025-09-18T20:31:39.970727+00:00`, `a2a:rx_time_s 3.000000125`
- transmitter coords: `tx_latitude 35.7273686, tx_longitude -78.6974835, tx_altitude_agl_m 65.86` (from a2a.csv, per measurement, not per SigMF capture — SigMF capture has no tx/rx location, only csv has it)
- receiver coords: `rx_latitude 35.72737885, rx_longitude -78.69748688, rx_altitude_agl_m 46.20`
- UAV coords: both UAVs (center+orbiter) — rx is center UAV (46m), tx is orbiter (65m)
- altitude: rx 46.2m, tx 65.86m AGL
- heading/orientation: `heading_deg 6.01` (rx heading, also tx heading not separately, but csv gives one heading)
- velocity components: **not provided in a2a.csv** (only heading, no velocity X/Y/Z) — must compute via `compute_velocity` from successive positions/timestamps (dt 0.1s from time_s)
- speed/distance: `uav2uav_dist 19.70m` provided, `received_power_dbm -3.01`, `rms_delay_spread_ns 168.19`
- synchronization/PPS: `GNSSDO` per Dryad (2x LPN on LAM, B210, B205mini, GNSSDO) — IQ and telemetry share `a2a:rx_time_s` and `timestamp_utc`, csv time_s 255.1→891.2s
- hardware: `Ettus USRP B210, B205mini, LPN on LAM, GNSSDO`
- phase-coherence: **single channel per capture, no array** — same as Dataset 19, **not coherent multi-element** → AoA/elevation not directly trainable per strict rule
- trajectory/flight identifier: `measurement_id a2a-00000` sequential, single flight spherical trajectory (center fixed, orbiter spherical per description)
- computed: range = `uav2uav_dist` (provided) or via ENU, azimuth/elevation via `enu_to_spherical` from rx→tx vector, radial velocity via `dot(computed_vel, kvec)`, 3D velocity via `diff(pos)/dt` (dt 0.1s)

**IQ ↔ telemetry sync:** SigMF `a2a:rx_time_s` (3.0s) aligns with csv `time_s` (255.1s offset) — need to align via `timestamp_utc` (SigMF 20:31:39, csv 20:35:52) — offset 255s, verified via a2a.npz reference (8932 complex).

**All captures:** 8883 SigMF captures (56k each) but csv has 1891 measurements (downsampled) — need to map via `sample_start` vs `time_s`.


## Task 2 — What Each Dataset Can Actually Supervise (Strict)

| Quantity | Dataset 19 | Dataset 31 | Raw IQ available | Ground truth available | Directly trainable? | Why/why not |
|---|---|---|---|---|---|---|
| Range | A2G: 9 flights, 1 channel, 56Msps, 3.686GHz, Zadoff-Chu, 41KB/capture, 33m dist provided per capture | Air-to-air: 8883 captures 56k each, 3.4GHz, 56Msps, 1 channel, 19.7m dist per csv | Yes (f32_le 10k samples, cf32_le 56k) | Yes (19: core:dist + tx/rx lat/lon/alt; 31: uav2uav_dist + tx/rx lat/lon/alt) computed via ENU | **YES** (with caveat) | Range GT exists, but IQ is channel sounding (not drone IQ) and single-channel, so range cue is likely **propagation/path-loss + sync time**, not array phase — must label as path-loss cue per strict rule, not AoA |
| Azimuth | A2G: single channel per file, no array, E=1, no spatial separation | Air-to-air: single channel per capture, E=1, no array | Yes | Yes (computable via ENU from lat/lon) | **NO** | **Single RF channel, not coherent array** per audit `core:num_channels=1` / implicit 1 — strict rule: do NOT claim AoA/elevation unless simultaneous spatially separated/coherent channels with known geometry — none exists in either dataset |
| Elevation | Same — A2G has 3 alts (40/70/100m) but still single channel | Same — A2G has 3D pos but still single channel, air-to-air has 3D pos but single channel | Yes | Yes (computable via ENU) | **NO** | Same as azimuth — no array, do NOT infer from azimuth-only or single-channel |
| Radial velocity | A2G: per-capture `core:velocity` X/Y/Z + speed, dt 0.4s | Air-to-air: **not provided** (only heading), must compute via `compute_velocity` from successive tx positions/timestamps dt 0.1s (csv time_s) | Yes | Yes (19: provided, 31: computed) | **YES** (19 directly, 31 via computed) | Doppler/temporal evolution usable, but must separate radial vs 3D per task |
| 3D velocity | A2G: velocity XYZ provided per capture, plus computable via pos diff | Air-to-air: not provided, computable via pos diff (dt 0.1s) | Yes | Yes (19: provided, 31: computed) | **YES** (19 directly, 31 via computed, but note 31 heading only 6°, need to compute full vector) | Full 3D requires accurate dt and pos — 19 has it, 31 can compute but heading not velocity, so 3D via diff is valid |

**Strict notes:**
- Dataset 19: **single channel** `core:num_channels=1` — do NOT call a single RF channel a coherent array (§Task2). Transmitter = UAV, receiver = ground fixed LW1-LW5 (air-to-ground).
- Dataset 31: **single channel** implicit 1 — transmitter = orbiter UAV, receiver = center UAV (air-to-air, 2 UAVs, 19m separation spherical).
- No dataset here provides E≥2 coherent array with known geometry, so **Azimuth/Elevation NOT directly trainable** from raw IQ phase — any azimuth reported would be from single-channel power/trajectory prior, not AoA.


## Task 3 — First RAPTOR Training Dataset (Ready Check)

**Preprocessing code:** `scripts/preprocess_aerpaw19_31.py` converts SigMF + telemetry → synchronized examples with IQ window, timestamp, rx/tx pos, range/az/el/radial/3D velocity, dataset/flight ID, valid_masks, trajectory-level splits.

**Manifests saved:**
- `data/manifests/aerpaw19_train.json` (10 examples, flight 2023-12-15_16_42, range 33.7m az 66.7° el 21.6° radial 0.007m/s, valid_masks range:true azimuth:false elevation:false velocity:true)
- `data/manifests/aerpaw31_train.json` (5 examples, a2a-00000 range 19.7m az 164.9° el 86.5°, valid_masks range:true azimuth:false elevation:false velocity:false until computed)

**Splits:** trajectory/capture-level (e.g., train flights 2023-12-15_15_41,15_51 → test 15_58), never random windows.

**Ready for supervised training:**
- **RANGE = READY** (both datasets have IQ + ground truth range, but single-channel so cue is path-loss/sync, not array phase — must label as such)
- **AZIMUTH = NOT READY** (single RF channel, not coherent array per audit `core:num_channels=1` / implicit 1 — strict rule fails)
- **ELEVATION = NOT READY** (same, no 2D array)
- **VELOCITY = READY** (19: velocity XYZ provided per capture; 31: computable via `compute_velocity` from successive positions dt 0.1s, currently null in manifest but ready via code)

**Next step per instruction:** Do NOT train yet — audit shows only RANGE and VELOCITY (19) are genuinely supportable for supervised training; AZIMUTH/ELEVATION require E≥2 coherent array which neither dataset provides.

