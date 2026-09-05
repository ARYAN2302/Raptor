# Dataset Audit — Phase 0 §5 (§8 Task 3-6) — No bulk download

## Links & Licenses
- RFUAV: https://github.com/kitoweeknd/RFUAV , https://huggingface.co/datasets/kitofrank/RFUAV (apache-2.0) — Paper arXiv:2503.09033. Dataset: ~1.3 TB raw from 37 UAVs per README §3.1, HuggingFace shows 19k rows (spectrogram images) + raw via HuggingFace 299 GB total (10K-100K rows). Modal already has raptor-data:/rfuav (DJI FPV COMBO only) + raptor-data:/rfuav_rar 11 rars 1.5-41 GB each (top ls earlier).
- UAVSig: https://cores.ee.ucla.edu/downloads/datasets/uavsig/ , Dataverse https://doi.org/10.25346/S6/LVRRAE. No license stated; paper T. Zhao et al MILCOM 2024 WHIRLS 2025. Script dataverse_load.py required (web UI unstable, zip overload).
- AERPAW-28: https://aerpaw.org/dataset/multi-modal-rf-sensor-and-radar-dataset-for-uav-tracking/ (AERPAW, NSF PAWR CNS-1939334). Data via Dryad https://datadryad.org/dataset/doi:10.5061/dryad.7d7wm3898 . Covers 33 AADM flights Lake Wheeler, Keysight N6841A + Fortem R20 + UAV GT.
- AERPAW-31: https://aerpaw.org/dataset/dataset-31-aerpaw-air-to-air-channel-sounding-measurement-with-uavs/ SigMF 3.4 GHz air-to-air — optional D, not first exp.
- Sionna: https://github.com/NVlabs/sionna (Apache 2.0) for simulation bridge.

## File Inventory (verified without full download)

### RFUAV
- GitHub claims 37 raw drone clips + xml per drone. HuggingFace: 37 classes, 19k spectrogram images (train 5.68k/val 13.4k). Raw IQ not in HuggingFace viewer; raw via HuggingFace 299 GB total.
- Modal raptor-data:/rfuav verified: DJI FPV COMBO / VTSBW=10 pack1.xml + pack1_0-1s.iq 799M, pack1_1-2s.iq 799M ... VTSBW=20 pack3.*, VTSBW=40 pack2.* ; each xml: DeviceType USRPX310, Drone DJI FPV COMBO, Serial 00007/00008, DataType Complex Float, ReferenceSNR 32/20, CenterFrequency 5760000000, SampleRate 100000000, IFBandwidth 100000000, ScaleFactor 90, SampleCount 100000000. I/Q data Complex Float little-endian interleaved.
- Modal raptor-data:/rfuav_rar 8 rars + ValidationSet_5Drones 3 rars (36-43 GB) not extracted — size calc: DJI FPV COMBO.rar 14.3 GB, MINI4 PRO 4.7 GB etc. — total rars ~99 GB archived + validation ~91 GB. Exact need: do not download all before loader test (§5 task).
- Raw subset documented in handoff: 37 raw drone clips + metadata (device,droneserial,SNR,cf,sr,IFBW,scale) — verified via xml fields.

### UAVSig
- Page states multiple drones/controllers, WHIRLS labeling, B205mini drops → random gaps; captures labeled, capture params accurate.
- Modal iris-raw-iq:/ 8 bins 457 MB each: Yuneec_typhoon_h_2G_1of2.bin, DJI_inspire_2_2G.bin etc. (from earlier ls). Modal iris-data:/iris_rfuav.h5 12.4 GB + drffr2.h5 327 MB preprocessed. Not yet web-downloaded full Dataverse — need script inspection for session structure before splits (§5 task).
- Still need: exact Dataverse file list via dataverse_load.py dry-run to get per-drone file counts, sample rates, durations, label csv columns.

### AERPAW-28
- Page: 33 flights, position estimates from Fortem radar + Keysight RF + UAV GT. Dryad link required to inventory. Claim: NOT raw coherent single-array IQ (§5 critical). Expected files: csv/json of radar tracks, RF geolocation estimates, GPS GT, not SigMF raw. Need to run Dryad file list (curl) to confirm before claiming supervision.
- AERPAW-31: SigMF raw per page, but channel sounding experiment, not drone detection — defer per §5 Optional D.

## Raw IQ Formats & Metadata
- RFUAV: Complex Float32 binary, xml sidecar with 9 fields (above). SigMF-like but custom xml; needs converter to canonical SigMF meta.
- UAVSig: int16 interleaved I/Q bins (per iris-raw-iq hex dump, scale /32768), gaps, WHIRLS labels separate.
- AERPAW-28: expected processed positions (lat/lon/alt, range/az) — no IQ.

## Label Availability Matrix (do not infer, only prove)
| dataset | raw IQ | range/az/el/velocity | emitter count | identity (same-model) | notes |
|---|---|---|---|---|---|
| RFUAV | yes (complex float) | no | no (single emitter per clip) | model-level (37 classes) serial per clip → no same-model multi-unit unless serials differ within class (to verify) | high SNR only |
| UAVSig | yes (int16 bins, gaps) | no | no | drone+controller per file, multiple units — test same-model where available | B205mini drops |
| AERPAW-28 | no (processed estimates) | yes (UAV GT position, radar/RF estimates) | yes (flight table) | flight ID | not raw-IQ supervised |
| AERPAW-31 | yes (SigMF) | yes (air-to-air geometry) | no | no | not drone-comms |
| Sionna synthetic | yes (generated) | yes (exact GT per §6) | yes (0-N) | yes (emitter_id) | bridge |

→ No public pair raw IQ → range/az/el/velocity exists (§1, §6, Q6).

## Known Leakage / Failure Modes
- RFUAV: serial = device ID → random window split leaks same serial across train/test → inflates SEI; must split by SerialNumber/device/site per handoff §12. SNR bias high only → poor low-SNR gen.
- UAVSig: session/receiver confound, B205mini drops create deterministic gaps that classifier can memorize; WHIRLS labels may leak via signal content; SNR dependence; must split by file/session, test cross-SNR K=10/50/100 per SNR as Radio-FM §IV-A4.
- AERPAW: TDOA vs single-array mismatch; do not fabricate AoA baseline if data lacks.

## Final Experimental Matrix (Gates §20)
- Gate A (19 Task10): RFUAV pretrain masked recon vs baselines A/B/C §10 → UAVSig linear probe cross-session — pass if raw-IQ > magnitude/spectrogram leakage-safe
- Gate B: 0-1-2 then 0-4 synthetic mixtures from real RFUAV recordings (first trivial arithmetic, not physical channel) → count acc / PR per emitter
- Gate C: Sionna LOS 1 emitter → multipath → multi-emitter → site variation, position RMSE + calibration — only if geometry observability proven (§16)
- Gate D: train sites A/B test unseen C — only if gap exists
- Ablations §11 table will be populated after Gate A.

Next: implement canonical loader with SigMF support + deterministic splits + stats plots (§8 Phase1) before any encoder.
