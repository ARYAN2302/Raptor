# Literature Notes — Handoff §4 (read before locking arch)

## Radio-FM (Prio 1) arXiv:2608.05793
- Focus per handoff: raw complex I/Q handling; dual-channel; masked recon; tokenization; heterogeneous variable lengths; emitter-ID
- Input: r[n]=I[n]+jQ[n] length N -> RMS norm σ=sqrt(1/N Σ|r|^2), r'=r/σ -> 1D conv per channel patch P=8 stride S -> L=floor((N-P)/S)+1 -> H∈R^{B×2×(L+1)×D} (CLS per channel) Fig3, Eq3
- Dual-level attention: intra-channel MHSA RoPE QK^T/sqrt(dk) (Eq4, position-preserving original indices) + inter-channel GAP s_c=1/(L+1) Σ H_{c,t} (Eq5) -> α=softmax((SWq)(SWk)^T/√dh) (Eq6) -> Ĥ=H+Σ α·HWv (Eq7) + LayerScale γ=1e-5 + DropPath p_i=p_max*(i-1)/(L_enc-1) Eq8-9
- Pretraining: Channel-Independent Masking (CI) per channel, 60% ratio (Fig9b), MSE on masked only ℒ=1/|M| Σ||ŷ-y||2^2 (Eq10); token-budgeted dynamic batching Σ|x|≤C_max, B=floor(C_max/2L) Eq11 + dataset sampling + sharding Fig2
- Datasets: 15 pretrain (Table I) 128-4096 length, 8×/2× oversampling, 1MHz/5MHz/10MHz/25MHz/100MHz, 468k-9.9M each, total diverse; 15 downstream Table II
- Evaluation: Accuracy/Precision/F1 Table IV, few-shot K=10/50/100 per SNR Table V, SOTA 13/15 full-FT, emergent t-SNE clustering Fig5-7; patch 8 peak Fig9c, CI > shared +2%, mask 90% collapses
- Novel: dual-channel independence + lightweight inter-channel α vs SpectrumFM (trunc 128) / EMind (adaptive packing, joint I/Q); scaling Tiny 2.38M→XLarge 114M
- Assumption tied to acquisition: sample_rate/bandwidth/center_freq vary per dataset (Table I Sample Rate / Oversampling) — not invariant, handled via RMS + RoPE + dynamic batching, not claimed invariant

## Perceiver (Prio 1) PMLR139 jaegle21a
- Why bottleneck: O(MN)+O(LN^2) vs O(LM^2); M=50k pixels (224×224), N=512 latents decouple depth from input size; 48 latent blocks possible
- Input: byte array X∈R^{M×C_in} + Fourier pos [sin(f_kπx),cos(...)] K=64, modality emb; permutation-equivariant, pos injected only via features — permuted ImageNet Perceiver 78.0→78.0 vs ViT 76.7→61.7
- Novel: asymmetric cross-attention iteratively distills X→Z, weight-shared latent Transformers + cross-attends (2..K share) to scale; latent = learned aggregation slots

## Mamba (Prio 1) 2312.00752
- Selective SSM: make Δ,B,C functions of input x → time-varying S6 vs LTI S4; Δ=softplus(param+Linear1), B,C=Linear_N(x); breaks convolution → hardware-aware parallel scan (kernel fusion, no HBM materialization) O(L) vs O(L^2)
- Selectivity = content gating: Δ~keep/forget, B/C~what to write/read; solves Selective Copying & Induction Heads where LTI fails (random spacing, associative recall)
- For RAPTOR: test hypothesis that temporal context recovers range via Doppler evolution / multipath memory; do not claim solves range; ablate single-window vs Mamba

## DETR (Prio 1) 2005.12872
- Set prediction: N=100 learned object queries (>> avg 7 COCO) → decoder parallel → FFN box+class + ∅ no-object; Hungarian matching σ̂=argmin Σ ℒ_match where ℒ_match=-1_{c≠∅}p(c)+1_{c≠∅}ℒ_box, ℒ_box=λ_iou*L_gIoU+λ_L1*||b-b̂||1; Hungarian loss Σ[-log p(c)+1_{c≠∅}ℒ_box] down-weight ∅ ×0.1
- No NMS/anchors; auxiliary loss per decoder layer; needs long schedule 300-500ep AdamW

## Channel2World (Prio 2) 2608.17544
- Env-level latent Z_e∈R^{Kz×Dm} from MIMO multipath tokens (delay, AoA, gain + pos) via Transformer encoder with alternating cross/self-attention; context-query pretraining predicting UE position/gain on disjoint query channels; 26k env ×5k channels Sionna ray-tracing; frozen transfer to unseen env competitive vs site-specific fine-tune
- Inspiration not drop-in: RFUAV is raw IQ, not decomposed paths

## NeRF2 (Prio 2) 2305.06118
- Per-scene neural radiance field: complex MLPs δ,a/θ + EM ray tracing Eq9-14; learns attenuation/retransmission voxel 1/8 λ; turbo-learning mixes true+synthetic → +50% (AoA 3.78°→1.96°, loc 2.52m→1.41m); requires OptiTrack, 10k positions, 10h/scene, no cross-scene transfer
- Use as synthetic data generator idea, not module

## RF fingerprinting / UAVSig leakage traps
- UAVSig: B205mini drops → random gaps; WHIRLS labels; dataset multiple drones/controllers 2024 paper; leakage if split by window within same capture → 99% inflated; must split by recording/session/device, test cross-session and SNR stratified K per SNR; channel/receiver confound

## Passive localization
- AERPAW8: 4 sensors TDOA, not single-array coherent IQ — reference only
- AERPAW28: 33 AADM flights, Fortem radar + Keysight RF position estimates + UAV GT via Dryad (not raw IQ) — processed outputs
- AERPAW31: 3.4GHz SigMF air-to-air channel sounding — raw IQ but geometry experiment, not drone-comms localization; do not mislabel
