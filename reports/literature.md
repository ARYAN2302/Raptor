# Literature Notes — Phase 0 (read before locking architecture)

## Radio-FM (P1) — raw IQ foundation
- Input: complex I/Q as dual-channel, 1D conv patch tokenizer, masked recon (BERT-style), heterogeneous datasets variable lengths, sample_rate/center_freq as metadata not invariance claim.
- Takeaway: keep I+Q separate learned channels, preserve phase, metadata-aware resampling. **No VQ first** (§2.1).

## Perceiver (P1)
- Asymmetric latent bottleneck: large token stream → fixed L latents via cross-attention. Proven for huge inputs. Antenna elements are NOT exchangeable — feed geometry as pos emb.

## Mamba (P1) — selective SSM
- Recurrence with selective gating, streaming, long context. Hypothesis: temporal context may help range/velocity, but must ablate (single-window vs short vs long).

## DETR (P1)
- Set prediction via learned queries + Hungarian matching, null/no-object. No NMS. Use for 0..N emitters; existence + regression + uncertainty + optional identity.

## Channel2World / NeRF² (P2)
- Environment-level latent, site conditioning as **branch not V1 prereq**. Only add after measured cross-site gap (Gate D).

## Failure modes to guard
- Same-model identity: receiver/channel/session leakage inflates accuracy. SNR dependence. Split by session/device.
- Range from single passive node is not directly observable — treat as testable hypothesis, need temporal/array priors.
