# RAPTOR — Raw-IQ RF Perception for C-UAS

```
RAW coherent IQ [T × E × I/Q]
        ↓
Complex IQ Tokenizer (dual-channel, phase-preserving)
        ↓
Perceiver latent bottleneck
        ↓
Temporal state (Mamba/SSM)
        ↓
DETR set decoder → {existence, range, az, el, velocity, uncertainty, identity}
```

**V1 output is position+state, not identity. Identity is optional.**

Paper spine: Radio-FM (IQ representation) · Perceiver (large-input bottleneck) · Mamba (temporal) · DETR (variable-cardinality) · Sionna (synthetic GT).

## Repo layout (§14)

```
configs/{datasets,model,training,experiments}/
src/{io,datasets,preprocessing,models,losses,simulation,evaluation,utils}/
scripts/{inspect_datasets,build_manifest,pretrain,train_state,evaluate}.py
modal_app.py  ← Modal training (volumes: raptor-data, raptor-ckpt)
reports/{dataset_audit,literature,experiment_log}.md
```

## Modal (essential data only)

Existing volumes already hold what Phases 1-6 need — no full re-download:

- `raptor-data:/rfuav` — RFUAV subset (DJI FPV COMBO, VTSBW=10/20/40, 800 MB/chunk) + 11 rars for later
- `iris-raw-iq` — 8× 457 MB UAVSig bins (DJI/Parrot/Yuneec)
- `iris-data:/iris_rfuav.h5` (12.4 GB) — preprocessed RFUAV manifest

```bash
python3 -m modal token set --token-id ak-MqJJzGohPrfaXsSDkEoDhH --token-secret as-Urix11LVcYuOSDHFiiQ3fU
python3 -m modal run modal_app.py::pretrain --config configs/training/pretrain.yaml
python3 -m modal run modal_app.py::train_state --config configs/training/state.yaml
```

Local:
```bash
pip install -e ".[dev]"
pytest -q
python scripts/inspect_datasets.py --local  # reads /data volume if mounted else synthetic
python scripts/pretrain.py --config configs/training/pretrain.yaml --dry
```

## Phases (Handoff §8)

- **P0** audit → `reports/dataset_audit.md`
- **P1** canonical SigMF IQ loader `[T,E,2]` + metadata preservation
- **P2** masked recon encoder (Radio-FM-inspired, no VQ first)
- **P3** representation eval (emitter presence / same-model ID)
- **P4** 0-2/0-4 synthetic mixtures + counting
- **P5** temporal ablations (single-window vs Mamba)
- **P6** DETR state decoder (Hungarian, az wrap, heteroscedastic uncertainty)

See `reports/literature.md` for P1 paper notes before locking architecture.
