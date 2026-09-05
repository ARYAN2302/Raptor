#!/bin/bash
set -e
echo "Step 12: one emitter LOS via Sionna synthetic, full RAPTOR, small config for speed"
PYTHONPATH=/tmp/raptor_build/Raptor python3 scripts/train_raptor.py
PYTHONPATH=/tmp/raptor_build/Raptor python3 scripts/evaluate.py
