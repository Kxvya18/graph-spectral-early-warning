#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
export PYTHONPYCACHEPREFIX="/private/tmp/graph_spectral_pycache"
export PYTHONUNBUFFERED=1
mkdir -p results/chbmit_v3_ml
CHBMIT_V3_LOG="results/chbmit_v3_ml/full_run.log"
exec > >(tee -a "$CHBMIT_V3_LOG") 2>&1

echo "[CHB-MIT v3] Starting or resuming the frozen cohort download"
.venv/bin/python src/download_chbmit_locked.py \
  --config configs/chbmit_v3_ml_protocol.json

echo "[CHB-MIT v3] Download complete; starting frozen cohort analysis"
.venv/bin/python src/run_chbmit_v3_ml.py \
  --config configs/chbmit_v3_ml_protocol.json

echo "[CHB-MIT v3] Full workflow complete"
