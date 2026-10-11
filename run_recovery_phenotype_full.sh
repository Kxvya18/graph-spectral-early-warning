#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
export PYTHONPYCACHEPREFIX="/private/tmp/graph_spectral_recovery_pycache"
export PYTHONUNBUFFERED=1
mkdir -p results/chbmit_recovery_phenotype
log="results/chbmit_recovery_phenotype/full_run.log"
exec > >(tee -a "$log") 2>&1

echo "[Recovery phenotype] Starting or resuming the frozen CHB-MIT download"
.venv/bin/python src/download_chbmit_locked.py \
  --config configs/chbmit_v3_ml_protocol.json

echo "[Recovery phenotype] Download complete; starting locked real-data analysis"
.venv/bin/python src/run_chbmit_recovery.py \
  --config configs/chbmit_recovery_phenotype.json

echo "[Recovery phenotype] Full workflow complete"
