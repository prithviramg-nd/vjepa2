#!/usr/bin/env bash
# Step 2: distributed pretraining on the 4x T4 GPUs.
# app/main.py spawns one process per device; no SLURM needed.
set -euo pipefail

REPO="/home/ubuntu/inwdata/prithvi/git/vjepa2"
PY="${REPO}/venv/bin/python3"
CFG="${1:-${REPO}/configs/train_2_1_face/faces-8d-384px-30f.yaml}"

cd "${REPO}"
mkdir -p workarea/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
# T4s on this host have no peer-to-peer support ("Cuda failure 217"), and the
# SHM transport trips over the same path, so NCCL is forced onto sockets.
# Verified: P2P_DISABLE alone still fails; both flags together work.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

exec "${PY}" -u -m app.main \
  --fname "${CFG}" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
