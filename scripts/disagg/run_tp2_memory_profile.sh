#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/root/install/miniconda3/envs/lightx2v/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "${ROOT}/scripts/disagg/run_tp2_memory_profile.py" "$@"
