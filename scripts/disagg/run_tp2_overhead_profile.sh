#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/root/install/miniconda3/envs/lightx2v/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "${ROOT}/scripts/disagg/run_tp2_overhead_profile.py" \
  --tensor_p_size 2 \
  --config_json "${ROOT}/configs/disagg/baseline/wan22_moe_i2v_tp_fair_bench.json" \
  --output_json "${ROOT}/save_results/optimization_study/wan22_tp2_overhead_profile.json" \
  --output_md "${ROOT}/save_results/optimization_study/wan22_tp2_overhead_analysis.md"
