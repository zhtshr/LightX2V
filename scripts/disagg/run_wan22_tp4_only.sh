#!/usr/bin/env bash
# Standalone TP=4 benchmark. Default: unload_modules=true (fits A10 23GB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY="${ROOT}/save_results/optimization_study"
mkdir -p "${STUDY}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/root/install/miniconda3/envs/lightx2v/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/disagg/baseline/wan22_moe_i2v_tp_fair_unload_bench.json}"
WARMUP="${WARMUP:-2}"
MEASURE="${MEASURE:-5}"
OUT="${OUT:-${STUDY}/wan22_distill_tp4_fair.json}"

pkill -f "torch.distributed.run.*run_phase1_transformer_bench" 2>/dev/null || true
sleep 2

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== TP=4 standalone GPUs=${CUDA_VISIBLE_DEVICES} warmup=${WARMUP} measure=${MEASURE} ==="
echo "config=${CONFIG}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${ROOT}/scripts/disagg/run_phase1_transformer_bench.py" \
  --config_json "${CONFIG}" \
  --model_path "${ROOT}/models/lightx2v/Wan2.2-Distill-Models" \
  --task i2v \
  --model_cls wan2.2_moe \
  --tensor_p_size 4 \
  --seq_p_size 1 \
  --inputs_cache "${STUDY}/phase1_encoder_inputs.pt" \
  --output_json "${OUT}" \
  --warmup "${WARMUP}" \
  --measure_iters "${MEASURE}"

echo "wrote ${OUT}"
