#!/usr/bin/env bash
# Wan2.2 480×832×32f TP=2 fine-grained dual overlap throughput test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY="${ROOT}/save_results/optimization_study"
mkdir -p "${STUDY}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/root/install/miniconda3/envs/lightx2v/bin/python}"
BENCH="${ROOT}/scripts/disagg/run_wan22_tp2_fine_dual_overlap_bench.py"
P1="${ROOT}/scripts/disagg/run_phase1_transformer_bench.py"

CFG="${ROOT}/configs/disagg/baseline/wan22_moe_i2v_480_f32_tp2_fair.json"
INPUTS="${STUDY}/phase1_encoder_inputs_480_f32.pt"
OUT="${STUDY}/wan22_480_f32_tp2_fine_dual_overlap.json"

pkill -f "torch.distributed.run.*run_wan22_tp2_fine_dual_overlap" 2>/dev/null || true
sleep 2

if [[ ! -f "${INPUTS}" ]]; then
  echo "=== Preparing 480×832×32f inputs cache ==="
  export CUDA_VISIBLE_DEVICES=0
  "${PYTHON}" "${P1}" \
    --config_json "${CFG}" \
    --model_path "${ROOT}/models/lightx2v/Wan2.2-Distill-Models" \
    --task i2v --model_cls wan2.2_moe \
    --tensor_p_size 1 --seq_p_size 1 \
    --inputs_cache "${INPUTS}" --refresh_inputs_cache \
    --output_json "${STUDY}/wan22_480_f32_inputs_build.json" \
    --warmup 0 --measure_iters 0 2>&1 | tail -10
fi

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "${BENCH}" \
  --tensor_p_size 2 \
  --config_json "${CFG}" \
  --inputs_cache "${INPUTS}" \
  --output_json "${OUT}" \
  --warmup 1 \
  --measure_steps 0 \
  2>&1 | tee "${STUDY}/wan22_480_f32_tp2_fine_dual_overlap.log"

echo "Done: ${OUT}"
