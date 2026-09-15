#!/usr/bin/env bash
# Run Wan TP phase validations with torchrun.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TP_SIZE="${TP_SIZE:-2}"
MODEL_PATH_T2V="${MODEL_PATH_T2V:-$ROOT/models/lightx2v/Wan2.1-T2V-1.3B}"
MODEL_PATH_MOE="${MODEL_PATH_MOE:-$ROOT/models/lightx2v/Wan2.2-Distill-Models}"

run_phase() {
  local phase="$1"
  shift
  echo "========== Wan TP ${phase} (tp=${TP_SIZE}) =========="
  torchrun --standalone --nproc_per_node="${TP_SIZE}" \
    "scripts/disagg/validate_wan_tp_${phase}.py" \
    --tensor_p_size "${TP_SIZE}" "$@"
}

run_phase phase0 \
  --config_json configs/dist_infer/wan_t2v_tensorp.json \
  --model_path "$MODEL_PATH_T2V"

run_phase phase1 \
  --config_json configs/dist_infer/wan_t2v_tensorp.json \
  --model_path "$MODEL_PATH_T2V"

if [[ -d "$MODEL_PATH_T2V" ]]; then
  run_phase phase2 \
    --config_json configs/dist_infer/wan_t2v_tensorp.json \
    --model_path "$MODEL_PATH_T2V"
else
  echo "========== Wan TP phase2 skipped (no T2V model) =========="
  run_phase phase2 --skip_load \
    --config_json configs/dist_infer/wan_t2v_tensorp.json \
    --model_path "$MODEL_PATH_T2V"
fi

if [[ -d "$MODEL_PATH_MOE" ]]; then
  run_phase phase3 \
    --config_json configs/dist_infer/wan22_moe_i2v_tensorp.json \
    --model_path "$MODEL_PATH_MOE" \
    --task i2v \
    --model_cls wan2.2_moe \
    --model_type wan2.2_moe_high_noise
else
  echo "========== Wan TP phase3 skipped (no MoE model) =========="
fi

echo "All Wan TP validation phases completed."
