#!/usr/bin/env bash
# Wan2.2-Distill-Models real tensor-parallel latency sweep (TP=1,2,4,8).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY="${ROOT}/save_results/optimization_study"
mkdir -p "${STUDY}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/root/install/miniconda3/envs/lightx2v/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python
fi

MODEL_PATH="${MODEL_PATH:-${ROOT}/models/lightx2v/Wan2.2-Distill-Models}"
CONFIG_TP="${CONFIG_TP:-${ROOT}/configs/disagg/baseline/wan22_moe_i2v_tp_bench.json}"
CONFIG_TP1="${CONFIG_TP1:-${ROOT}/configs/disagg/baseline/wan22_moe_i2v_tp1_offload_baseline.json}"
INPUTS_CACHE="${INPUTS_CACHE:-${STUDY}/phase1_encoder_inputs.pt}"
BENCH="${ROOT}/scripts/disagg/run_phase1_transformer_bench.py"
WARMUP="${WARMUP:-1}"
MEASURE="${MEASURE:-3}"
FORCE="${FORCE_RERUN:-0}"

run_tp() {
  local tp="$1"
  local tag="wan22_distill_tp${tp}_real"
  local out="${STUDY}/${tag}.json"
  local log="${STUDY}/${tag}.log"
  local cfg="${CONFIG_TP}"
  if [[ "${tp}" -eq 1 ]]; then
    cfg="${CONFIG_TP1}"
  fi

  if [[ -f "${out}" && "${FORCE}" != "1" ]]; then
    echo "=== skip TP=${tp} (exists: ${out}) ==="
    return 0
  fi

  local -a cmd=(
    "${PYTHON}" "${BENCH}"
    --config_json "${cfg}"
    --model_path "${MODEL_PATH}"
    --task i2v
    --model_cls wan2.2_moe
    --tensor_p_size "${tp}"
    --seq_p_size 1
    --inputs_cache "${INPUTS_CACHE}"
    --output_json "${out}"
    --warmup "${WARMUP}"
    --measure_iters "${MEASURE}"
  )

  if [[ "${tp}" -gt 1 ]]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((tp - 1)))"
    cmd=(
      "${PYTHON}" -m torch.distributed.run
      --standalone
      "--nproc_per_node=${tp}"
      "${BENCH}"
      --config_json "${cfg}"
      --model_path "${MODEL_PATH}"
      --task i2v
      --model_cls wan2.2_moe
      --tensor_p_size "${tp}"
      --seq_p_size 1
      --inputs_cache "${INPUTS_CACHE}"
      --output_json "${out}"
      --warmup "${WARMUP}"
      --measure_iters "${MEASURE}"
    )
  else
    export CUDA_VISIBLE_DEVICES=0
  fi

  echo "=== Wan2.2-Distill TP=${tp} GPUs=${CUDA_VISIBLE_DEVICES} ===" | tee "${log}"
  "${cmd[@]}" 2>&1 | tee -a "${log}"
}

for tp in 1 2 4 8; do
  run_tp "${tp}"
done

"${PYTHON}" "${ROOT}/scripts/disagg/summarize_wan22_tp_latency.py" \
  --study_dir "${STUDY}" \
  --output "${STUDY}/wan22_distill_tp_scaling_real.md"

echo "Done. See ${STUDY}/wan22_distill_tp_scaling_real.md"
