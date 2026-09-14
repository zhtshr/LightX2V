#!/usr/bin/env bash
# Fair Wan2.2-Distill TP=2/4/8 rerun: same config, no unload, multi-sample, interleaved order.
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
CONFIG="${CONFIG:-${ROOT}/configs/disagg/baseline/wan22_moe_i2v_tp_fair_bench.json}"
INPUTS_CACHE="${INPUTS_CACHE:-${STUDY}/phase1_encoder_inputs.pt}"
BENCH="${ROOT}/scripts/disagg/run_phase1_transformer_bench.py"
WARMUP="${WARMUP:-2}"
MEASURE="${MEASURE:-5}"
# Interleaved order reduces monotonic thermal / GPU state bias (2->4->8 trend artifact).
TP_ORDER="${TP_ORDER:-2 8 4}"

cleanup_gpus() {
  pkill -f "torch.distributed.run.*run_phase1_transformer_bench" 2>/dev/null || true
  sleep 3
}

run_tp() {
  local tp="$1"
  local tag="wan22_distill_tp${tp}_fair"
  local out="${STUDY}/${tag}.json"
  local log="${STUDY}/${tag}.log"

  cleanup_gpus
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((tp - 1)))"

  echo "=== Fair rerun TP=${tp} GPUs=${CUDA_VISIBLE_DEVICES} warmup=${WARMUP} measure=${MEASURE} ===" | tee "${log}"
  echo "config=${CONFIG}" | tee -a "${log}"

  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    "--nproc_per_node=${tp}" \
    "${BENCH}" \
    --config_json "${CONFIG}" \
    --model_path "${MODEL_PATH}" \
    --task i2v \
    --model_cls wan2.2_moe \
    --tensor_p_size "${tp}" \
    --seq_p_size 1 \
    --inputs_cache "${INPUTS_CACHE}" \
    --output_json "${out}" \
    --warmup "${WARMUP}" \
    --measure_iters "${MEASURE}" \
    2>&1 | tee -a "${log}"
}

for tp in ${TP_ORDER}; do
  run_tp "${tp}"
done

"${PYTHON}" "${ROOT}/scripts/disagg/summarize_wan22_tp_fair_latency.py" \
  --study_dir "${STUDY}"

echo "Done. See ${STUDY}/wan22_distill_tp_fair_comparison.md"
