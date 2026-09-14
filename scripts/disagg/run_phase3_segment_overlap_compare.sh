#!/bin/bash
# Compare serial vs legacy vs segment overlap (MoE 480p SLA P=4).
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

baseline_conda_env=${BASELINE_CONDA_ENV:-lightx2v}
if [[ "${CONDA_DEFAULT_ENV:-}" != "${baseline_conda_env}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        set +u
        eval "$(conda shell.bash hook)"
        conda activate "${baseline_conda_env}"
        set -u
    fi
fi

config="${study_dir}/baseline_moe_i2v_480_sla_triton_seqp4.json"
out="${study_dir}/p3_segment_overlap_compare_moe_480_sla_seqp4.json"
log="${study_dir}/phase3_segment_overlap_compare.log"

torchrun --standalone --nproc_per_node=4 \
    "${lightx2v_path}/scripts/disagg/run_phase3_segment_overlap_compare.py" \
    --config_json "${config}" \
    --measure_steps 4 \
    --warmup_steps 1 \
    --output_json "${out}" \
    2>&1 | tee "${log}"

echo "results: ${out}"
