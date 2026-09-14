#!/bin/bash
# Fine-grained self-attn comm breakdown: MoE 480p dense vs SLA, P=4.
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

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

bench="${lightx2v_path}/scripts/disagg/run_phase3_layer_self_attn_comm_detail_profile.py"
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
nproc=4

run_one() {
    local config_json="$1"
    local tag="$2"
    local out_json="$3"
    echo "=== comm detail profile: ${tag} ===" | tee -a "${study_dir}/phase3_self_attn_comm_detail.log"
    torchrun --standalone --nproc_per_node="${nproc}" "${bench}" \
        --config_json "${config_json}" \
        --tag "${tag}" \
        --inputs_cache "${inputs_cache}" \
        --layer_start 1 \
        --layer_end 40 \
        --warmup_layers 2 \
        --output_json "${out_json}" \
        2>&1 | tee -a "${study_dir}/phase3_self_attn_comm_detail.log"
}

run_one "${study_dir}/baseline_seqp4.json" \
    "moe_dense" \
    "${study_dir}/p3_self_attn_comm_detail_moe_480_dense_seqp4.json"

run_one "${study_dir}/baseline_moe_i2v_480_sla_triton_seqp4.json" \
    "moe_sla" \
    "${study_dir}/p3_self_attn_comm_detail_moe_480_sla_seqp4.json"

echo "done: ${study_dir}/phase3_self_attn_comm_detail.log"
