#!/bin/bash
# Mixed-resolution SP dual a2a overlap: tenant A=512², tenant B=1024² (Wan2.2-MoE I2V).
# Usage: SEQ_PS=2,4,8 bash scripts/disagg/run_phase3_dual_overlap_moe_mixed_512_1024.sh

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

python_executable=${BASELINE_PYTHON_EXECUTABLE:-python}
bench="${lightx2v_path}/scripts/disagg/run_phase3_dual_overlap_bench.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
force=${FORCE_RERUN:-0}
seq_ps=${SEQ_PS:-2,4,8}
align_iters=${ALIGN_ITERS:-1}

cache_512="${study_dir}/phase1_moe_i2v_512x512_encoder_inputs.pt"
cache_1024="${study_dir}/phase1_moe_i2v_1024x1024_encoder_inputs.pt"

for c in "${cache_512}" "${cache_1024}"; do
    if [[ ! -f "${c}" ]]; then
        echo "missing encoder cache: ${c}" >&2
        exit 1
    fi
done

IFS=',' read -ra P_ARR <<< "${seq_ps}"

for p in "${P_ARR[@]}"; do
    if ! python3 -c "import sys; sys.exit(0 if 40 % ${p} == 0 else 1)"; then
        echo "=== skip P=${p}: num_heads=40 not divisible ===" | tee -a "${study_dir}/phase3_moe_mixed_512_1024.log"
        continue
    fi

    config_json="${study_dir}/baseline_moe_i2v_1024x1024_seqp${p}.json"
    if [[ ! -f "${config_json}" ]]; then
        config_json="${study_dir}/baseline_moe_i2v_512x512_seqp${p}.json"
    fi
    if [[ ! -f "${config_json}" ]]; then
        echo "=== skip P=${p}: missing config ===" | tee -a "${study_dir}/phase3_moe_mixed_512_1024.log"
        continue
    fi

    phase3_json="${study_dir}/p3_moe_i2v_1024x1024_sp_seqp${p}.json"
    if [[ ! -f "${phase3_json}" ]]; then
        phase3_json="${study_dir}/p3_moe_i2v_512x512_sp_seqp${p}.json"
    fi
    if [[ ! -f "${phase3_json}" ]]; then
        phase3_json="${study_dir}/p3_sp_seqp${p}.json"
    fi

    tag="p3_dual_overlap_moe_i2v_mixed_512_1024_seqp${p}"
    out_json="${study_dir}/${tag}.json"

    if [[ -f "${out_json}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out_json}')); sys.exit(0 if d.get('dual_a2a_overlap_s') and not d.get('error') else 1)"; then
            echo "=== skip ${tag} (exists) ===" | tee -a "${study_dir}/phase3_moe_mixed_512_1024.log"
            continue
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== ${tag} P=${p} A=512² B=1024² gpus=${CUDA_VISIBLE_DEVICES} ===" \
        | tee -a "${study_dir}/phase3_moe_mixed_512_1024.log"

    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${bench}" \
        --seq_p_size "${p}" \
        --task i2v \
        --model_cls wan2.2_moe \
        --model_path "${model_path}" \
        --config_json "${config_json}" \
        --inputs_cache "${cache_512}" \
        --inputs_cache_b "${cache_1024}" \
        --resolution_a "512x512" \
        --resolution_b "1024x1024" \
        --phase3_json "${phase3_json}" \
        --output_json "${out_json}" \
        --align_iters "${align_iters}" \
        2>&1 | tee -a "${study_dir}/${tag}.log"
done

echo "=== mixed 512+1024 sweep done ===" | tee -a "${study_dir}/phase3_moe_mixed_512_1024.log"
