#!/bin/bash
# Dual-tenant a2a-overlap for Wan2.1-T2V-1.3B at configurable resolution.
# Usage:
#   RESOLUTIONS=256,512 SEQ_PS=4,6 bash scripts/disagg/run_phase3_dual_overlap_t2v_resolution.sh

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

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
sp_bench="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}
force=${FORCE_RERUN:-0}
resolutions=${RESOLUTIONS:-256,512}
seq_ps=${SEQ_PS:-4,6}
align_iters=${ALIGN_ITERS:-3}

run_case() {
    local res="$1"
    local p="$2"
    local config_json="${study_dir}/baseline_t2v_1.3b_${res}x${res}_seqp${p}.json"
    local inputs_cache="${study_dir}/phase1_t2v_1.3b_${res}x${res}_encoder_inputs.pt"
    local phase3_json="${study_dir}/p3_t2v_1.3b_${res}x${res}_sp_seqp${p}.json"
    local tag="p3_dual_overlap_t2v_1.3b_${res}x${res}_seqp${p}"
    local out_json="${study_dir}/${tag}.json"

    if [[ ! -f "${config_json}" ]]; then
        echo "=== skip ${tag}: missing ${config_json} (run phase1 resolution sweep first) ===" \
            | tee -a "${study_dir}/phase3_dual_overlap_t2v_res_runner.log"
        return 0
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== ${tag} res=${res} seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" \
        | tee -a "${study_dir}/phase3_dual_overlap_t2v_res_runner.log"

    if [[ ! -f "${phase3_json}" || "${force}" == "1" ]]; then
        "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
            "${sp_bench}" \
            --seq_p_size "${p}" \
            --task t2v \
            --model_cls wan2.1 \
            --model_path "${model_path}" \
            --config_json "${config_json}" \
            --inputs_cache "${inputs_cache}" \
            --output_json "${phase3_json}" \
            2>&1 | tee -a "${study_dir}/${tag}_sp.log" || true
    fi

    if [[ -f "${out_json}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out_json}')); sys.exit(0 if d.get('dual_a2a_overlap_s') else 1)"; then
            echo "=== skip ${tag} (exists) ===" | tee -a "${study_dir}/phase3_dual_overlap_t2v_res_runner.log"
            return 0
        fi
    fi

    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${bench}" \
        --seq_p_size "${p}" \
        --task t2v \
        --model_cls wan2.1 \
        --model_path "${model_path}" \
        --config_json "${config_json}" \
        --inputs_cache "${inputs_cache}" \
        --phase3_json "${phase3_json}" \
        --output_json "${out_json}" \
        --align_iters "${align_iters}" \
        2>&1 | tee -a "${study_dir}/${tag}.log"
}

IFS=',' read -ra RES_ARR <<< "${resolutions}"
IFS=',' read -ra P_ARR <<< "${seq_ps}"

for res in "${RES_ARR[@]}"; do
    for p in "${P_ARR[@]}"; do
        if ! python3 -c "import sys; sys.exit(0 if 12 % ${p} == 0 else 1)"; then
            echo "=== skip res=${res} P=${p}: num_heads=12 not divisible ===" \
                | tee -a "${study_dir}/phase3_dual_overlap_t2v_res_runner.log"
            continue
        fi
        run_case "${res}" "${p}"
    done
done

"${python_executable}" "${lightx2v_path}/scripts/disagg/summarize_t2v_dual_overlap_resolution.py"
