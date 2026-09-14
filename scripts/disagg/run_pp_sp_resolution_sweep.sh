#!/bin/bash
# PP×SP hybrid + quad at square resolutions (Wan2.2-MoE I2V, no offload, lps=2).
# Usage: RESOLUTIONS=1024,2048 bash scripts/disagg/run_pp_sp_resolution_sweep.sh

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LIGHTX2V_NCCL_TIMEOUT_S="${LIGHTX2V_NCCL_TIMEOUT_S:-10800}"

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
hybrid_bench="${lightx2v_path}/scripts/disagg/run_pp_sp_hybrid_bench.py"
quad_bench="${lightx2v_path}/scripts/disagg/run_pp_sp_quad_overlap_bench.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
force=${FORCE_RERUN:-0}
resolutions=${RESOLUTIONS:-256,512,1024,2048}
lps=${LPS:-2}

write_pp_config() {
    local res="$1"
    local sp="$2"
    local template="${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_sp${sp}_bench.json"
    local cfg="${study_dir}/baseline_moe_i2v_pp2_sp${sp}_${res}x${res}.json"
    python3 - <<PY
import json
from pathlib import Path
data = json.loads(Path("${template}").read_text(encoding="utf-8"))
data["target_height"] = ${res}
data["target_width"] = ${res}
data["enable_cfg"] = False
data["cpu_offload"] = False
data["unload_modules"] = False
data["pp_layers_per_stage"] = ${lps}
if ${res} >= 1024:
    data["vae_cpu_offload"] = True
    data["t5_cpu_offload"] = True
if ${res} >= 2048:
    data["use_tiling_vae"] = True
Path("${cfg}").write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
}

run_hybrid() {
    local res="$1"
    local sp="$2"
    local n_gpu=$((2 * sp))
    local tag="wan22_pp2_sp${sp}_hybrid_${res}x${res}_lps${lps}_m2"
    local out="${study_dir}/${tag}.json"
    local cfg="${study_dir}/baseline_moe_i2v_pp2_sp${sp}_${res}x${res}.json"
    local cache="${study_dir}/phase1_moe_i2v_${res}x${res}_encoder_inputs.pt"

    if [[ ! -f "${cache}" ]]; then
        echo "=== skip ${tag}: missing encoder cache ${cache} ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"
        return 0
    fi

    write_pp_config "${res}" "${sp}"

    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('single_transformer_s') else 1)"; then
            echo "=== skip ${tag} (exists) ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"
            return 0
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpu - 1)))"
    echo "=== ${tag} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"

    if ! "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${n_gpu}" \
        "${hybrid_bench}" \
        --config_json "${cfg}" \
        --model_path "${model_path}" \
        --inputs_cache "${cache}" \
        --layers_per_stage "${lps}" \
        --microbatch_list 2 \
        --per-request-latency \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/${tag}.log"; then
        echo "=== FAILED ${tag} (see ${tag}.log) ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"
    fi
}

run_quad() {
    local res="$1"
    local sp="$2"
    local n_gpu=$((2 * sp))
    local layout="oct"
    [[ "${sp}" -ge 4 ]] && layout="dual"
    local tag="wan22_pp2_sp${sp}_quad_${res}x${res}_lps${lps}_${layout}"
    local out="${study_dir}/${tag}.json"
    local cfg="${study_dir}/baseline_moe_i2v_pp2_sp${sp}_${res}x${res}.json"
    local cache="${study_dir}/phase1_moe_i2v_${res}x${res}_encoder_inputs.pt"

    if [[ ! -f "${cache}" ]]; then
        echo "=== skip ${tag}: missing encoder cache ${cache} ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"
        return 0
    fi

    write_pp_config "${res}" "${sp}"

    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('single_transformer_s') else 1)"; then
            echo "=== skip ${tag} (exists) ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"
            return 0
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpu - 1)))"
    echo "=== ${tag} gpus=${CUDA_VISIBLE_DEVICES} layout=${layout} ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"

    if ! "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${n_gpu}" \
        "${quad_bench}" \
        --config_json "${cfg}" \
        --model_path "${model_path}" \
        --inputs_cache "${cache}" \
        --layers_per_stage "${lps}" \
        --quad-layout "${layout}" \
        --per-request-latency \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/${tag}.log"; then
        echo "=== FAILED ${tag} (see ${tag}.log) ===" | tee -a "${study_dir}/pp_sp_res_sweep_runner.log"
    fi
}

IFS=',' read -ra RES_ARR <<< "${resolutions}"

for res in "${RES_ARR[@]}"; do
    run_hybrid "${res}" 2
    run_quad "${res}" 2
    run_hybrid "${res}" 4
    run_quad "${res}" 4
done

"${python_executable}" "${lightx2v_path}/scripts/disagg/summarize_pp_sp_resolution.py"
