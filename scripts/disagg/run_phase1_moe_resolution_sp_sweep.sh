#!/bin/bash
# Wan2.2-MoE I2V: Ulysses SP scaling across resolutions.
# Resolutions: 256², 512², 1024², 2048² | seq_p: 1, 2, 4, 8 (num_heads=40)

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 2048² MoE denoise can exceed the default 10 min NCCL watchdog per collective.
export LIGHTX2V_NCCL_TIMEOUT_S="${LIGHTX2V_NCCL_TIMEOUT_S:-10800}"

rm -f "${HOME}/.cache/torch_extensions/py312_cu128/quant_cuda/lock" \
    "${HOME}/.cache/torch_extensions/py312_cu128/quant_cuda/.ninja_lock" 2>/dev/null || true

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
bench="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
image_path=${IMAGE_PATH:-/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg}
force=${FORCE_RERUN:-0}
resolutions=${RESOLUTIONS:-256,512,1024,2048}
seq_ps=${SEQ_PS:-1,2,4,8}
num_heads=${NUM_HEADS:-40}

write_config() {
    local res="$1"
    local p="$2"
    local cfg="${study_dir}/baseline_moe_i2v_${res}x${res}_seqp${p}.json"
    python3 - <<PY
import json
from pathlib import Path

base = Path("${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_baseline.json")
data = json.loads(base.read_text(encoding="utf-8"))
data["target_height"] = ${res}
data["target_width"] = ${res}
data["enable_cfg"] = False
data["cpu_offload"] = True
data["offload_granularity"] = "block"
# Encoder prep is excluded from transformer_compute_s; offload T5/VAE for 1024+ to fit 24GB.
if ${res} >= 1024:
    data["t5_cpu_offload"] = True
    data["vae_cpu_offload"] = True
else:
    data["t5_cpu_offload"] = data.get("t5_cpu_offload", False)
    data["vae_cpu_offload"] = data.get("vae_cpu_offload", False)
if ${res} >= 2048:
    data["use_tiling_vae"] = True
data["parallel"] = {"seq_p_size": ${p}, "seq_p_attn_type": "ulysses"}
Path("${cfg}").write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
}

prepare_encoder_cache() {
    local res="$1"
    local cache="${study_dir}/phase1_moe_i2v_${res}x${res}_encoder_inputs.pt"
    if [[ -f "${cache}" && "${force}" != "1" ]]; then
        return 0
    fi
    write_config "${res}" 1
    export CUDA_VISIBLE_DEVICES=0
    echo "=== prepare encoder cache res=${res} ===" | tee -a "${study_dir}/phase1_moe_res_sp_runner.log"
    local -a enc_cmd=(
        "${python_executable}" "${bench}"
        --task i2v
        --model_cls wan2.2_moe
        --model_path "${model_path}"
        --config_json "${study_dir}/baseline_moe_i2v_${res}x${res}_seqp1.json"
        --image_path "${image_path}"
        --inputs_cache "${cache}"
        --encoder_only
    )
    if [[ "${force}" == "1" ]]; then
        enc_cmd+=(--refresh_inputs_cache)
    fi
    "${enc_cmd[@]}" 2>&1 | tee -a "${study_dir}/phase1_moe_i2v_${res}x${res}_encoder.log"
}

run_case() {
    local res="$1"
    local p="$2"
    local tag="p1_moe_i2v_${res}x${res}_seqp${p}"
    local out="${study_dir}/${tag}.json"
    local cfg="${study_dir}/baseline_moe_i2v_${res}x${res}_seqp${p}.json"
    local cache="${study_dir}/phase1_moe_i2v_${res}x${res}_encoder_inputs.pt"

    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip ${tag} (exists) ===" | tee -a "${study_dir}/phase1_moe_res_sp_runner.log"
            return 0
        fi
    fi

    write_config "${res}" "${p}"
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== ${tag} res=${res} seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_moe_res_sp_runner.log"

    local -a cmd=(
        "${python_executable}" "${bench}"
        --task i2v
        --model_cls wan2.2_moe
        --model_path "${model_path}"
        --config_json "${cfg}"
        --seq_p_size "${p}"
        --image_path "${image_path}"
        --inputs_cache "${cache}"
        --output_json "${out}"
        --warmup 1
        --measure_iters 1
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${bench}"
            --task i2v
            --model_cls wan2.2_moe
            --model_path "${model_path}"
            --config_json "${cfg}"
            --seq_p_size "${p}"
            --image_path "${image_path}"
            --inputs_cache "${cache}"
            --output_json "${out}"
            --warmup 1
            --measure_iters 1
        )
    fi

    if ! "${cmd[@]}" 2>&1 | tee -a "${study_dir}/${tag}.log"; then
        python3 - <<PY
import json
from pathlib import Path
out = Path("${out}")
out.write_text(json.dumps({
    "resolution": ${res},
    "seq_p_size": ${p},
    "error": "run_failed",
}, indent=2) + "\\n", encoding="utf-8")
PY
        echo "=== FAILED ${tag} ===" | tee -a "${study_dir}/phase1_moe_res_sp_runner.log"
        return 0
    fi
}

IFS=',' read -ra RES_ARR <<< "${resolutions}"
IFS=',' read -ra P_ARR <<< "${seq_ps}"

for res in "${RES_ARR[@]}"; do
    if [[ "${res}" -ge 1024 ]]; then
        prepare_encoder_cache "${res}"
    fi
    for p in "${P_ARR[@]}"; do
        if ! python3 -c "import sys; sys.exit(0 if ${num_heads} % ${p} == 0 else 1)"; then
            echo "=== skip res=${res} P=${p}: num_heads=${num_heads} not divisible ===" \
                | tee -a "${study_dir}/phase1_moe_res_sp_runner.log"
            continue
        fi
        run_case "${res}" "${p}"
    done
done

"${python_executable}" "${lightx2v_path}/scripts/disagg/summarize_moe_resolution_sp.py"
echo "Done. Summary: ${study_dir}/phase1_moe_i2v_resolution_sp_scaling.md"
