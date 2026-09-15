#!/bin/bash
# Wan2.1-T2V-1.3B: Ulysses SP scaling across resolutions.
# Resolutions: 256², 512², 1024², 2048² | seq_p: 1,2,3,4,6 (num_heads=12)

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
bench="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}
force=${FORCE_RERUN:-0}
resolutions=${RESOLUTIONS:-256,512,1024,2048}
seq_ps=${SEQ_PS:-1,2,3,4,6}

write_config() {
    local res="$1"
    local p="$2"
    local cfg="${study_dir}/baseline_t2v_1.3b_${res}x${res}_seqp${p}.json"
    python3 - <<PY
import json
from pathlib import Path
data = {
    "infer_steps": 4,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": ${res},
    "target_width": ${res},
    "self_attn_1_type": "sage_attn2",
    "cross_attn_1_type": "sage_attn2",
    "cross_attn_2_type": "sage_attn2",
    "sample_guide_scale": 5,
    "sample_shift": 5,
    "enable_cfg": False,
    "cpu_offload": False,
    "fps": 16,
    "parallel": {"seq_p_size": ${p}, "seq_p_attn_type": "ulysses"},
}
Path("${cfg}").write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
}

run_case() {
    local res="$1"
    local p="$2"
    local tag="p1_t2v_1.3b_${res}x${res}_seqp${p}"
    local out="${study_dir}/${tag}.json"
    local cfg="${study_dir}/baseline_t2v_1.3b_${res}x${res}_seqp${p}.json"
    local cache="${study_dir}/phase1_t2v_1.3b_${res}x${res}_encoder_inputs.pt"

    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip ${tag} (exists) ===" | tee -a "${study_dir}/phase1_t2v_res_sp_runner.log"
            return 0
        fi
    fi

    write_config "${res}" "${p}"
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== ${tag} res=${res} seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_t2v_res_sp_runner.log"

    local -a cmd=(
        "${python_executable}" "${bench}"
        --task t2v
        --model_cls wan2.1
        --model_path "${model_path}"
        --config_json "${cfg}"
        --seq_p_size "${p}"
        --inputs_cache "${cache}"
        --output_json "${out}"
        --warmup 1
        --measure_iters 1
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${bench}"
            --task t2v
            --model_cls wan2.1
            --model_path "${model_path}"
            --config_json "${cfg}"
            --seq_p_size "${p}"
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
        echo "=== FAILED ${tag} ===" | tee -a "${study_dir}/phase1_t2v_res_sp_runner.log"
        return 0
    fi
}

IFS=',' read -ra RES_ARR <<< "${resolutions}"
IFS=',' read -ra P_ARR <<< "${seq_ps}"

for res in "${RES_ARR[@]}"; do
    for p in "${P_ARR[@]}"; do
        if ! python3 -c "import sys; sys.exit(0 if 12 % ${p} == 0 else 1)"; then
            echo "=== skip res=${res} P=${p}: num_heads=12 not divisible ===" | tee -a "${study_dir}/phase1_t2v_res_sp_runner.log"
            continue
        fi
        run_case "${res}" "${p}"
    done
done

"${python_executable}" "${lightx2v_path}/scripts/disagg/summarize_t2v_resolution_sp.py"
