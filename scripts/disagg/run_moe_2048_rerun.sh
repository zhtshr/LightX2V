#!/bin/bash
# MoE 2048²: SP P=2/4/8 + PP×SP sweep (encoder cache must exist).
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
python="${BASELINE_PYTHON_EXECUTABLE:-/root/install/miniconda3/envs/lightx2v/bin/python}"
bench="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
model_path="${MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}"
image_path="${IMAGE_PATH:-/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg}"
cache="${study_dir}/phase1_moe_i2v_2048x2048_encoder_inputs.pt"
log="${study_dir}/phase1_moe_2048_rerun.log"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export LIGHTX2V_NCCL_TIMEOUT_S="${LIGHTX2V_NCCL_TIMEOUT_S:-10800}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

run_sp() {
    local p="$1"
    local tag="p1_moe_i2v_2048x2048_seqp${p}"
    echo "=== ${tag} $(date -Is) ===" | tee -a "${log}"
    if ! "${python}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${bench}" \
        --task i2v \
        --model_cls wan2.2_moe \
        --model_path "${model_path}" \
        --config_json "${study_dir}/baseline_moe_i2v_2048x2048_seqp${p}.json" \
        --image_path "${image_path}" \
        --inputs_cache "${cache}" \
        --output_json "${study_dir}/${tag}.json" \
        --warmup 1 \
        --measure_iters 1 \
        --seq_p_size "${p}" \
        >> "${study_dir}/${tag}_rerun.log" 2>&1; then
        echo "=== FAILED ${tag} $(date -Is) ===" | tee -a "${log}"
        "${python}" - <<PY
import json
from pathlib import Path
out = Path("${study_dir}/${tag}.json")
out.write_text(json.dumps({"resolution": 2048, "seq_p_size": ${p}, "error": "run_failed"}, indent=2) + "\\n")
PY
    fi
}

for p in 2 4 8; do
    run_sp "${p}"
done

echo "=== PP×SP sweep 2048 $(date -Is) ===" | tee -a "${log}"
RESOLUTIONS=2048 FORCE_RERUN=1 bash "${lightx2v_path}/scripts/disagg/run_pp_sp_resolution_sweep.sh" >> "${log}" 2>&1
echo "=== ALL DONE $(date -Is) ===" | tee -a "${log}"
