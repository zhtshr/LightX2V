#!/bin/bash
# Phase 1 + 3: Wan2.1-T2V-1.3B Ulysses SP scaling (P=1,2,4,8), cpu_offload=false.

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

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
phase1="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
phase3="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}
inputs_cache="${study_dir}/phase1_t2v_1.3b_encoder_inputs.pt"
config_base="${study_dir}/baseline_t2v_1.3b_seqp1.json"
force=${FORCE_RERUN:-0}

# Baseline configs for each P
for p in 1 2 4 8; do
    cfg="${study_dir}/baseline_t2v_1.3b_seqp${p}.json"
    if [[ ! -f "${cfg}" || "${force}" == "1" ]]; then
        python3 - <<PY
import json
from pathlib import Path
base = Path("${config_base}")
if base.is_file():
    data = json.loads(base.read_text(encoding="utf-8"))
else:
    data = {
        "infer_steps": 4,
        "target_video_length": 81,
        "text_len": 512,
        "target_height": 480,
        "target_width": 832,
        "self_attn_1_type": "sage_attn2",
        "cross_attn_1_type": "sage_attn2",
        "cross_attn_2_type": "sage_attn2",
        "sample_guide_scale": 5,
        "sample_shift": 5,
        "enable_cfg": False,
        "cpu_offload": False,
        "fps": 16,
    }
p = ${p}
data["parallel"] = {"seq_p_size": p, "seq_p_attn_type": "ulysses"}
Path("${cfg}").write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
    fi
done

run_phase1() {
    local p="$1"
    local out="${study_dir}/p1_t2v_1.3b_transformer_seqp${p}.json"
    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip phase1 T2V-1.3B seq_p=${p} (exists) ===" | tee -a "${study_dir}/phase3_t2v_sp_runner.log"
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== phase1 T2V-1.3B seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase3_t2v_sp_runner.log"
    local -a cmd=(
        "${python_executable}" "${phase1}"
        --task t2v
        --model_cls wan2.1
        --model_path "${model_path}"
        --config_json "${study_dir}/baseline_t2v_1.3b_seqp${p}.json"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --output_json "${out}"
        --warmup 1
        --measure_iters 1
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${phase1}"
            --task t2v
            --model_cls wan2.1
            --model_path "${model_path}"
            --config_json "${study_dir}/baseline_t2v_1.3b_seqp${p}.json"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --output_json "${out}"
            --warmup 1
            --measure_iters 1
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/p1_t2v_1.3b_transformer_seqp${p}.log"
}

run_phase3() {
    local p="$1"
    local out="${study_dir}/p3_t2v_1.3b_sp_seqp${p}.json"
    if [[ -f "${out}" && "${force}" != "1" ]]; then
        echo "=== skip phase3 T2V-1.3B seq_p=${p} (exists) ===" | tee -a "${study_dir}/phase3_t2v_sp_runner.log"
        return 0
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== phase3 T2V-1.3B seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase3_t2v_sp_runner.log"
    local -a cmd=(
        "${python_executable}" "${phase3}"
        --seq_p_size "${p}"
        --task t2v
        --model_cls wan2.1
        --model_path "${model_path}"
        --config_json "${study_dir}/baseline_t2v_1.3b_seqp${p}.json"
        --inputs_cache "${inputs_cache}"
        --phase1_json "${study_dir}/p1_t2v_1.3b_transformer_seqp${p}.json"
        --output_json "${out}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${phase3}"
            --seq_p_size "${p}"
            --task t2v
            --model_cls wan2.1
            --model_path "${model_path}"
            --config_json "${study_dir}/baseline_t2v_1.3b_seqp${p}.json"
            --inputs_cache "${inputs_cache}"
            --phase1_json "${study_dir}/p1_t2v_1.3b_transformer_seqp${p}.json"
            --output_json "${out}"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/p3_t2v_1.3b_sp_seqp${p}.log"
}

# Wan2.1-T2V-1.3B has num_heads=12; Ulysses requires num_heads % seq_p == 0 → P=8 unsupported.
MAX_P=${MAX_P:-8}
for p in 1 2 4 8; do
    if [[ "${p}" -gt "${MAX_P}" ]]; then
        continue
    fi
    if [[ "${p}" -eq 8 ]]; then
        if ! python3 -c "import sys; sys.exit(0 if 12 % 8 == 0 else 1)"; then
            echo "=== skip P=8: num_heads=12 not divisible by 8 (Ulysses) ===" | tee -a "${study_dir}/phase3_t2v_sp_runner.log"
            continue
        fi
    fi
    run_phase1 "${p}"
done

for p in 1 2 4 8; do
    if [[ "${p}" -gt "${MAX_P}" ]]; then
        continue
    fi
    if [[ "${p}" -eq 8 ]] && ! python3 -c "import sys; sys.exit(0 if 12 % 8 == 0 else 1)"; then
        continue
    fi
    run_phase3 "${p}"
done

"${python_executable}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
p1_base = None
rows = []
for p in [1, 2, 4, 8]:
    p1 = study / f"p1_t2v_1.3b_transformer_seqp{p}.json"
    p3 = study / f"p3_t2v_1.3b_sp_seqp{p}.json"
    t = None
    if p1.is_file():
        t = json.loads(p1.read_text()).get("transformer_compute_s")
    if p == 1:
        p1_base = t
    speedup = (p1_base / t) if (p1_base and t) else None
    eff = (speedup / p) if speedup else None
    comm_ratio = gpu_avg = None
    if p3.is_file():
        d = json.loads(p3.read_text())
        comm_ratio = d.get("one_step_profile", {}).get("comm_cuda_ratio")
        pooled = d.get("gpu_util_during_denoise", {}).get("pooled_active_gpus", {})
        gpu_avg = pooled.get("avg")
    rows.append({"p": p, "t_s": t, "speedup": speedup, "eff": eff, "comm": comm_ratio, "gpu": gpu_avg})

lines = [
    "# Phase 3 — Wan2.1-T2V-1.3B Ulysses SP Scaling",
    "",
    "Config: `cpu_offload=false`, 4 denoise steps, 480×832×81, `enable_cfg=false`.",
    "Model: `Wan2.1-T2V-1.3B` (`num_heads=12`). **P=8 不可用**：Ulysses 要求 `num_heads % seq_p == 0`。",
    "",
    "| seq_p | transformer_s | speedup vs P=1 | scaling eff | comm CUDA % (1 step) | avg GPU util |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    t, sp, ef, cm, gpu = r["t_s"], r["speedup"], r["eff"], r["comm"], r["gpu"]
    if r["p"] == 8 and 12 % 8 != 0:
        lines.append(f"| 8 | N/A | — | — | — | `num_heads=12` 不能被 8 整除 |")
        continue
    if t is not None and sp is not None and ef is not None:
        cm_s = f"{cm * 100:.1f}%" if cm is not None else "—"
        gpu_s = f"{gpu:.1f}%" if gpu is not None else "—"
        lines.append(f"| {r['p']} | {t:.2f} | {sp:.2f} | {ef * 100:.1f}% | {cm_s} | {gpu_s} |")
    else:
        lines.append("| {p} | — | — | — | — | — |".format(p=r["p"]))

out = study / "phase3_t2v_1.3b_sp_scaling.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY

echo "Done. Summary: ${study_dir}/phase3_t2v_1.3b_sp_scaling.md"
