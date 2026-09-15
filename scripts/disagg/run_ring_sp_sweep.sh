#!/bin/bash
# Ring SP scaling: phase1 transformer compute + phase3 comm/gpu profile for P=1,2,4,8.

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
bench="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
analysis="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
config_base="${study_dir}/baseline_seqp1.json"
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
model_path=${BASELINE_MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
force=${FORCE_RERUN:-0}

# Ring baseline configs (seq_p_attn_type=ring).
for p in 1 2 4 8; do
    cfg="${study_dir}/baseline_ring_seqp${p}.json"
    if [[ ! -f "${cfg}" || "${force}" == "1" ]]; then
        "${python_executable}" - <<PY
import json
from pathlib import Path
src = Path("${config_base}")
dst = Path("${cfg}")
data = json.loads(src.read_text(encoding="utf-8"))
if ${p} > 1:
    data["parallel"] = {"seq_p_size": ${p}, "seq_p_attn_type": "ring"}
else:
    data["parallel"] = False
dst.write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
print(dst)
PY
    fi
done

run_phase1() {
    local p="$1"
    local phase="$2"
    local cfg="${study_dir}/baseline_ring_seqp${p}.json"
    local tag="p1_ring_transformer_seqp${p}"
    local out_json="${study_dir}/${tag}.json"
    if [[ "${phase}" == "warmup" ]]; then
        tag="${tag}_warmup"
        out_json="${study_dir}/${tag}.json"
    fi

    if [[ "${force}" != "1" && -f "${out_json}" ]]; then
        if [[ "${phase}" == "measure" ]]; then
            if "${python_executable}" -c "import json,sys; d=json.load(open('${out_json}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
                echo "=== skip ring phase1 ${phase} p=${p} (exists) ===" | tee -a "${study_dir}/ring_sp_runner.log"
                return 0
            fi
        else
            echo "=== skip ring phase1 ${phase} p=${p} (exists) ===" | tee -a "${study_dir}/ring_sp_runner.log"
            return 0
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    local -a cmd=(
        "${python_executable}" "${bench}"
        --config_json "${cfg}"
        --model_path "${model_path}"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --output_json "${out_json}"
    )
    if [[ "${phase}" == "warmup" ]]; then
        cmd+=(--warmup 1 --measure_iters 0)
    else
        cmd+=(--warmup 0 --measure_iters 1)
    fi
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${bench}"
            --config_json "${cfg}"
            --model_path "${model_path}"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --output_json "${out_json}"
        )
        if [[ "${phase}" == "warmup" ]]; then
            cmd+=(--warmup 1 --measure_iters 0)
        else
            cmd+=(--warmup 0 --measure_iters 1)
        fi
    fi

    echo "=== ring phase1 ${phase} seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/ring_sp_runner.log"
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/${tag}.log"
}

run_phase3() {
    local p="$1"
    local cfg="${study_dir}/baseline_ring_seqp${p}.json"
    local tag="p3_ring_sp_seqp${p}"
    local out_json="${study_dir}/${tag}.json"
    local p1_json="${study_dir}/p1_ring_transformer_seqp${p}.json"

    if [[ "${force}" != "1" && -f "${out_json}" ]]; then
        echo "=== skip ring phase3 p=${p} (exists) ===" | tee -a "${study_dir}/ring_sp_runner.log"
        return 0
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    local -a cmd=(
        "${python_executable}" "${analysis}"
        --seq_p_size "${p}"
        --config_json "${cfg}"
        --output_json "${out_json}"
        --phase1_json "${p1_json}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${analysis}"
            --seq_p_size "${p}"
            --config_json "${cfg}"
            --output_json "${out_json}"
            --phase1_json "${p1_json}"
        )
    fi

    echo "=== ring phase3 seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/ring_sp_runner.log"
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/${tag}.log"
}

for p in 1 2 4 8; do
    run_phase1 "${p}" warmup
done
for p in 1 2 4 8; do
    run_phase1 "${p}" measure
done
for p in 1 2 4 8; do
    run_phase3 "${p}"
done

"${python_executable}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
p1_base = None
rows = []
for p in [1, 2, 4, 8]:
    p1 = study / f"p1_ring_transformer_seqp{p}.json"
    p3 = study / f"p3_ring_sp_seqp{p}.json"
    t = None
    if p1.is_file():
        t = json.loads(p1.read_text()).get("transformer_compute_s")
    if p == 1:
        p1_base = t
    speedup = (p1_base / t) if (p1_base and t) else None
    eff = (speedup / p) if speedup else None
    comm_ratio = None
    gpu_avg = None
    if p3.is_file():
        d = json.loads(p3.read_text())
        comm_ratio = d.get("one_step_profile", {}).get("comm_cuda_ratio")
        pooled = d.get("gpu_util_during_denoise", {}).get("pooled_active_gpus", {})
        gpu_avg = pooled.get("avg")
    rows.append({"p": p, "t_s": t, "speedup": speedup, "efficiency": eff, "comm_cuda_ratio": comm_ratio, "gpu_util_avg": gpu_avg})

# Compare with ulysses (if available)
ulysses = {}
for p in [1, 2, 4, 8]:
    path = study / f"p1_transformer_seqp{p}.json"
    if path.is_file():
        ulysses[p] = json.loads(path.read_text()).get("transformer_compute_s")

lines = [
    "# Ring SP Scaling (Phase 1 + 3)",
    "",
    "Config: `seq_p_attn_type: ring`, Wan2.2-MoE distill 4-step, `cpu_offload=block`.",
    "",
    "| seq_p | transformer_s | speedup vs P=1 | scaling eff | comm CUDA % | GPU util |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    if r["t_s"] is not None and r["speedup"] is not None and r["efficiency"] is not None:
        comm_s = f"{r['comm_cuda_ratio'] * 100:.1f}%" if r["comm_cuda_ratio"] is not None else "—"
        gpu_s = f"{r['gpu_util_avg']:.1f}%" if r["gpu_util_avg"] is not None else "—"
        lines.append(
            f"| {r['p']} | {r['t_s']:.2f} | {r['speedup']:.2f} | {r['efficiency'] * 100:.1f}% | {comm_s} | {gpu_s} |"
        )
    else:
        lines.append(f"| {r['p']} | — | — | — | — | — |")

if ulysses:
    lines.extend([
        "",
        "## vs Ulysses (transformer_s)",
        "",
        "| seq_p | ring_s | ulysses_s | ring/ulysses |",
        "|---:|---:|---:|---:|",
    ])
    for p in [1, 2, 4, 8]:
        ring_t = next((r["t_s"] for r in rows if r["p"] == p), None)
        uly_t = ulysses.get(p)
        if ring_t and uly_t:
            lines.append(f"| {p} | {ring_t:.2f} | {uly_t:.2f} | {uly_t / ring_t:.2f}× |")
        else:
            lines.append(f"| {p} | — | — | — |")

lines.extend([
    "",
    "Run: `FORCE_RERUN=1 bash scripts/disagg/run_ring_sp_sweep.sh`",
])

out = study / "phase3_ring_scaling.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY
