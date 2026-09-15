#!/bin/bash
# Phase 1: Wan2.2-Distill-Models tensor-parallel degree sweep (probe).
# Note: Wan2.2 MoE weights/infer do not implement true TP sharding yet.

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
bench_script="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
config_base="${study_dir}/baseline_seqp1.json"
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
model_path=${BASELINE_MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}

run_tp() {
    local tp="$1"
    local tag="p1_wan22_distill_tensorp${tp}"
    local out_json="${study_dir}/${tag}.json"
    local cfg="${study_dir}/baseline_tensorp${tp}.json"

    if [[ ! -f "${cfg}" ]]; then
        python3 - <<PY
import json
from pathlib import Path
src = Path("${config_base}")
dst = Path("${cfg}")
data = json.loads(src.read_text(encoding="utf-8"))
data["parallel"] = {"tensor_p_size": ${tp}}
dst.write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
    fi

    if [[ -f "${out_json}" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "=== skip tensor_p=${tp} (exists) ===" | tee -a "${study_dir}/phase1_tp_runner.log"
        return 0
    fi

    local -a cmd=(
        "${python_executable}" "${bench_script}"
        --config_json "${cfg}"
        --model_path "${model_path}"
        --tensor_p_size "${tp}"
        --seq_p_size 1
        --inputs_cache "${inputs_cache}"
        --output_json "${out_json}"
        --warmup 1
        --measure_iters 1
    )

    if [[ "${tp}" -gt 1 ]]; then
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((tp - 1)))"
        cmd=(
            "${python_executable}" -m torch.distributed.run
            --standalone
            "--nproc_per_node=${tp}"
            "${bench_script}"
            --config_json "${cfg}"
            --model_path "${model_path}"
            --tensor_p_size "${tp}"
            --seq_p_size 1
            --inputs_cache "${inputs_cache}"
            --output_json "${out_json}"
            --warmup 1
            --measure_iters 1
        )
    else
        export CUDA_VISIBLE_DEVICES=0
    fi

    echo "=== phase1 tensor_p=${tp} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_tp_runner.log"
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/${tag}.log"
}

for tp in 1 2 4 8; do
    run_tp "${tp}"
done

"${python_executable}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
rows = []
for tp in [1, 2, 4, 8]:
    path = study / f"p1_wan22_distill_tensorp{tp}.json"
    row = {"tensor_p": tp, "ok": False}
    if path.is_file():
        d = json.loads(path.read_text())
        t = d.get("transformer_compute_s")
        row.update({"ok": t is not None, "transformer_compute_s": t, "world_size": d.get("world_size")})
    rows.append(row)

base = next((r["transformer_compute_s"] for r in rows if r["tensor_p"] == 1 and r.get("transformer_compute_s")), None)

# Load SP results for comparison
sp_rows = []
for p in [1, 2, 4, 8]:
    sp_path = study / f"p1_transformer_seqp{p}.json"
    if sp_path.is_file():
        d = json.loads(sp_path.read_text())
        sp_rows.append({"seq_p": p, "t": d.get("transformer_compute_s")})

lines = [
    "# Wan2.2-Distill-Models — Tensor Parallel Sweep (probe)",
    "",
    "Model: `Wan2.2-Distill-Models` / `wan2.2_moe` | `cpu_offload=block` | 4 denoise steps | 480×832×81",
    "",
    "> **Important**: Wan2.2 MoE **does not implement weight-sharded tensor parallel** in",
    "> `transformer_weights` / `infer`. `tensor_p_size>1` only builds a `tensor_p` device mesh;",
    "> each rank still loads the **full** model. Numbers below measure that behavior.",
    "",
    "## Tensor parallel (config only — no real sharding)",
    "",
    "| tensor_p | world | transformer_s | speedup vs TP=1 | scaling eff |",
    "|---:|---:|---:|---:|---:|",
]
for row in rows:
    t = row.get("transformer_compute_s")
    sp_up = f"{base / t:.2f}×" if base and t else "—"
    eff = f"{100 * (base / t) / row['tensor_p']:.1f}%" if base and t and row["tensor_p"] else "—"
    lines.append(
        f"| {row['tensor_p']} | {row.get('world_size', '—')} | "
        f"{t if t else '—'} | {sp_up} | {eff} |"
    )

if sp_rows:
    sp_base = next((r["t"] for r in sp_rows if r["seq_p"] == 1), None)
    lines.extend([
        "",
        "## Sequence parallel (Ulysses) — actually supported",
        "",
        "| seq_p | transformer_s | speedup vs P=1 | scaling eff |",
        "|---:|---:|---:|---:|",
    ])
    for r in sp_rows:
        t = r["t"]
        sp_up = f"{sp_base / t:.2f}×" if sp_base and t else "—"
        eff = f"{100 * (sp_base / t) / r['seq_p']:.1f}%" if sp_base and t else "—"
        lines.append(f"| {r['seq_p']} | {t} | {sp_up} | {eff} |")

lines.extend([
    "",
    "## Conclusion",
    "",
    "- **TP**: no real speedup expected until Wan wires `MMWeightTP` into Wan weights/infer.",
    "- **SP (Ulysses)**: use `seq_p_size` sweep (`run_phase1_baseline_sweep.sh`) for multi-GPU scaling.",
    "",
    "Run: `FORCE_RERUN=1 bash scripts/disagg/run_phase1_tp_sweep.sh`",
])

out = study / "phase1_wan22_tp_scaling.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY
