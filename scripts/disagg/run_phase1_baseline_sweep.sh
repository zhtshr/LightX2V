#!/bin/bash
# Phase 1: validate seq parallel speedup on transformer denoise compute only.

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Avoid stale torch extension locks from interrupted runs.
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
bench_script="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
config_base="${study_dir}/baseline_seqp1.json"
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
model_path=${BASELINE_MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}

run_case() {
    local p="$1"
    local phase="$2" # warmup|measure

    local cfg="${study_dir}/baseline_seqp${p}.json"
    local tag="p1_transformer_seqp${p}"
    local out_json="${study_dir}/${tag}.json"
    if [[ "${phase}" == "warmup" ]]; then
        tag="${tag}_warmup"
        out_json="${study_dir}/${tag}.json"
        if [[ -f "${out_json}" ]]; then
            echo "=== skip phase1 ${phase} seq_p=${p} (exists: ${out_json}) ===" | tee -a "${study_dir}/phase1_runner.log"
            return 0
        fi
    else
        if [[ -f "${out_json}" ]]; then
            if python3 -c "import json,sys; d=json.load(open('${out_json}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
                echo "=== skip phase1 ${phase} seq_p=${p} (exists: ${out_json}) ===" | tee -a "${study_dir}/phase1_runner.log"
                return 0
            fi
        fi
    fi

    local gpus
    gpus=$(seq -s, 0 $((p - 1)))
    export CUDA_VISIBLE_DEVICES="${gpus}"

    local -a cmd=(
        "${python_executable}"
        "${bench_script}"
        --config_json "${cfg}"
        --model_path "${model_path}"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --output_json "${study_dir}/${tag}.json"
    )

    if [[ "${phase}" == "warmup" ]]; then
        cmd+=(--warmup 1 --measure_iters 0)
    else
        cmd+=(--warmup 0 --measure_iters 1)
    fi

    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run
            --standalone
            "--nproc_per_node=${p}"
            "${bench_script}"
            --config_json "${cfg}"
            --model_path "${model_path}"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --output_json "${study_dir}/${tag}.json"
        )
        if [[ "${phase}" == "warmup" ]]; then
            cmd+=(--warmup 1 --measure_iters 0)
        else
            cmd+=(--warmup 0 --measure_iters 1)
        fi
    fi

    echo "=== phase1 ${phase} transformer seq_p=${p} gpus=${gpus} ===" | tee -a "${study_dir}/phase1_runner.log"
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/${tag}.log"
}

# Ensure baseline configs exist (parallel.seq_p_size matches P).
for p in 1 2 4 8; do
    cfg="${study_dir}/baseline_seqp${p}.json"
    if [[ ! -f "${cfg}" ]]; then
        python3 - <<PY
import json
from pathlib import Path
src = Path("${config_base}")
dst = Path("${cfg}")
data = json.loads(src.read_text(encoding="utf-8"))
data["parallel"] = {"seq_p_size": ${p}, "seq_p_attn_type": "ulysses"}
dst.write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
    fi
done

for p in 1 2 4 8; do
    run_case "${p}" warmup
done

for p in 1 2 4 8; do
    run_case "${p}" measure
done

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
rows = []
for p in [1, 2, 4, 8]:
    path = study / f"p1_transformer_seqp{p}.json"
    row = {"seq_p": p, "ok": False}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        t = data.get("transformer_compute_s")
        row.update(
            {
                "ok": t is not None,
                "transformer_compute_s": t,
                "model_load_s_excluded": data.get("model_load_s_excluded"),
                "samples_s": data.get("transformer_compute_samples_s"),
            }
        )
    rows.append(row)

base = next((r["transformer_compute_s"] for r in rows if r["seq_p"] == 1 and r.get("transformer_compute_s")), None)
lines = [
    "# Phase 1 — Transformer Denoise Compute (seq parallel)",
    "",
    "Metric: `transformer_compute_s` = scheduler.prepare + denoise loop only.",
    "Excluded: model load, encoder, decoder, controller/E2E latency.",
    "",
    "| seq_p | ok | transformer_compute_s | speedup_vs_p1 | model_load_s (excluded) |",
    "|---:|:---:|---:|---:|---:|",
]
for row in rows:
    speedup = ""
    if base and row.get("transformer_compute_s"):
        speedup = f"{base / row['transformer_compute_s']:.3f}"
    lines.append(
        f"| {row['seq_p']} | {'Y' if row['ok'] else 'N'} | "
        f"{row.get('transformer_compute_s', '')} | {speedup} | {row.get('model_load_s_excluded', '')} |"
    )

out = study / "phase1_transformer_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY
