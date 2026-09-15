#!/bin/bash
# Sweep Ulysses SP for wan2.1_sf transformer denoise (P=1,2,3,4,6).
# Wan2.1-T2V-1.3B has num_heads=12 → valid seq_p in {1,2,3,4,6,12}; P=8 is invalid.

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

python_executable=${BASELINE_PYTHON_EXECUTABLE:-/root/install/miniconda3/envs/lightx2v/bin/python}
bench="${lightx2v_path}/scripts/disagg/run_sf_transformer_sp_bench.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/self_forcing/wan_t2v_sf_sp_bench.json}
inputs_cache="${study_dir}/sf_phase1_encoder_inputs.pt"
force=${FORCE_RERUN:-0}
seq_p_list=${SEQ_P_LIST:-1,2,3,4,6}
warmup=${WARMUP:-1}
measure_iters=${MEASURE_ITERS:-1}

run_bench() {
    local p="$1"
    local out="${study_dir}/sf_transformer_seqp${p}.json"
    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if "${python_executable}" -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip SF seq_p=${p} (exists) ==="
            return 0
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== SF transformer bench seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ==="

    local -a cmd=(
        "${python_executable}" "${bench}"
        --model_path "${model_path}"
        --config_json "${config_json}"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --output_json "${out}"
        --warmup "${warmup}"
        --measure_iters "${measure_iters}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${bench}"
            --model_path "${model_path}"
            --config_json "${config_json}"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --output_json "${out}"
            --warmup "${warmup}"
            --measure_iters "${measure_iters}"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/sf_transformer_seqp${p}.log"
}

IFS=',' read -r -a p_values <<< "${seq_p_list}"
for p in "${p_values[@]}"; do
    p="$(echo "${p}" | tr -d ' ')"
    [[ -z "${p}" ]] && continue
    if [[ "${p}" -eq 8 ]]; then
        echo "=== skip P=8: num_heads=12 not divisible by 8 ==="
        continue
    fi
    run_bench "${p}"
done

"${python_executable}" - <<PY
import json
from pathlib import Path

study = Path("${study_dir}")
p_values = [int(x.strip()) for x in "${seq_p_list}".split(",") if x.strip()]
rows = []
base_t = None
for p in p_values:
    path = study / f"sf_transformer_seqp{p}.json"
    if not path.is_file():
        rows.append({"seq_p": p, "transformer_s": None})
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    t = d.get("transformer_compute_s")
    if p == 1:
        base_t = t
    speedup = (base_t / t) if (base_t and t) else None
    eff = (speedup / p) if speedup else None
    rows.append({
        "seq_p": p,
        "transformer_s": t,
        "speedup_vs_p1": round(speedup, 4) if speedup else None,
        "scaling_efficiency": round(eff, 4) if eff else None,
        "num_chunks": d.get("num_chunks"),
        "infer_steps_per_chunk": d.get("infer_steps_per_chunk"),
    })

summary = {
    "model_cls": "wan2.1_sf",
    "metric": "sf_transformer_compute_s",
    "baseline_seq_p": 1,
    "baseline_transformer_s": base_t,
    "results": rows,
}
summary_path = study / "sf_transformer_sp_scaling.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    "# SF (wan2.1_sf) Transformer Ulysses SP Scaling",
    "",
    "Metric: full AR denoise (all chunks × 4 steps + rerun), excludes load/encoder/VAE.",
    "Model: Wan2.1-T2V-1.3B + Self-Forcing checkpoint, cpu_offload=false.",
    "Valid seq_p: divisors of num_heads=12 -> 1,2,3,4,6,12 (not 8).",
    "",
    "| seq_p | transformer_s | speedup vs P=1 | scaling eff | chunks | steps/chunk |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    t = r["transformer_s"]
    sp = r["speedup_vs_p1"]
    ef = r["scaling_efficiency"]
    if t is None:
        lines.append(f"| {r['seq_p']} | — | — | — | — | — |")
    else:
        lines.append(
            f"| {r['seq_p']} | {t:.3f} | {sp:.2f} | {ef * 100:.1f}% | {r.get('num_chunks')} | {r.get('infer_steps_per_chunk')} |"
        )
md_path = study / "sf_transformer_sp_scaling.md"
md_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(summary_path)
print(md_path)
PY

echo "Done. See ${study_dir}/sf_transformer_sp_scaling.md"
