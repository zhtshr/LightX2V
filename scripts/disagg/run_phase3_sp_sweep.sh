#!/bin/bash
# Phase 3: SP scaling efficiency, comm/compute profile, GPU util during denoise.

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
analysis="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"

run_p() {
    local p="$1"
    local cfg="${study_dir}/baseline_seqp${p}.json"
    local tag="p3_sp_seqp${p}"
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"

    echo "=== phase3 seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase3_runner.log"

    local -a cmd=(
        "${python_executable}" "${analysis}"
        --seq_p_size "${p}"
        --config_json "${cfg}"
        --output_json "${study_dir}/${tag}.json"
        --phase1_json "${study_dir}/p1_transformer_seqp${p}.json"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${analysis}"
            --seq_p_size "${p}"
            --config_json "${cfg}"
            --output_json "${study_dir}/${tag}.json"
            --phase1_json "${study_dir}/p1_transformer_seqp${p}.json"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/${tag}.log"
}

for p in 1 2 4 8; do
    if [[ -f "${study_dir}/p3_sp_seqp${p}.json" ]]; then
        echo "=== skip phase3 seq_p=${p} (exists) ===" | tee -a "${study_dir}/phase3_runner.log"
        continue
    fi
    run_p "${p}"
done

"${python_executable}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
p1_base = None
rows = []
for p in [1, 2, 4, 8]:
    p1 = study / f"p1_transformer_seqp{p}.json"
    p3 = study / f"p3_sp_seqp{p}.json"
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

lines = [
    "# Phase 3 — Seq Parallel Scaling & GPU Util (Denoise)",
    "",
    "## Q1: Why not ~8× at P=8?",
    "",
    "Strong scaling efficiency = speedup / P. Comm share from one profiled denoise step (CUDA, NCCL/all_to_all).",
    "",
    "| seq_p | transformer_s | speedup vs P=1 | scaling eff | comm CUDA % (1 step) | avg GPU util (denoise) |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    t_s = r["t_s"]
    speedup = r["speedup"]
    eff = r["efficiency"]
    comm = r["comm_cuda_ratio"]
    gpu = r["gpu_util_avg"]
    if t_s is not None and speedup is not None and eff is not None:
        comm_s = f"{comm * 100:.1f}%" if comm is not None else "—"
        gpu_s = f"{gpu:.1f}%" if gpu is not None else "—"
        lines.append(
            f"| {r['p']} | {t_s:.2f} | {speedup:.2f} | {eff * 100:.1f}% | {comm_s} | {gpu_s} |"
        )
    else:
        lines.append(f"| {r['p']} | — | — | — | — | — |")

lines.extend([
    "",
    "## Q2: Batch headroom?",
    "",
    "- **GPU util during active denoise** (not E2E): if avg util << 100% with high mem use → kernel gaps / comm wait → limited batch gain on same SP group.",
    "- **Phase 0 disagg E2E util ~12%** mixes queue wait; use this table for compute-bound assessment.",
    "- **Phase 2**: enc/dec per-sample flat → multi-request batch does not help E2E without VAE batching.",
    "",
    "## Interpretation guide",
    "",
    "- Rising **comm CUDA %** with P explains sub-linear speedup (Ulysses all_to_all per layer).",
    "- If **efficiency** drops while **GPU util** stays high → comm/sync bound, not idle compute.",
    "- If **GPU util** is low during denoise → room for overlap / fusion, not naive multi-batch on same ranks.",
])

out = study / "phase3_sp_scaling.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY

echo "=== phase3 dual-overlap hypothesis (optional) ===" | tee -a "${study_dir}/phase3_runner.log"
echo "Run: bash ${lightx2v_path}/scripts/disagg/run_phase3_dual_overlap.sh" | tee -a "${study_dir}/phase3_runner.log"
