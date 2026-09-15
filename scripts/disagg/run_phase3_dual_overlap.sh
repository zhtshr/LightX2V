#!/bin/bash
# Phase 3: layer-level dual-tenant overlap — fair compare vs layer-serial decomposition.

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
bench="${lightx2v_path}/scripts/disagg/run_phase3_dual_overlap_bench.py"
p=${SEQ_P_SIZE:-4}
offload_suffix=""
config_json="${study_dir}/baseline_seqp${p}.json"
if [[ "${NO_CPU_OFFLOAD:-0}" == "1" ]]; then
    offload_suffix="_no_offload"
    config_json="${study_dir}/baseline_seqp${p}_no_offload.json"
fi
tag="p3_dual_overlap_seqp${p}${offload_suffix}"
out_json="${study_dir}/${tag}.json"
report_md="${study_dir}/phase3_dual_overlap${offload_suffix}.md"
phase3_json="${study_dir}/p3_sp_seqp${p}.json"

export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"

echo "=== phase3 dual overlap seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} cpu_offload=$([[ "${NO_CPU_OFFLOAD:-0}" == "1" ]] && echo false || echo true) ===" | tee -a "${study_dir}/phase3_dual_overlap_runner.log"

if [[ -f "${out_json}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "=== skip ${tag} (exists, FORCE_RERUN=1 to rerun) ===" | tee -a "${study_dir}/phase3_dual_overlap_runner.log"
else
    if [[ "${p}" -gt 1 ]]; then
        "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
            "${bench}" --seq_p_size "${p}" --config_json "${config_json}" --phase3_json "${phase3_json}" --output_json "${out_json}" \
            2>&1 | tee -a "${study_dir}/${tag}.log"
    else
        "${python_executable}" "${bench}" --seq_p_size "${p}" --config_json "${config_json}" --phase3_json "${phase3_json}" --output_json "${out_json}" \
            2>&1 | tee -a "${study_dir}/${tag}.log"
    fi
fi

"${python_executable}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
import os
offload_suffix = "_no_offload" if os.environ.get("NO_CPU_OFFLOAD", "0") == "1" else ""
p = os.environ.get("SEQ_P_SIZE", "4")
path = study / f"p3_dual_overlap_seqp{p}{offload_suffix}.json"
report = study / f"phase3_dual_overlap{offload_suffix}.md"
if not path.is_file():
    raise SystemExit(0)

d = json.loads(path.read_text())
tp = d.get("throughput", {})
sp = d.get("speedup", {})
ops = d.get("one_step_pair_overlap_stats", {})

def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"

aas = d.get("a2a_overlap_stats", {})
ass = d.get("a2a_serial_stats", {})

cpu_offload = d.get("cpu_offload", "—")
offload_gran = d.get("offload_granularity", "—")
unload_mod = d.get("unload_modules", "—")
lines = [
    "# Phase 3 — Dual-Tenant Overlap (measured)",
    "",
    f"Config: `cpu_offload={cpu_offload}`, `offload_granularity={offload_gran}`, `unload_modules={unload_mod}`",
    "(no **block** streaming — disagg-style: weights resident on GPU during active model)",
    "",
    "Fair baselines:",
    "- **dual back-to-back**: two full `model.infer()` runs",
    "- **dual a2a-serial**: same pipeline; every `all_to_all*` synchronous (patch on, overlap off)",
    "- **dual a2a-overlap**: each `all_to_all*` + `all_gather` async; other tenant `cross_ffn` on compute stream during wait",
    "- **dual layer-serial / layer-overlap**: coarser layer-granularity reference",
    "",
    "## Throughput (req/s)",
    "",
    "| mode | wall (2 req) | throughput |",
    "|---|---:|---:|",
    f"| single | {f(d.get('single_transformer_s'))}s | {f(tp.get('single_rps'))} |",
    f"| dual back-to-back | {f(d.get('dual_back_to_back_s'))}s | {f(tp.get('dual_back_to_back_rps'))} |",
    f"| dual **a2a-serial** | {f(d.get('dual_a2a_serial_s'))}s | {f(tp.get('dual_a2a_serial_rps'))} |",
    f"| dual **a2a-overlap** | {f(d.get('dual_a2a_overlap_s'))}s | {f(tp.get('dual_a2a_overlap_rps'))} |",
    f"| dual layer-serial | {f(d.get('dual_layer_serial_s'))}s | {f(tp.get('dual_layer_serial_rps'))} |",
    f"| dual layer-overlap | {f(d.get('dual_layer_overlap_s'))}s | {f(tp.get('dual_layer_overlap_rps'))} |",
    f"| pure-compute ceiling | — | {f(tp.get('pure_compute_ceiling_rps'))} |",
    "",
    "## Speedup — **all_to_all window overlap** (primary)",
    "",
    f"- a2a-overlap vs **a2a-serial**: **{f(sp.get('a2a_overlap_vs_a2a_serial'), 2)}×**",
    f"- a2a-overlap vs back-to-back: **{f(sp.get('a2a_overlap_vs_back_to_back'), 2)}×**",
    f"- one-step a2a overlap vs serial: **{f(sp.get('one_step_a2a_overlap_vs_serial'), 2)}×**",
    f"- comm windows (full run): a2a {aas.get('a2a_calls', '—')} ({aas.get('a2a_overlap_windows', '—')} overlapped), "
    f"all_gather {aas.get('all_gather_calls', '—')} ({aas.get('all_gather_overlap_windows', '—')} overlapped)",
    "",
    "## Speedup — layer-granularity (reference)",
    "",
    f"- layer-overlap vs layer-serial: **{f(sp.get('layer_overlap_vs_layer_serial'), 2)}×**",
    f"- one-step layer overlap vs serial: **{f(sp.get('one_step_overlap_vs_one_step_serial'), 2)}×**",
    "",
    "## One-step layer pair stats",
    "",
    f"- pairs: {ops.get('num_pairs', '—')}",
    f"- serial sum: {f(ops.get('serial_sum_s'))}s | parallel sum: {f(ops.get('parallel_sum_s'))}s",
    f"- saved sum: {f(ops.get('per_pair_saved_sum_s'))}s | overlap fraction of serial: {f(ops.get('overlap_fraction_of_serial'), 2)}",
]

if d.get("error"):
    lines.extend(["", f"**error**: `{d['error']}`"])

lines.extend([
    "",
    "If **a2a-overlap vs a2a-serial ≤ 1.0**, per-collective overlap does not hide NCCL on this stack.",
    "",
    "Run: `NO_CPU_OFFLOAD=1 FORCE_RERUN=1 bash scripts/disagg/run_phase3_dual_overlap.sh`",
])

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
PY
