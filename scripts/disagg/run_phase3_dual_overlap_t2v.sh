#!/bin/bash
# Phase 3: dual-tenant comm overlap for Wan2.1-T2V-1.3B (cpu_offload=false, Ulysses SP).

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
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}
config_json="${study_dir}/baseline_t2v_1.3b_seqp${p}.json"
inputs_cache="${study_dir}/phase1_t2v_1.3b_encoder_inputs.pt"
phase3_json="${study_dir}/p3_t2v_1.3b_sp_seqp${p}.json"
tag="p3_dual_overlap_t2v_1.3b_seqp${p}"
out_json="${study_dir}/${tag}.json"
report_md="${study_dir}/phase3_dual_overlap_t2v_1.3b.md"
measure_steps=${MEASURE_STEPS:-0}
align_iters=${ALIGN_ITERS:-3}

export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"

echo "=== phase3 dual overlap T2V-1.3B seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} cpu_offload=false ===" \
    | tee -a "${study_dir}/phase3_dual_overlap_t2v_runner.log"

# Optional: phase3 comm profile for comm_cuda_ratio
if [[ ! -f "${phase3_json}" || "${FORCE_RERUN:-0}" == "1" ]]; then
    sp_bench="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${sp_bench}" \
        --seq_p_size "${p}" \
        --task t2v \
        --model_cls wan2.1 \
        --model_path "${model_path}" \
        --config_json "${config_json}" \
        --inputs_cache "${inputs_cache}" \
        --output_json "${phase3_json}" \
        2>&1 | tee -a "${study_dir}/p3_t2v_1.3b_sp_seqp${p}.log" || true
fi

if [[ -f "${out_json}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "=== skip ${tag} (exists, FORCE_RERUN=1 to rerun) ===" | tee -a "${study_dir}/phase3_dual_overlap_t2v_runner.log"
else
    bench_args=(
        --seq_p_size "${p}"
        --task t2v
        --model_cls wan2.1
        --model_path "${model_path}"
        --config_json "${config_json}"
        --inputs_cache "${inputs_cache}"
        --phase3_json "${phase3_json}"
        --output_json "${out_json}"
    )
    if [[ "${measure_steps}" != "0" ]]; then
        bench_args+=(--measure_steps "${measure_steps}")
    fi
    bench_args+=(--align_iters "${align_iters}")
    if [[ "${p}" -gt 1 ]]; then
        "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
            "${bench}" "${bench_args[@]}" \
            2>&1 | tee -a "${study_dir}/${tag}.log"
    else
        "${python_executable}" "${bench}" "${bench_args[@]}" \
            2>&1 | tee -a "${study_dir}/${tag}.log"
    fi
fi

"${python_executable}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
import os
p = os.environ.get("SEQ_P_SIZE", "4")
path = study / f"p3_dual_overlap_t2v_1.3b_seqp{p}.json"
report = study / "phase3_dual_overlap_t2v_1.3b.md"
if not path.is_file():
    raise SystemExit(0)

d = json.loads(path.read_text())
tp = d.get("throughput", {})
sp = d.get("speedup", {})
ops = d.get("one_step_pair_overlap_stats", {})
aas = d.get("a2a_overlap_stats", {})

def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"

lines = [
    "# Phase 3 — Dual-Tenant Overlap: Wan2.1-T2V-1.3B (fair)",
    "",
    f"Model: `{d.get('model_path', 'Wan2.1-T2V-1.3B')}` | task=`{d.get('task', 't2v')}` | "
    f"model_cls=`{d.get('model_cls', 'wan2.1')}` | Ulysses SP={d.get('seq_p_size', 4)}",
    f"Config: `cpu_offload={d.get('cpu_offload', False)}`, `enable_cfg={d.get('enable_cfg', False)}`",
    f"measure_steps={d.get('measure_steps', 'all')}",
    "",
    "**Fair baselines**: `enable_cfg=false`; single-path means are from **interleaved** alignment runs.",
    "",
    "## Single-path alignment (interleaved means)",
    "",
]
align = d.get("single_path_alignment") or {}
stats = align.get("stats") or {}
for key, label in (
    ("model_infer", "single `model.infer()`"),
    ("decomposed_plain", "single decomposed (no a2a patch)"),
    ("decomposed_patch", "single decomposed (+ a2a patch, dual path)"),
):
    st = stats.get(key) or {}
    lines.append(
        f"- **{label}**: mean {f(st.get('mean_s'))}s "
        f"(min {f(st.get('min_s'))}, max {f(st.get('max_s'))}, n={st.get('n', '—')})",
    )
if align.get("align_iters"):
    lines.append(f"- align_iters={align.get('align_iters')}, rotation={align.get('rotation', '—')}")
lines.extend([
    "",
    "## Per-request cost (1 req wall, aligned means)",
    "",
    "| mode | wall (1 req) |",
    "|---|---:|",
    f"| single `model.infer()` | {f(d.get('single_transformer_s'))}s |",
    f"| single decomposed (no patch) | {f(d.get('single_decomposed_plain_s'))}s |",
    f"| single decomposed (+ patch) | {f(d.get('single_decomposed_s'))}s |",
    "",
    "## Throughput (2 req wall)",
    "",
    "| mode | wall (2 req) | throughput |",
    "|---|---:|---:|",
    f"| dual back-to-back (`model.infer`) | {f(d.get('dual_back_to_back_s'))}s | {f(tp.get('dual_back_to_back_rps'))} |",
    f"| dual **decomposed** back-to-back (fair) | {f(d.get('dual_decomposed_back_to_back_s'))}s | {f(tp.get('dual_decomposed_back_to_back_rps'))} |",
    f"| dual **a2a-serial** | {f(d.get('dual_a2a_serial_s'))}s | {f(tp.get('dual_a2a_serial_rps'))} |",
    f"| dual **a2a-overlap** | {f(d.get('dual_a2a_overlap_s'))}s | {f(tp.get('dual_a2a_overlap_rps'))} |",
    f"| dual layer-serial | {f(d.get('dual_layer_serial_s'))}s | {f(tp.get('dual_layer_serial_rps'))} |",
    f"| dual layer-overlap | {f(d.get('dual_layer_overlap_s'))}s | {f(tp.get('dual_layer_overlap_rps'))} |",
    "",
    "## Speedup — **all_to_all window overlap** (primary, fair)",
    "",
    f"- a2a-overlap vs **a2a-serial**: **{f(sp.get('a2a_overlap_vs_a2a_serial'), 2)}×**",
    f"- a2a-overlap vs **fair dual baseline** (2× single decomposed): **{f(sp.get('a2a_overlap_vs_fair_dual_baseline'), 2)}×**",
    f"- a2a-serial vs fair dual baseline: **{f(sp.get('a2a_serial_vs_fair_dual_baseline'), 2)}×**",
    f"- one-step a2a overlap vs serial: **{f(sp.get('one_step_a2a_overlap_vs_serial'), 2)}×**",
    f"- comm windows: a2a {aas.get('a2a_calls', '—')} ({aas.get('a2a_overlap_windows', '—')} overlapped), "
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
])

if d.get("error"):
    lines.extend(["", f"**error**: `{d['error']}`"])

lines.extend([
    "",
    "Run: `FORCE_RERUN=1 bash scripts/disagg/run_phase3_dual_overlap_t2v.sh`",
])

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
PY
