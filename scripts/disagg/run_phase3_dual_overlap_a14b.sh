#!/bin/bash
# Phase 3: dual-tenant comm overlap for Wan2.2-I2V-A14B (cpu_offload=false, Ulysses SP).

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
p1_bench="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
sp_bench="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
p=${SEQ_P_SIZE:-4}
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
config_json="${study_dir}/baseline_a14b_no_offload_seqp${p}.json"
inputs_cache="${study_dir}/phase1_a14b_encoder_inputs.pt"
phase3_json="${study_dir}/p3_a14b_no_offload_sp_seqp${p}.json"
p1_json="${study_dir}/p1_a14b_no_offload_transformer_seqp1.json"
tag="p3_dual_overlap_a14b_no_offload_seqp${p}"
out_json="${study_dir}/${tag}.json"
report_md="${study_dir}/phase3_dual_overlap_a14b_no_offload.md"
measure_steps=${MEASURE_STEPS:-0}
align_iters=${ALIGN_ITERS:-3}

export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"

echo "=== phase3 dual overlap A14B seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} cpu_offload=false ===" \
    | tee -a "${study_dir}/phase3_dual_overlap_a14b_runner.log"

# P=1 baseline for throughput ceiling (4 / T_P1)
if [[ ! -f "${p1_json}" || "${FORCE_RERUN:-0}" == "1" ]]; then
    echo "=== P=1 ceiling baseline ===" | tee -a "${study_dir}/phase3_dual_overlap_a14b_runner.log"
    CUDA_VISIBLE_DEVICES=0 "${python_executable}" "${p1_bench}" \
        --seq_p_size 1 \
        --task i2v \
        --model_cls wan2.2_moe \
        --model_path "${model_path}" \
        --config_json "${study_dir}/baseline_a14b_no_offload_seqp1.json" \
        --inputs_cache "${inputs_cache}" \
        --output_json "${p1_json}" \
        2>&1 | tee -a "${study_dir}/p1_a14b_no_offload_transformer_seqp1.log" || true
fi

if [[ ! -f "${phase3_json}" || "${FORCE_RERUN:-0}" == "1" ]]; then
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${sp_bench}" \
        --seq_p_size "${p}" \
        --task i2v \
        --model_cls wan2.2_moe \
        --model_path "${model_path}" \
        --config_json "${config_json}" \
        --inputs_cache "${inputs_cache}" \
        --output_json "${phase3_json}" \
        2>&1 | tee -a "${study_dir}/p3_a14b_no_offload_sp_seqp${p}.log" || true
fi

if [[ -f "${out_json}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "=== skip ${tag} (exists, FORCE_RERUN=1 to rerun) ===" | tee -a "${study_dir}/phase3_dual_overlap_a14b_runner.log"
else
    bench_args=(
        --seq_p_size "${p}"
        --task i2v
        --model_cls wan2.2_moe
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
path = study / f"p3_dual_overlap_a14b_no_offload_seqp{p}.json"
p1_path = study / "p1_a14b_no_offload_transformer_seqp1.json"
report = study / "phase3_dual_overlap_a14b_no_offload.md"
if not path.is_file():
    raise SystemExit(0)

d = json.loads(path.read_text())
tp = d.get("throughput", {})
sp = d.get("speedup", {})
ops = d.get("one_step_pair_overlap_stats", {})
aas = d.get("a2a_overlap_stats", {})

ceiling_rps = None
t_p1 = None
if p1_path.is_file():
    t_p1 = json.loads(p1_path.read_text()).get("transformer_compute_s")
    if t_p1 and t_p1 > 0:
        ceiling_rps = 4.0 / t_p1

ov_rps = tp.get("dual_a2a_overlap_rps")
pct_ceiling = 100.0 * ov_rps / ceiling_rps if ov_rps and ceiling_rps else None

def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"

lines = [
    "# Phase 3 — Dual-Tenant Overlap: Wan2.2-I2V-A14B (no offload)",
    "",
    f"Model: `{d.get('model_path')}` | task=`i2v` | model_cls=`wan2.2_moe` | Ulysses SP={d.get('seq_p_size', 4)}",
    f"Config: `cpu_offload={d.get('cpu_offload', False)}`, `enable_cfg={d.get('enable_cfg', False)}`",
    f"measure_steps={d.get('measure_steps', 'all')}",
    "",
    "## Throughput ceiling",
    "",
    f"- P=1 single-GPU wall: **{f(t_p1)}s** (no offload)",
    f"- Hardware ceiling `4 / T_P1`: **{f(ceiling_rps)} req/s**",
    f"- dual a2a-overlap: **{f(ov_rps)} req/s** → **{f(pct_ceiling, 1)}%** of ceiling",
    "",
    "## Throughput (2 req wall)",
    "",
    "| mode | wall (2 req) | throughput | vs ceiling |",
    "|---|---:|---:|---:|",
]
for label, key, rkey in (
    ("dual back-to-back", "dual_back_to_back_s", "dual_back_to_back_rps"),
    ("dual decomposed b2b (fair)", "dual_decomposed_back_to_back_s", "dual_decomposed_back_to_back_rps"),
    ("dual a2a-serial", "dual_a2a_serial_s", "dual_a2a_serial_rps"),
    ("dual **a2a-overlap**", "dual_a2a_overlap_s", "dual_a2a_overlap_rps"),
):
    rps = tp.get(rkey)
    pct = f"{100*rps/ceiling_rps:.1f}%" if rps and ceiling_rps else "—"
    lines.append(f"| {label} | {f(d.get(key))}s | {f(rps)} | {pct} |")

lines.extend([
    "",
    "## Speedup — all_to_all window overlap",
    "",
    f"- a2a-overlap vs a2a-serial: **{f(sp.get('a2a_overlap_vs_a2a_serial'), 2)}×**",
    f"- a2a-overlap vs fair dual baseline: **{f(sp.get('a2a_overlap_vs_fair_dual_baseline'), 2)}×**",
    f"- one-step a2a overlap vs serial: **{f(sp.get('one_step_a2a_overlap_vs_serial'), 2)}×**",
    f"- overlap fraction of serial (one-step pairs): **{f(ops.get('overlap_fraction_of_serial'), 2)}**",
    f"- comm windows: a2a {aas.get('a2a_calls', '—')} ({aas.get('a2a_overlap_windows', '—')} overlapped)",
    "",
    "Run: `FORCE_RERUN=1 bash scripts/disagg/run_phase3_dual_overlap_a14b.sh`",
])

if d.get("error"):
    lines.extend(["", f"**error**: `{d['error']}`"])

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
PY
