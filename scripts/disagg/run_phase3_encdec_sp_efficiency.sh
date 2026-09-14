#!/bin/bash
# Exp-3b: monolithic SP encoder/decoder parallel efficiency + GPU util.
# Safe GPUs only: P=1 -> 0; P=4 -> 0,2,4,5 (never 1 or 3).

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/root/zht/LightX2V}
study_dir=${STUDY_DIR:-${lightx2v_path}/save_results/optimization_study}
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

baseline_conda_env=${BASELINE_CONDA_ENV:-lightx2v}
if [[ "${BASELINE_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${baseline_conda_env}" ]]; then
        if command -v conda >/dev/null 2>&1; then
            set +u
            eval "$(conda shell.bash hook)"
            conda activate "${baseline_conda_env}"
            set -u
            echo "activated conda env: ${baseline_conda_env}"
        fi
    fi
fi

python_executable=${BASELINE_PYTHON_EXECUTABLE:-python}
script="${lightx2v_path}/scripts/disagg/run_phase3_encdec_sp_efficiency.py"
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_baseline.json}
model_path=${BASELINE_MODEL_PATH:-${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models}

out_p1="${study_dir}/p3_encdec_sp_efficiency_seqp1.json"
out_p4="${study_dir}/p3_encdec_sp_efficiency_seqp4.json"
summary_md="${study_dir}/p3_encdec_sp_efficiency.md"
log="${study_dir}/p3_encdec_sp_efficiency.log"

echo "=== Exp-3b P=1 on GPU 0 ===" | tee "${log}"
if [[ -f "${out_p1}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "skip P=1 (exists ${out_p1})" | tee -a "${log}"
else
    CUDA_VISIBLE_DEVICES=0 \
      "${python_executable}" "${script}" \
      --config_json "${config_json}" \
      --model_path "${model_path}" \
      --seq_p_size 1 \
      --output_json "${out_p1}" \
      2>&1 | tee -a "${log}"
fi

echo "=== Exp-3b P=4 on GPUs 0,2,4,5 ===" | tee -a "${log}"
CUDA_VISIBLE_DEVICES=0,2,4,5 \
  "${python_executable}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${script}" \
  --config_json "${config_json}" \
  --model_path "${model_path}" \
  --seq_p_size 4 \
  --baseline_json "${out_p1}" \
  --output_json "${out_p4}" \
  2>&1 | tee -a "${log}"

"${python_executable}" - <<PY
import json
from pathlib import Path

study = Path("${study_dir}")
p1 = json.loads((study / "p3_encdec_sp_efficiency_seqp1.json").read_text())
p4 = json.loads((study / "p3_encdec_sp_efficiency_seqp4.json").read_text())

def row(tag, d):
    s = d["stages_s"]
    return f"| {tag} | {s['encoder']:.2f} | {s['denoise']:.2f} | {s['decoder']:.2f} | {s['e2e_sum']:.2f} |"

def util_line(d, stage):
    st = d.get("gpu_util_by_stage", {}).get(stage, {})
    return (
        f"mean={st.get('mean_util_all_gpus_pct', 0):.1f}% "
        f"busiest={st.get('busiest_gpu_mean_util_pct', 0):.1f}% "
        f"idle_lt20={st.get('idle_gpu_count_mean_lt20', '?')} "
        f"balance={st.get('util_balance_ratio', 0):.2f}"
    )

eff = p4.get("parallel_efficiency_vs_p1", {})
lines = [
    "# Exp-3b: Monolithic SP encoder/decoder parallel efficiency",
    "",
    "Pattern: enc/dec on rank0 only + barrier; denoise Ulysses SP.",
    "",
    "## Stage wall time (s)",
    "",
    "| Config | encoder | denoise | decoder | e2e sum |",
    "|---|---:|---:|---:|---:|",
    row("SP P=1 (GPU0)", p1),
    row("SP P=4 (0,2,4,5)", p4),
    "",
    "## Parallel efficiency η = T₁/(T_P·P)",
    "",
    f"- encoder η = **{(eff.get('encoder') or 0):.3f}** (ideal if rank0-only ≈ 0.250)",
    f"- decoder η = **{(eff.get('decoder') or 0):.3f}** (ideal if rank0-only ≈ 0.250)",
    f"- denoise η = **{(eff.get('denoise') or 0):.3f}**",
    f"- e2e η = **{(eff.get('e2e_sum') or 0):.3f}**",
    "",
    "## GPU util during stages (P=4)",
    "",
    f"- encoder: {util_line(p4, 'encoder')}",
    f"- denoise: {util_line(p4, 'denoise')}",
    f"- decoder: {util_line(p4, 'decoder')}",
    "",
    "## Takeaway",
    "",
    "Encoder/decoder wall time barely improves with SP while ~P-1 GPUs stay idle "
    "(low util_balance during enc/dec, high during denoise). "
    "This is why stage disaggregation (dedicated enc/dec cards + SP only on denoise) "
    "beats binding all GPUs into one monolithic SP group for the full E2E pipeline.",
    "",
]
(study / "p3_encdec_sp_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "summary: ${summary_md}"
