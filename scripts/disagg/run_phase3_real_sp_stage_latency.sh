#!/bin/bash
# Real dist_infer stage latency at SP P=1,2,4 (safe GPUs only).
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
        fi
    fi
fi

python_executable=${BASELINE_PYTHON_EXECUTABLE:-python}
script="${lightx2v_path}/scripts/disagg/run_phase3_real_sp_stage_latency.py"
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_baseline.json}
model_path=${BASELINE_MODEL_PATH:-${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models}
vae_parallel=${VAE_PARALLEL:-1}

out_p1="${study_dir}/p3_real_sp_stage_latency_seqp1.json"
out_p2="${study_dir}/p3_real_sp_stage_latency_seqp2.json"
out_p4="${study_dir}/p3_real_sp_stage_latency_seqp4.json"
summary_md="${study_dir}/p3_real_sp_stage_latency.md"
log="${study_dir}/p3_real_sp_stage_latency.log"

echo "=== Real SP P=1 GPU0 vae_parallel=${vae_parallel} ===" | tee "${log}"
CUDA_VISIBLE_DEVICES=0 \
  "${python_executable}" "${script}" \
  --config_json "${config_json}" \
  --model_path "${model_path}" \
  --seq_p_size 1 \
  --vae_parallel "${vae_parallel}" \
  --output_json "${out_p1}" \
  2>&1 | tee -a "${log}"

echo "=== Real SP P=2 GPUs 0,2 ===" | tee -a "${log}"
CUDA_VISIBLE_DEVICES=0,2 \
  "${python_executable}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "${script}" \
  --config_json "${config_json}" \
  --model_path "${model_path}" \
  --seq_p_size 2 \
  --vae_parallel "${vae_parallel}" \
  --baseline_json "${out_p1}" \
  --output_json "${out_p2}" \
  2>&1 | tee -a "${log}"

echo "=== Real SP P=4 GPUs 0,2,4,5 ===" | tee -a "${log}"
CUDA_VISIBLE_DEVICES=0,2,4,5 \
  "${python_executable}" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "${script}" \
  --config_json "${config_json}" \
  --model_path "${model_path}" \
  --seq_p_size 4 \
  --vae_parallel "${vae_parallel}" \
  --baseline_json "${out_p1}" \
  --output_json "${out_p4}" \
  2>&1 | tee -a "${log}"

"${python_executable}" - <<PY
import json
from pathlib import Path
study = Path("${study_dir}")
ps = {}
for p in (1, 2, 4):
    ps[p] = json.loads((study / f"p3_real_sp_stage_latency_seqp{p}.json").read_text())

def row(p):
    s = ps[p]["stages_s"]
    return (
        f"| {p} | {s['text_encoder']:.2f} | {s['vae_encoder']:.2f} | {s['encoder_total']:.2f} | "
        f"{s['denoise']:.2f} | {s['decoder']:.2f} | {s['e2e_sum']:.2f} |"
    )

def eta_row(name, key):
    cells = ["1.000"]
    for p in (2, 4):
        v = ps[p].get("parallel_efficiency_vs_p1", {}).get(key)
        cells.append(f"**{v:.3f}**" if v is not None else "—")
    return f"| {name} | " + " | ".join(cells) + " |"

lines = [
    "# Real dist_infer SP stage latency (P=1/2/4)",
    "",
    "Behavior: **all ranks run T5**; **VAE \`vae_parallel=True\`**; **DiT Ulysses**. "
    "Latency = max(rank elapsed) after barrier+cuda sync.",
    f"GPUs: P1=\`0\`, P2=\`0,2\`, P4=\`0,2,4,5\`. vae_parallel={ps[2].get('vae_parallel')}.",
    "",
    "## Wall time (s)",
    "",
    "| SP | T5 | VAE enc | Encoder Σ | DiT | VAE dec | E2E |",
    "|---:|---:|---:|---:|---:|---:|---:|",
    row(1), row(2), row(4),
    "",
    "## Parallel efficiency η = T₁/(T_P·P)",
    "",
    "| Stage | P=1 | P=2 η | P=4 η |",
    "|---|---:|---:|---:|",
    eta_row("T5", "text_encoder"),
    eta_row("VAE enc", "vae_encoder"),
    eta_row("Encoder Σ", "encoder_total"),
    eta_row("DiT", "denoise"),
    eta_row("VAE dec", "decoder"),
    eta_row("E2E", "e2e_sum"),
    "",
    "Interpretation: T5 η≈1/P means redundant full compute (wall≈const). "
    "VAE η depends on vae_parallel scaling. DiT should be ≫1/P.",
    "",
]
(study / "p3_real_sp_stage_latency.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "summary: ${summary_md}"
