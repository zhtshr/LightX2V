#!/bin/bash
# 256² T2V-1.3B SLA P=1: batch=1 vs batch=2 (fused + serial)
set -euo pipefail

ROOT=/root/zht/LightX2V
STUDY="${ROOT}/save_results/optimization_study"
mkdir -p "${STUDY}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lightx2v" ]]; then
  set +u
  eval "$(conda shell.bash hook)"
  conda activate lightx2v
  set -u
fi

DENSE_CFG="${STUDY}/baseline_t2v_1.3b_256x256_seqp1.json"
SLA_CFG="${STUDY}/baseline_t2v_1.3b_256x256_sla_seqp1.json"
CACHE="${STUDY}/phase1_t2v_1.3b_256x256_encoder_inputs.pt"
BENCH="${ROOT}/scripts/disagg/run_phase1_transformer_batch_bench.py"
MODEL="${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}"

python3 - <<PY
import json
from pathlib import Path
src = Path("${DENSE_CFG}")
dst = Path("${SLA_CFG}")
data = json.loads(src.read_text(encoding="utf-8"))
data["self_attn_1_type"] = "sla_attn"
data["sla_attn_setting"] = {"sparsity_ratio": 0.8, "operator": "triton"}
data["parallel"] = {"seq_p_size": 1, "seq_p_attn_type": "ulysses"}
dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print("wrote", dst)
PY

# Build encoder cache if missing
if [[ ! -f "${CACHE}" ]]; then
  python "${ROOT}/scripts/disagg/run_phase1_transformer_bench.py" \
    --task t2v --model_cls wan2.1 --model_path "${MODEL}" \
    --config_json "${SLA_CFG}" --seq_p_size 1 \
    --inputs_cache "${CACHE}" --encoder_only
fi

for mode in fused serial; do
  out="${STUDY}/p1_t2v_1.3b_256_sla_batch2_${mode}.json"
  echo "=== SLA batch bench mode=${mode} ===" | tee -a "${STUDY}/p1_256_sla_batch_runner.log"
  python "${BENCH}" \
    --task t2v --model_cls wan2.1 --model_path "${MODEL}" \
    --config_json "${SLA_CFG}" --seq_p_size 1 \
    --inputs_cache "${CACHE}" \
    --batch_size 2 --batch_mode "${mode}" \
    --warmup 1 --measure_iters 3 \
    --output_json "${out}" \
    2>&1 | tee -a "${STUDY}/p1_256_sla_batch_${mode}.log"
done

# Dense sage baseline (same fused path, non-SLA attn)
DENSE_CFG="${STUDY}/baseline_t2v_1.3b_256x256_seqp1.json"
out="${STUDY}/p1_t2v_1.3b_256_dense_batch2_fused.json"
echo "=== Dense sage batch2 fused ===" | tee -a "${STUDY}/p1_256_sla_batch_runner.log"
python "${BENCH}" \
  --task t2v --model_cls wan2.1 --model_path "${MODEL}" \
  --config_json "${DENSE_CFG}" --seq_p_size 1 \
  --inputs_cache "${CACHE}" \
  --batch_size 2 --batch_mode fused \
  --warmup 1 --measure_iters 3 \
  --output_json "${out}" \
  2>&1 | tee -a "${STUDY}/p1_256_dense_batch_fused.log"

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
fused = json.loads((study / "p1_t2v_1.3b_256_sla_batch2_fused.json").read_text())
serial = json.loads((study / "p1_t2v_1.3b_256_sla_batch2_serial.json").read_text())
dense = json.loads((study / "p1_t2v_1.3b_256_dense_batch2_fused.json").read_text())
b1 = fused["cases"]["batch1"]["transformer_compute_s"]
lines = [
    "# 256² batch=2 throughput (T2V-1.3B, P=1, 4 steps) — fixed fused path",
    "",
    "Fused path: batched patch/text embed, batched QKV/FFN GEMMs, SLA/sage attn on `[B,H,L,D]`.",
    "",
    f"- batch=1 (SLA): **{b1:.3f} s** → **{1/b1:.3f} samples/s**",
]
for tag, data, label in [
    ("fused", fused, "SLA fused"),
    ("serial", serial, "SLA serial×2"),
    ("fused", dense, "Dense fused"),
]:
    key = "batch2_" + tag
    c = data["cases"][key]
    thr = c["throughput_samples_per_s"]
    eff = c["efficiency_vs_ideal"] * 100
    lines.append(
        f"- batch=2 **{label}**: wall **{c['transformer_compute_s']:.3f} s** → "
        f"**{thr:.3f} samples/s** ({c['ms_per_sample']:.1f} ms/sample, "
        f"batch eff vs 2×B1: **{eff:.1f}%**)"
    )
out = study / "p1_t2v_1.3b_256_sla_batch_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.read_text())
PY
