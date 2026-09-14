#!/usr/bin/env bash
# 256x256 Wan2.2 I2V: TP=1/2 latency + memory sweep
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY="${ROOT}/save_results/optimization_study"
mkdir -p "${STUDY}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/root/install/miniconda3/envs/lightx2v/bin/python}"
BENCH="${ROOT}/scripts/disagg/run_phase1_transformer_bench.py"
MEM="${ROOT}/scripts/disagg/run_tp2_memory_profile.py"

CFG_TP1="${ROOT}/configs/disagg/baseline/wan22_moe_i2v_256_tp1_offload.json"
CFG_TP2="${ROOT}/configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json"
INPUTS="${STUDY}/phase1_encoder_inputs_256x256.pt"
WARMUP="${WARMUP:-1}"
MEASURE="${MEASURE:-3}"

pkill -f "torch.distributed.run.*run_phase" 2>/dev/null || true
sleep 2

# Pre-build encoder inputs once (single process) to avoid NCCL hang on refresh
if [[ ! -f "${INPUTS}" ]]; then
  echo "=== Preparing 256x256 inputs cache ==="
  export CUDA_VISIBLE_DEVICES=0
  "${PYTHON}" "${BENCH}" \
    --config_json "${CFG_TP1}" \
    --model_path "${ROOT}/models/lightx2v/Wan2.2-Distill-Models" \
    --task i2v --model_cls wan2.2_moe \
    --tensor_p_size 1 --seq_p_size 1 \
    --inputs_cache "${INPUTS}" --refresh_inputs_cache \
    --output_json "${STUDY}/wan22_256_inputs_build.json" \
    --warmup 0 --measure_iters 0 2>&1 | tail -5 || true
fi

run_latency() {
  local tp="$1"
  local cfg="$2"
  local out="${STUDY}/wan22_256_tp${tp}_latency.json"
  local log="${STUDY}/wan22_256_tp${tp}_latency.log"
  echo "=== Latency TP=${tp} 256x256 ===" | tee "${log}"

  local -a cmd=(
    "${PYTHON}" "${BENCH}"
    --config_json "${cfg}"
    --model_path "${ROOT}/models/lightx2v/Wan2.2-Distill-Models"
    --task i2v --model_cls wan2.2_moe
    --tensor_p_size "${tp}" --seq_p_size 1
    --inputs_cache "${INPUTS}"
    --output_json "${out}"
    --warmup "${WARMUP}" --measure_iters "${MEASURE}"
  )

  if [[ "${tp}" -gt 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
    cmd=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${tp}"
      "${BENCH}" --config_json "${cfg}"
      --model_path "${ROOT}/models/lightx2v/Wan2.2-Distill-Models"
      --task i2v --model_cls wan2.2_moe
      --tensor_p_size "${tp}" --seq_p_size 1
      --inputs_cache "${INPUTS}"
      --output_json "${out}"
      --warmup "${WARMUP}" --measure_iters "${MEASURE}")
  else
    export CUDA_VISIBLE_DEVICES=0
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log}"
}

run_memory() {
  local tp="$1"
  local cfg="$2"
  local out="${STUDY}/wan22_256_tp${tp}_memory.json"
  local log="${STUDY}/wan22_256_tp${tp}_memory.log"
  echo "=== Memory TP=${tp} 256x256 ===" | tee "${log}"

  if [[ "${tp}" -gt 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${tp}" \
      "${MEM}" --tensor_p_size "${tp}" --config_json "${cfg}" \
      --inputs_cache "${INPUTS}" --output_json "${out}" \
      --warmup 1 2>&1 | tee -a "${log}"
  else
    export CUDA_VISIBLE_DEVICES=0
    "${PYTHON}" "${MEM}" --tensor_p_size 1 --config_json "${cfg}" \
      --inputs_cache "${INPUTS}" --output_json "${out}" \
      --warmup 1 2>&1 | tee -a "${log}"
  fi
}

run_latency 1 "${CFG_TP1}"
run_latency 2 "${CFG_TP2}"
run_memory 1 "${CFG_TP1}"
run_memory 2 "${CFG_TP2}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
rows = []
for tp in (1, 2):
    lat = json.loads((study / f"wan22_256_tp{tp}_latency.json").read_text())
    mem = json.loads((study / f"wan22_256_tp{tp}_memory.json").read_text())
    samples = lat.get("transformer_compute_samples_s") or []
    rows.append({
        "tp": tp,
        "latency_s": lat["transformer_compute_s"],
        "samples": samples,
        "peak_gb": mem["summary"]["single_request_peak_gb"],
        "load_gb": mem["summary"]["weights_and_static_gb"],
        "headroom_gb": mem["summary"]["single_request_headroom_gb"],
        "act_delta_gb": mem["summary"]["activation_one_step_delta_gb"],
        "cpu_offload": "block" if tp == 1 else "false",
    })

base = rows[0]["latency_s"]
lines = [
    "# Wan2.2-Distill 256×256 — TP=1 vs TP=2 延迟 & 显存",
    "",
    "Resolution: **256×256**, 81 frames, 4 denoise steps, int8-q8f",
    "TP=1: `cpu_offload=block` | TP=2: real TP, `cpu_offload=false`, `unload_modules=false`",
    "",
    "| TP | transformer_s | speedup vs TP=1 | peak VRAM (GiB) | load (GiB) | act Δ (GiB) | headroom (GiB) | samples (s) |",
    "|---:|---:|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    sp = f"{base / r['latency_s']:.3f}×" if base else "—"
    samp = ", ".join(f"{s:.3f}" for s in r["samples"]) or "—"
    lines.append(
        f"| {r['tp']} | {r['latency_s']:.3f} | {sp} | {r['peak_gb']:.2f} | "
        f"{r['load_gb']:.2f} | {r['act_delta_gb']:.2f} | {r['headroom_gb']:.2f} | {samp} |"
    )
if len(rows) == 2:
    w = rows[1]["latency_s"]
    overlap_pessim = rows[1]["load_gb"] + 2 * rows[1]["act_delta_gb"]
    lines.extend([
        "",
        "## 双请求 overlap 粗估（TP=2）",
        "",
        f"- 悲观重叠峰值 ≈ load + 2×act = **{overlap_pessim:.2f} GiB**",
        f"- 卡容量 22.18 GiB → 余量 **{22.18 - overlap_pessim:.2f} GiB**",
        "",
        f"JSON: `wan22_256_tp{{1,2}}_latency.json`, `wan22_256_tp{{1,2}}_memory.json`",
    ])
out = study / "wan22_256_tp12_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY

echo "Done: ${STUDY}/wan22_256_tp12_summary.md"
