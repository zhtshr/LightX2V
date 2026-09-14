#!/bin/bash
# Verify Ulysses dir1 (QKV fusion) and dir2 (async comm) for MoE SLA 480p P=4.
set -euo pipefail
ROOT=/root/zht/LightX2V
STUDY="$ROOT/save_results/optimization_study"
source /root/install/miniconda3/etc/profile.d/conda.sh
conda activate lightx2v
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
BASE="$STUDY/baseline_moe_i2v_480_sla_triton_seqp4.json"
MODEL=${BASELINE_MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
CACHE="$STUDY/phase1_encoder_inputs.pt"
P=4
T1=54.429

gen_cfg() {
  local out="$1" fusion="$2" async="$3"
  python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${BASE}").read_text())
data["parallel"] = {
    "seq_p_size": ${P},
    "seq_p_attn_type": "ulysses",
    "seq_p_tensor_fusion": ${fusion},
    "seq_p_async_comm": ${async},
}
Path("${out}").write_text(json.dumps(data, indent=2) + "\\n")
PY
}

CFG_BASE="$STUDY/baseline_moe_sla_fusion_async_base_seqp4.json"
CFG_F1="$STUDY/baseline_moe_sla_fusion_only_seqp4.json"
CFG_F2="$STUDY/baseline_moe_sla_async_only_seqp4.json"
CFG_F12="$STUDY/baseline_moe_sla_fusion_async_seqp4.json"
gen_cfg "$CFG_BASE" "False" "False"
gen_cfg "$CFG_F1" "True" "False"
gen_cfg "$CFG_F2" "False" "True"
gen_cfg "$CFG_F12" "True" "True"

run_bench() {
  local tag="$1" cfg="$2"
  echo "=== bench $tag ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/scripts/disagg/run_phase1_transformer_bench.py" \
    --config_json "$cfg" --model_path "$MODEL" --seq_p_size "$P" \
    --inputs_cache "$CACHE" --warmup 1 --measure_iters 1 \
    --output_json "$STUDY/${tag}.json" 2>&1 | tail -3
}

run_profile() {
  local tag="$1" cfg="$2"
  echo "=== profile $tag ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/scripts/disagg/run_phase3_layer_self_cross_profile.py" \
    --config_json "$cfg" --tag "$tag" \
    --output_json "$STUDY/p3_layer_profile_${tag}.json" 2>&1 | tail -5
}

run_bench p1_moe_sla_ulysses_baseline_seqp4 "$CFG_BASE"
run_bench p1_moe_sla_fusion_only_seqp4 "$CFG_F1"
run_bench p1_moe_sla_async_only_seqp4 "$CFG_F2"
run_bench p1_moe_sla_fusion_async_seqp4 "$CFG_F12"

run_profile moe_sla_baseline "$CFG_BASE"
run_profile moe_sla_fusion_only "$CFG_F1"
run_profile moe_sla_fusion_async "$CFG_F12"

python - <<PY
import json
from pathlib import Path
study = Path("${STUDY}")
t1 = ${T1}
rows = []
for tag, label in [
    ("p1_moe_sla_ulysses_baseline_seqp4", "baseline"),
    ("p1_moe_sla_fusion_only_seqp4", "fusion"),
    ("p1_moe_sla_async_only_seqp4", "async"),
    ("p1_moe_sla_fusion_async_seqp4", "fusion+async"),
]:
    t = json.loads((study / f"{tag}.json").read_text())["transformer_compute_s"]
    eff = t1 / (t * 4) * 100
    rows.append((label, t, eff))

base_t = rows[0][1]
lines = [
    "# Ulysses Fusion / Async Comm Verification (MoE SLA 480p P=4)",
    "",
    f"SLA T₁ = {t1:.2f}s",
    "",
    "| mode | transformer_s | vs baseline | ext eff |",
    "|---|---:|---:|---:|",
]
for label, t, eff in rows:
    vs = (base_t / t - 1) * 100
    lines.append(f"| {label} | {t:.2f} | {vs:+.1f}% | {eff:.1f}% |")

lines += ["", "## Layer profile (self-attn comm ms, mean over 40 layers)"]
lines.append("")
lines.append("| mode | self_attn_ms | comm_ms | comm% | a2a/layer |")
lines.append("|---|---:|---:|---:|---:|")
for tag, label in [
    ("moe_sla_baseline", "baseline"),
    ("moe_sla_fusion_only", "fusion"),
    ("moe_sla_fusion_async", "fusion+async"),
]:
    p = study / f"p3_layer_profile_{tag}.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())["summary"]
    lines.append(
        f"| {label} | {s['self_attn_ms_mean']:.1f} | {s['self_attn_comm_ms_mean']:.1f} | "
        f"{s['comm_pct_of_self_mean']:.1f}% | {s['a2a_calls_per_layer_mean']:.0f} |"
    )

(study / "p1_moe_sla_fusion_async_summary.md").write_text("\\n".join(lines) + "\\n")
print(rows)
PY
