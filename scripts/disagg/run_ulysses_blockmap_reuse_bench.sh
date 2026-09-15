#!/bin/bash
# Verify dir3: reuse SLA block-map (skip full get_block_map in sla_attn).
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
  local out="$1" reuse="$2"
  python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${BASE}").read_text())
data["parallel"] = {
    "seq_p_size": ${P},
    "seq_p_attn_type": "ulysses",
    "seq_p_reuse_sla_block_map": ${reuse},
}
Path("${out}").write_text(json.dumps(data, indent=2) + "\\n")
PY
}

CFG_BASE="$STUDY/baseline_moe_sla_blockmap_base_seqp4.json"
CFG_D3="$STUDY/baseline_moe_sla_blockmap_reuse_seqp4.json"
gen_cfg "$CFG_BASE" "False"
gen_cfg "$CFG_D3" "True"

run_bench() {
  local tag="$1" cfg="$2"
  echo "=== $tag ==="
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
    --output_json "$STUDY/p3_layer_profile_${tag}.json" 2>&1 | tail -3
}

run_bench p1_moe_sla_blockmap_baseline_seqp4 "$CFG_BASE"
run_bench p1_moe_sla_blockmap_reuse_seqp4 "$CFG_D3"
run_profile moe_sla_blockmap_baseline "$CFG_BASE"
run_profile moe_sla_blockmap_reuse "$CFG_D3"

python - <<PY
import json
from pathlib import Path
study = Path("${STUDY}")
t1 = ${T1}
rows = []
for tag, label in [
    ("p1_moe_sla_blockmap_baseline_seqp4", "baseline"),
    ("p1_moe_sla_blockmap_reuse_seqp4", "dir3_reuse_map"),
]:
    t = json.loads((study / f"{tag}.json").read_text())["transformer_compute_s"]
    eff = t1 / (t * 4) * 100
    rows.append((label, t, eff))
base = rows[0][1]
lines = [
    "# SLA Block-Map Reuse (Dir3) — MoE SLA 480p P=4",
    "",
    f"SLA T₁ = {t1:.2f}s",
    "",
    "| mode | transformer_s | vs baseline | ext eff |",
    "|---|---:|---:|---:|",
]
for label, t, eff in rows:
    pct = (base / t - 1) * 100
    lines.append(f"| {label} | {t:.2f} | {pct:+.1f}% | {eff:.1f}% |")
lines += ["", "## Layer profile"]
lines.append("")
lines.append("| mode | self_attn_ms | compute_ms | comm_ms |")
lines.append("|---|---:|---:|---:|")
for tag, label in [("moe_sla_blockmap_baseline", "baseline"), ("moe_sla_blockmap_reuse", "dir3")]:
    p = study / f"p3_layer_profile_{tag}.json"
    s = json.loads(p.read_text())["summary"]
    lines.append(
        f"| {label} | {s['self_attn_ms_mean']:.1f} | {s['self_attn_compute_ms_mean']:.1f} | {s['self_attn_comm_ms_mean']:.1f} |"
    )
(study / "p1_moe_sla_blockmap_reuse_summary.md").write_text("\\n".join(lines) + "\\n")
print(rows)
PY
