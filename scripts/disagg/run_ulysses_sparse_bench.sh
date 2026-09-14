#!/bin/bash
# Ulysses dense vs sparse K/V comm L1/L2 (MoE 480p SLA P=4).
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

gen_cfg() {
  local out="$1" sparse_py="$2" mode="$3" attn="$4"
  python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${BASE}").read_text())
data["parallel"] = {
    "seq_p_size": ${P},
    "seq_p_attn_type": "${attn}",
    "seq_p_sparse_kv_comm": ${sparse_py},
    "seq_p_sparse_kv_mode": ${mode},
}
Path("${out}").write_text(json.dumps(data, indent=2) + "\\n")
PY
}

CFG_D="$STUDY/baseline_moe_sla_ulysses_dense_seqp4.json"
CFG_O="$STUDY/baseline_moe_sla_ulysses_sparse_orig_seqp4.json"
CFG_L1="$STUDY/baseline_moe_sla_ulysses_sparse_l1_seqp4.json"
CFG_L2="$STUDY/baseline_moe_sla_ulysses_sparse_l2_seqp4.json"
gen_cfg "$CFG_D" "False" "0" "ulysses"
gen_cfg "$CFG_O" "True" "0" "ulysses_sparse"
gen_cfg "$CFG_L1" "True" "1" "ulysses_sparse"
gen_cfg "$CFG_L2" "True" "2" "ulysses_sparse_l2"

run() {
  local tag="$1" cfg="$2"
  echo "=== $tag ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/scripts/disagg/run_phase1_transformer_bench.py" \
    --config_json "$cfg" --model_path "$MODEL" --seq_p_size "$P" \
    --inputs_cache "$CACHE" --warmup 1 --measure_iters 1 \
    --output_json "$STUDY/${tag}.json" 2>&1 | tail -5
}

run p1_moe_sla_ulysses_dense_seqp4 "$CFG_D"
run p1_moe_sla_ulysses_sparse_orig_seqp4 "$CFG_O"
run p1_moe_sla_ulysses_sparse_l1_seqp4 "$CFG_L1"
run p1_moe_sla_ulysses_sparse_l2_seqp4 "$CFG_L2"

python - <<'PY'
import json
from pathlib import Path
study = Path("/root/zht/LightX2V/save_results/optimization_study")
t1 = json.loads((study/"p1_moe_i2v_480_sla_triton_seqp1.json").read_text())["transformer_compute_s"]
rows = []
for tag, label in [
    ("p1_moe_sla_ulysses_dense_seqp4", "ulysses_dense"),
    ("p1_moe_sla_ulysses_sparse_orig_seqp4", "sparse_orig"),
    ("p1_moe_sla_ulysses_sparse_l1_seqp4", "sparse_L1"),
    ("p1_moe_sla_ulysses_sparse_l2_seqp4", "sparse_L2"),
]:
    p = study / f"{tag}.json"
    if not p.exists():
        continue
    t = json.loads(p.read_text())["transformer_compute_s"]
    rows.append((label, t))
dense = rows[0][1]
lines = [
    "# Ulysses Sparse K/V Comm L1/L2 (MoE 480p SLA P=4)",
    "",
    f"SLA P=1: {t1:.2f}s",
    "",
    "| mode | transformer_s | vs dense | speedup vs P=1 | eff |",
    "|---|---:|---:|---:|---:|",
]
for label, t in rows:
    sp = t1 / t
    vs = t / dense
    lines.append(f"| {label} | {t:.2f} | {vs:.3f}x | {sp:.2f}x | {sp/4*100:.1f}% |")
if len(rows) >= 2:
    lines += ["", "## vs dense ulysses"]
    for label, t in rows[1:]:
        pct = (1 - t / dense) * 100
        lines.append(f"- **{label}**: {pct:+.1f}% wall-time ({'faster' if pct > 0 else 'slower'})")
(study / "p1_moe_sla_ulysses_sparse_summary.md").write_text("\n".join(lines) + "\n")
print(rows)
PY
