#!/bin/bash
# Compare Ulysses dense comm vs Ring SLA sparse comm (MoE 480p, SP P=4).

set -euo pipefail

ROOT=/root/zht/LightX2V
STUDY="${ROOT}/save_results/optimization_study"
mkdir -p "${STUDY}"

source /root/install/miniconda3/etc/profile.d/conda.sh
conda activate lightx2v
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE="${STUDY}/baseline_moe_i2v_480_sla_triton_seqp4.json"
MODEL=${BASELINE_MODEL_PATH:-/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models}
CACHE="${STUDY}/phase1_encoder_inputs.pt"
P=4
GPUS=$(seq -s, 0 $((P - 1)))

gen_cfg() {
  local out="$1"
  local attn_type="$2"
  local sparse="$3"
  local sparse_py="True"
  [[ "${sparse}" == "false" ]] && sparse_py="False"
  python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${BASE}").read_text())
data["parallel"] = {
    "seq_p_size": ${P},
    "seq_p_attn_type": "${attn_type}",
    "seq_p_sparse_comm": ${sparse_py},
}
Path("${out}").write_text(json.dumps(data, indent=2) + "\\n")
print("${out}")
PY
}

CFG_ULE="${STUDY}/baseline_moe_sla_ulysses_seqp4.json"
CFG_RING_D="${STUDY}/baseline_moe_sla_ring_dense_seqp4.json"
CFG_RING_S="${STUDY}/baseline_moe_sla_ring_sparse_seqp4.json"

gen_cfg "${CFG_ULE}" "ulysses" "false"
gen_cfg "${CFG_RING_D}" "ring_sla" "false"
gen_cfg "${CFG_RING_S}" "ring_sla" "true"

run_bench() {
  local tag="$1"
  local cfg="$2"
  local out="${STUDY}/${tag}.json"
  echo "=== ${tag} ==="
  CUDA_VISIBLE_DEVICES="${GPUS}" python -m torch.distributed.run --standalone --nproc_per_node="${P}" \
    "${ROOT}/scripts/disagg/run_phase1_transformer_bench.py" \
    --config_json "${cfg}" \
    --model_path "${MODEL}" \
    --seq_p_size "${P}" \
    --inputs_cache "${CACHE}" \
    --warmup 1 \
    --measure_iters 1 \
    --output_json "${out}" 2>&1 | tail -8
}

run_bench "p1_moe_sla_ulysses_seqp4" "${CFG_ULE}"
run_bench "p1_moe_sla_ring_dense_seqp4" "${CFG_RING_D}"
run_bench "p1_moe_sla_ring_sparse_seqp4" "${CFG_RING_S}"

python - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
rows = []
for tag, label in [
    ("p1_moe_sla_ulysses_seqp4", "ulysses+dense_a2a"),
    ("p1_moe_sla_ring_dense_seqp4", "ring_sla+dense_kv"),
    ("p1_moe_sla_ring_sparse_seqp4", "ring_sla+sparse_kv"),
]:
    p = study / f"{tag}.json"
    if not p.is_file():
        rows.append((label, None))
        continue
    d = json.loads(p.read_text())
    rows.append((label, d.get("transformer_compute_s")))

t1_path = study / "p1_transformer_seqp1.json"
t1 = json.loads(t1_path.read_text()).get("transformer_compute_s") if t1_path.is_file() else None
sla_t1 = study / "p1_moe_i2v_480_sla_triton_seqp1.json"
if sla_t1.is_file():
    t1 = json.loads(sla_t1.read_text()).get("transformer_compute_s")

lines = [
    "# Ring SLA Sparse Comm vs Ulysses (MoE 480×832, SP P=4, sla 0.8)",
    "",
    "| mode | transformer_s | speedup vs SLA P=1 | scaling eff vs P=1 |",
    "|---|---:|---:|---:|",
]
for label, t in rows:
    if t is None:
        lines.append(f"| {label} | — | — | — |")
        continue
    sp = (t1 / t) if t1 else None
    eff = (sp / 4) if sp else None
    lines.append(f"| {label} | {t:.2f} | {sp:.2f}× | {eff*100:.1f}% |" if sp else f"| {label} | {t:.2f} | — | — |")

if len(rows) >= 3 and all(r[1] for r in rows):
    uly, ring_d, ring_s = [r[1] for r in rows]
    lines.extend([
        "",
        f"- ring sparse vs ulysses dense: **{uly/ring_s:.2f}×** wall-time",
        f"- ring sparse vs ring dense kv: **{ring_d/ring_s:.2f}×**",
        f"- comm saved (ring sparse vs ulysses): **{(1 - ring_s/uly)*100:.1f}%** transformer time",
    ])

out = study / "p1_moe_sla_ring_sparse_summary.md"
out.write_text("\n".join(lines) + "\n")
print(out)
for r in rows:
    print(r)
PY
