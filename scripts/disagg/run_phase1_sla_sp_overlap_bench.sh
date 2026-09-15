#!/bin/bash
# SLA dual a2a-overlap: small (T2V-1.3B) + large (MoE I2V) @ 480×832.
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

rm -f "${HOME}/.cache/torch_extensions/py312_cu128/quant_cuda/lock" \
    "${HOME}/.cache/torch_extensions/py312_cu128/quant_cuda/.ninja_lock" 2>/dev/null || true

if [[ "${CONDA_DEFAULT_ENV:-}" != "lightx2v" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate lightx2v
    set -u
fi

python_executable=python
overlap="${lightx2v_path}/scripts/disagg/run_phase3_dual_overlap_bench.py"
sp_bench="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
force=${FORCE_RERUN:-0}
align_iters=${ALIGN_ITERS:-3}

run_moe_overlap() {
    local p="$1"
    local cfg="${study_dir}/baseline_moe_i2v_480_sla_triton_seqp${p}.json"
    local phase3_json="${study_dir}/p3_moe_i2v_480_sla_sp_seqp${p}.json"
    local out="${study_dir}/p3_dual_overlap_moe_i2v_480x832_sla_seqp${p}.json"
    local inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
    local model_path=/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models

    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('dual_a2a_overlap_s') else 1)"; then
            echo "=== skip MoE SLA overlap P=${p} (exists) ==="
            return 0
        fi
    fi

    if [[ ! -f "${phase3_json}" ]]; then
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
        "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
            "${sp_bench}" \
            --seq_p_size "${p}" \
            --task i2v --model_cls wan2.2_moe \
            --model_path "${model_path}" \
            --config_json "${cfg}" \
            --inputs_cache "${inputs_cache}" \
            --output_json "${phase3_json}" \
            2>&1 | tee -a "${study_dir}/p3_moe_i2v_480_sla_sp_seqp${p}.log" || true
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== MoE SLA overlap P=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_sla_overlap_runner.log"
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${overlap}" \
        --seq_p_size "${p}" \
        --task i2v --model_cls wan2.2_moe \
        --model_path "${model_path}" \
        --config_json "${cfg}" \
        --inputs_cache "${inputs_cache}" \
        --phase3_json "${phase3_json}" \
        --align_iters "${align_iters}" \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/p3_dual_overlap_moe_i2v_480x832_sla_seqp${p}.log" \
        || echo "WARN MoE SLA overlap P=${p} failed"
}

for p in 2 4 8; do
    run_moe_overlap "${p}"
done

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")

def load_t(paths):
    for name in paths:
        p = study / name
        if p.is_file():
            v = json.loads(p.read_text()).get("transformer_compute_s")
            if v is not None:
                return v
    return None

def load_overlap(path):
    p = study / path
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("dual_a2a_overlap_s")

def ov_eff(t1, t_ov, p):
    if t1 and t_ov and p:
        return 2 * t1 / (t_ov * p)
    return None

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "—"

def f3(x):
    return f"{x:.3f}" if x is not None else "—"

# --- small model ---
small_dense_t1 = load_t(["p3_t2v_1.3b_sp_seqp1.json", "p1_t2v_1.3b_transformer_seqp1.json"])
small_sla_t1 = load_t(["p1_t2v_1.3b_sla_transformer_seqp1.json"])

small_rows = []
for p in (2, 3, 4, 6):
    dense_ov = load_overlap(f"p3_dual_overlap_t2v_1.3b_seqp{p}.json")
    sla_ov = load_overlap(f"p3_dual_overlap_t2v_1.3b_sla_seqp{p}.json")
    small_rows.append({
        "seq_p": p,
        "dense_overlap_s": dense_ov,
        "sla_overlap_s": sla_ov,
        "dense_overlap_eff": ov_eff(small_dense_t1, dense_ov, p),
        "sla_overlap_eff": ov_eff(small_sla_t1, sla_ov, p),
    })

# --- large model ---
moe_dense_t1 = load_t(["p1_transformer_seqp1.json", "p3_sp_seqp1.json"])
moe_sla_t1 = load_t(["p1_moe_i2v_480_sla_triton_seqp1.json"])

moe_dense_ov_paths = {
    2: "p3_dual_overlap_seqp2.json",
    4: "p3_dual_overlap_seqp4.json",
    8: "p3_dual_overlap_moe_i2v_480x832_seqp8.json",
}
moe_rows = []
for p in (2, 4, 8):
    dense_ov = load_overlap(moe_dense_ov_paths[p])
    sla_ov = load_overlap(f"p3_dual_overlap_moe_i2v_480x832_sla_seqp{p}.json")
    moe_rows.append({
        "seq_p": p,
        "dense_overlap_s": dense_ov,
        "sla_overlap_s": sla_ov,
        "dense_overlap_eff": ov_eff(moe_dense_t1, dense_ov, p),
        "sla_overlap_eff": ov_eff(moe_sla_t1, sla_ov, p),
    })

summary = {
    "small_model": {
        "name": "Wan2.1-T2V-1.3B",
        "resolution": "480x832",
        "dense_t1_s": small_dense_t1,
        "sla_t1_s": small_sla_t1,
        "overlap": small_rows,
    },
    "large_model": {
        "name": "Wan2.2-MoE I2V",
        "resolution": "480x832",
        "config": "block offload, int8-q8f, 4 steps",
        "dense_t1_s": moe_dense_t1,
        "sla_t1_s": moe_sla_t1,
        "overlap": moe_rows,
    },
    "overlap_eff_formula": "2*T1/(T_overlap*P)",
}

out_json = study / "p1_sla_sp_overlap_summary.json"
out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    "# SLA SP dual a2a-overlap efficiency @ 480×832",
    "",
    "Overlap efficiency = `2·T₁ / (T_overlap × P)` (2 req in-flight, P GPUs).",
    "",
    "## 小模型 Wan2.1-T2V-1.3B（无 offload）",
    "",
    f"Dense T₁={f3(small_dense_t1)}s | SLA T₁={f3(small_sla_t1)}s",
    "",
    "| P | dense overlap (s) | SLA overlap (s) | dense eff | SLA eff |",
    "|---:|---:|---:|---:|---:|",
]
for r in small_rows:
    lines.append(
        f"| {r['seq_p']} | {f3(r['dense_overlap_s'])} | {f3(r['sla_overlap_s'])} | "
        f"{pct(r['dense_overlap_eff'])} | {pct(r['sla_overlap_eff'])} |"
    )

lines += [
    "",
    "## 大模型 Wan2.2-MoE I2V（block offload）",
    "",
    f"Dense T₁={f3(moe_dense_t1)}s | SLA T₁={f3(moe_sla_t1)}s",
    "",
    "| P | dense overlap (s) | SLA overlap (s) | dense eff | SLA eff |",
    "|---:|---:|---:|---:|---:|",
]
for r in moe_rows:
    lines.append(
        f"| {r['seq_p']} | {f3(r['dense_overlap_s'])} | {f3(r['sla_overlap_s'])} | "
        f"{pct(r['dense_overlap_eff'])} | {pct(r['sla_overlap_eff'])} |"
    )

md = study / "p1_sla_sp_overlap_summary.md"
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
