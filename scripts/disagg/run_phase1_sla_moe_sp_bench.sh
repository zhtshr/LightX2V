#!/bin/bash
# Wan2.2-MoE I2V @ 480×832: SLA triton Ulysses SP scaling (P=1,2,4,8).
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
phase1="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
phase3_sp="${lightx2v_path}/scripts/disagg/run_phase3_sp_analysis.py"
model_path=/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
force=${FORCE_RERUN:-0}
run_comm=${RUN_COMM_PROFILE:-1}

python3 - <<'PY'
import copy
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
dense = json.loads((study / "baseline_seqp1.json").read_text(encoding="utf-8"))

for p in (1, 2, 4, 8):
    cfg = copy.deepcopy(dense)
    cfg["self_attn_1_type"] = "sla_attn"
    cfg["sla_attn_setting"] = {"sparsity_ratio": 0.8, "operator": "triton"}
    cfg.pop("general_sparse_attn_setting", None)
    cfg["parallel"] = {"seq_p_size": p, "seq_p_attn_type": "ulysses"}
    (study / f"baseline_moe_i2v_480_sla_triton_seqp{p}.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )
print("configs ok")
PY

run_sp() {
    local p="$1"
    local cfg="${study_dir}/baseline_moe_i2v_480_sla_triton_seqp${p}.json"
    local out="${study_dir}/p1_moe_i2v_480_sla_triton_seqp${p}.json"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip SLA SP P=${p} (exists) ==="
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== SLA SP P=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_sla_moe_sp_runner.log"
    local -a cmd=(
        "${python_executable}" "${phase1}"
        --config_json "${cfg}"
        --model_path "${model_path}"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --warmup 1 --measure_iters 3
        --output_json "${out}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${phase1}"
            --config_json "${cfg}"
            --model_path "${model_path}"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --warmup 1 --measure_iters 3
            --output_json "${out}"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/p1_moe_i2v_480_sla_triton_seqp${p}.log"
}

run_comm_profile() {
    local p="$1"
    [[ "${run_comm}" == "0" ]] && return 0
    [[ "${p}" -eq 1 ]] && return 0
    local cfg="${study_dir}/baseline_moe_i2v_480_sla_triton_seqp${p}.json"
    local out="${study_dir}/p3_moe_i2v_480_sla_sp_seqp${p}.json"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip comm profile SLA P=${p} (exists) ==="
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== comm profile SLA P=${p} ===" | tee -a "${study_dir}/phase1_sla_moe_sp_runner.log"
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${phase3_sp}" \
        --seq_p_size "${p}" \
        --config_json "${cfg}" \
        --model_path "${model_path}" \
        --inputs_cache "${inputs_cache}" \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/p3_moe_i2v_480_sla_sp_seqp${p}.log" || true
}

for p in 1 2 4 8; do
    run_sp "${p}" || echo "WARN SP P=${p} failed"
done
for p in 2 4 8; do
    run_comm_profile "${p}" || echo "WARN comm P=${p} failed"
done

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")

def load_t(name):
    p = study / name
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("transformer_compute_s")

def load_comm(name):
    p = study / name
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    prof = d.get("one_step_profile") or {}
    return prof.get("comm_cuda_ratio")

dense_t1 = load_t("p1_transformer_seqp1.json") or load_t("p3_sp_seqp1.json")
sla_t1 = load_t("p1_moe_i2v_480_sla_triton_seqp1.json")

rows = []
for p in (1, 2, 4, 8):
    dense = load_t(f"p1_transformer_seqp{p}.json") or load_t(f"p3_sp_seqp{p}.json")
    sla = load_t(f"p1_moe_i2v_480_sla_triton_seqp{p}.json")
    dense_comm = load_comm(f"p3_sp_seqp{p}.json") if p > 1 else 0.0
    sla_comm = load_comm(f"p3_moe_i2v_480_sla_sp_seqp{p}.json") if p > 1 else 0.0

    def eff(t1, t, pp):
        if t1 and t and pp:
            return t1 / (t * pp)
        return None

    rows.append({
        "seq_p": p,
        "dense_s": dense,
        "sla_s": sla,
        "dense_sp_eff": eff(dense_t1, dense, p),
        "sla_sp_eff": eff(sla_t1, sla, p),
        "sla_vs_dense": (dense / sla) if dense and sla else None,
        "dense_comm_ratio": dense_comm,
        "sla_comm_ratio": sla_comm,
    })

summary = {
    "model": "Wan2.2-MoE I2V",
    "resolution": "480x832",
    "config": "block offload, int8-q8f, enable_cfg=false, 4 steps",
    "sla_attn": "sla_attn triton sparsity_ratio=0.8",
    "dense_t1_s": dense_t1,
    "sla_t1_s": sla_t1,
    "sla_speedup_p1": dense_t1 / sla_t1 if dense_t1 and sla_t1 else None,
    "sp_scaling": rows,
}
out_json = study / "p1_moe_i2v_480_sla_sp_summary.json"
out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "—"

def f3(x):
    return f"{x:.3f}" if x is not None else "—"

def pct_comm(x):
    return f"{100*x:.1f}%" if x is not None else "—"

lines = [
    "# Wan2.2-MoE I2V @ 480×832 — SLA SP scaling",
    "",
    "block offload, int8-q8f, 4 steps, Ulysses SP, self-attn `sla_attn` (triton, 0.8).",
    "",
    f"Dense T₁={f3(dense_t1)}s | SLA T₁={f3(sla_t1)}s | SLA P=1 speedup: {f3(summary['sla_speedup_p1'])}×",
    "",
    "## SP 扩展效率 = T₁/(T×P)",
    "",
    "| P | dense (s) | SLA (s) | dense eff | SLA eff | SLA vs dense | dense comm% | SLA comm% |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['seq_p']} | {f3(r['dense_s'])} | {f3(r['sla_s'])} | "
        f"{pct(r['dense_sp_eff'])} | {pct(r['sla_sp_eff'])} | {f3(r['sla_vs_dense'])}× | "
        f"{pct_comm(r['dense_comm_ratio'])} | {pct_comm(r['sla_comm_ratio'])} |"
    )

md = study / "p1_moe_i2v_480_sla_sp_summary.md"
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
