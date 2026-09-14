#!/bin/bash
# Wan2.1-T2V-1.3B @ 480×832: SLA triton SP scaling + dual a2a-overlap efficiency.
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
overlap="${lightx2v_path}/scripts/disagg/run_phase3_dual_overlap_bench.py"
model_path=/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B
inputs_cache="${study_dir}/phase1_t2v_1.3b_encoder_inputs.pt"
seq_ps=${SEQ_PS:-1,2,3,4,6}
overlap_ps=${OVERLAP_PS:-2,3,4,6}
force=${FORCE_RERUN:-0}

python3 - <<'PY'
import copy
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
dense_base = json.loads((study / "baseline_t2v_1.3b_seqp1.json").read_text(encoding="utf-8"))

for p in (1, 2, 3, 4, 6):
    cfg = copy.deepcopy(dense_base)
    cfg["self_attn_1_type"] = "sla_attn"
    cfg["sla_attn_setting"] = {"sparsity_ratio": 0.8, "operator": "triton"}
    cfg["parallel"] = {"seq_p_size": p, "seq_p_attn_type": "ulysses"}
    (study / f"baseline_t2v_1.3b_sla_seqp{p}.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )
print("SLA configs written")
PY

run_phase1() {
    local p="$1"
    local cfg="${study_dir}/baseline_t2v_1.3b_sla_seqp${p}.json"
    local out="${study_dir}/p1_t2v_1.3b_sla_transformer_seqp${p}.json"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip phase1 SLA P=${p} (exists) ==="
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== phase1 SLA P=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_sla_t2v_runner.log"
    local -a cmd=(
        "${python_executable}" "${phase1}"
        --task t2v --model_cls wan2.1
        --model_path "${model_path}"
        --config_json "${cfg}"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --warmup 1 --measure_iters 3
        --output_json "${out}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${phase1}"
            --task t2v --model_cls wan2.1
            --model_path "${model_path}"
            --config_json "${cfg}"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --warmup 1 --measure_iters 3
            --output_json "${out}"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/p1_t2v_1.3b_sla_transformer_seqp${p}.log"
}

run_sp_profile() {
    local p="$1"
    local cfg="${study_dir}/baseline_t2v_1.3b_sla_seqp${p}.json"
    local out="${study_dir}/p3_t2v_1.3b_sla_sp_seqp${p}.json"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip sp_profile SLA P=${p} (exists) ==="
            return 0
        fi
    fi
    [[ "${p}" -eq 1 ]] && return 0
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== sp_profile SLA P=${p} ===" | tee -a "${study_dir}/phase1_sla_t2v_runner.log"
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${phase3_sp}" \
        --seq_p_size "${p}" \
        --task t2v --model_cls wan2.1 \
        --model_path "${model_path}" \
        --config_json "${cfg}" \
        --inputs_cache "${inputs_cache}" \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/p3_t2v_1.3b_sla_sp_seqp${p}.log" || true
}

run_overlap() {
    local p="$1"
    local cfg="${study_dir}/baseline_t2v_1.3b_sla_seqp${p}.json"
    local phase3_json="${study_dir}/p3_t2v_1.3b_sla_sp_seqp${p}.json"
    local out="${study_dir}/p3_dual_overlap_t2v_1.3b_sla_seqp${p}.json"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('dual_a2a_overlap_s') else 1)"; then
            echo "=== skip overlap SLA P=${p} (exists) ==="
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== overlap SLA P=${p} gpus=${CUDA_VISIBLE_DEVICES} ===" | tee -a "${study_dir}/phase1_sla_t2v_runner.log"
    if [[ ! -f "${phase3_json}" ]]; then
        run_sp_profile "${p}"
    fi
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${overlap}" \
        --seq_p_size "${p}" \
        --task t2v --model_cls wan2.1 \
        --model_path "${model_path}" \
        --config_json "${cfg}" \
        --inputs_cache "${inputs_cache}" \
        --phase3_json "${phase3_json}" \
        --align_iters 3 \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/p3_dual_overlap_t2v_1.3b_sla_seqp${p}.log" || echo "WARN overlap P=${p} failed"
}

IFS=',' read -r -a sp_arr <<< "${seq_ps}"
for p in "${sp_arr[@]}"; do
    run_phase1 "${p}" || echo "WARN phase1 P=${p} failed"
done

IFS=',' read -r -a ov_arr <<< "${overlap_ps}"
for p in "${ov_arr[@]}"; do
    run_overlap "${p}" || echo "WARN overlap P=${p} failed"
done

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")

def load_t(path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return d.get("transformer_compute_s")

def load_dense_sp(p):
    return load_t(study / f"p3_t2v_1.3b_sp_seqp{p}.json") or load_t(study / f"p1_t2v_1.3b_transformer_seqp{p}.json")

def load_sla_sp(p):
    return load_t(study / f"p1_t2v_1.3b_sla_transformer_seqp{p}.json") or load_t(study / f"p3_t2v_1.3b_sla_sp_seqp{p}.json")

def eff(t1, t_multi, p):
    if t1 and t_multi and p:
        return t1 / (t_multi * p)
    return None

dense_t1 = load_dense_sp(1)
sla_t1 = load_sla_sp(1)

sp_rows = []
for p in (1, 2, 3, 4, 6):
    dense = load_dense_sp(p)
    sla = load_sla_sp(p)
    sp_rows.append({
        "seq_p": p,
        "dense_s": dense,
        "sla_s": sla,
        "dense_sp_eff": eff(dense_t1, dense, p),
        "sla_sp_eff": eff(sla_t1, sla, p),
        "sla_vs_dense": (dense / sla) if dense and sla else None,
    })

ov_rows = []
for p in (2, 3, 4, 6):
    dense_ov = study / f"p3_dual_overlap_t2v_1.3b_seqp{p}.json"
    sla_ov = study / f"p3_dual_overlap_t2v_1.3b_sla_seqp{p}.json"
    row = {"seq_p": p}
    if dense_ov.is_file():
        d = json.loads(dense_ov.read_text())
        row["dense_overlap_s"] = d.get("dual_a2a_overlap_s")
        row["dense_single_s"] = d.get("single_transformer_s")
        if dense_t1 and d.get("dual_a2a_overlap_s"):
            row["dense_overlap_eff"] = 2 * dense_t1 / (d["dual_a2a_overlap_s"] * p)
    if sla_ov.is_file():
        d = json.loads(sla_ov.read_text())
        row["sla_overlap_s"] = d.get("dual_a2a_overlap_s")
        row["sla_single_s"] = d.get("single_transformer_s")
        if sla_t1 and d.get("dual_a2a_overlap_s"):
            row["sla_overlap_eff"] = 2 * sla_t1 / (d["dual_a2a_overlap_s"] * p)
    ov_rows.append(row)

summary = {
    "model": "Wan2.1-T2V-1.3B",
    "resolution": "480x832",
    "dense_t1_s": dense_t1,
    "sla_t1_s": sla_t1,
    "sla_speedup_p1": dense_t1 / sla_t1 if dense_t1 and sla_t1 else None,
    "sp_scaling": sp_rows,
    "overlap": ov_rows,
}
out_json = study / "p1_t2v_1.3b_sla_summary.json"
out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "—"

def f3(x):
    return f"{x:.3f}" if x is not None else "—"

lines = [
    "# Wan2.1-T2V-1.3B @ 480×832 — SLA SP & overlap",
    "",
    f"Dense T₁={f3(dense_t1)}s | SLA T₁={f3(sla_t1)}s | SLA speedup P=1: {f3(dense_t1/sla_t1 if dense_t1 and sla_t1 else None)}×",
    "",
    "## SP scaling efficiency = T₁/(T×P)",
    "",
    "| P | dense (s) | SLA (s) | dense eff | SLA eff | SLA vs dense |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in sp_rows:
    lines.append(
        f"| {r['seq_p']} | {f3(r['dense_s'])} | {f3(r['sla_s'])} | "
        f"{pct(r['dense_sp_eff'])} | {pct(r['sla_sp_eff'])} | {f3(r['sla_vs_dense'])}× |"
    )

lines += [
    "",
    "## Dual a2a-overlap efficiency = 2·T₁/(T_overlap×P)",
    "",
    "| P | dense overlap (s) | SLA overlap (s) | dense eff | SLA eff |",
    "|---:|---:|---:|---:|---:|",
]
for r in ov_rows:
    lines.append(
        f"| {r['seq_p']} | {f3(r.get('dense_overlap_s'))} | {f3(r.get('sla_overlap_s'))} | "
        f"{pct(r.get('dense_overlap_eff'))} | {pct(r.get('sla_overlap_eff'))} |"
    )

md = study / "p1_t2v_1.3b_sla_summary.md"
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
