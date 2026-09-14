#!/bin/bash
# 480p Wan2.2-MoE I2V: SLA attention variants (P=1) + SLA Ulysses SP sweep (P=2,4,8).
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
bench="${lightx2v_path}/scripts/disagg/run_phase1_transformer_bench.py"
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
model_path=/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models
base_cfg="${study_dir}/baseline_seqp1.json"

# Generate variant + SP configs from baseline_seqp1.json
python3 - <<'PY'
import copy
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
base = json.loads((study / "baseline_seqp1.json").read_text(encoding="utf-8"))

variants = {
    "sla_triton": {
        "self_attn_1_type": "sla_attn",
        "sla_attn_setting": {"sparsity_ratio": 0.8, "operator": "triton"},
    },
    "sla_flex_block": {
        "self_attn_1_type": "general_sparse_attn",
        "general_sparse_attn_setting": {
            "sparse_mask_generator": "sla_mask_generator",
            "sparse_setting": {"sparsity_ratio": 0.8},
            "sparse_operator": "flex_block_operator",
        },
    },
    "sla_flashinfer": {
        "self_attn_1_type": "general_sparse_attn",
        "general_sparse_attn_setting": {
            "sparse_mask_generator": "sla_mask_generator",
            "sparse_setting": {"sparsity_ratio": 0.8},
            "sparse_operator": "flashinfer_operator",
        },
    },
    "sla_fa4": {
        "self_attn_1_type": "general_sparse_attn",
        "general_sparse_attn_setting": {
            "sparse_mask_generator": "sla_mask_generator",
            "sparse_setting": {"sparsity_ratio": 0.8},
            "sparse_operator": "spas_fa4_operator",
        },
    },
    "nbhd_sla": {
        "self_attn_1_type": "general_sparse_attn",
        "general_sparse_attn_setting": {
            "sparse_mask_generator": "nbhd_mask_generator",
            "sparse_setting": {
                "nbhd_coefficient": [1.0, 0.5, 0.056],
                "nbhd_min_width": 1.0,
            },
            "sparse_operator": "sla_triton_operator",
        },
    },
}

for name, attn_patch in variants.items():
    cfg = copy.deepcopy(base)
    for k in ("sla_attn_setting", "general_sparse_attn_setting"):
        cfg.pop(k, None)
    cfg.update(attn_patch)
    cfg["parallel"] = {"seq_p_size": 1, "seq_p_attn_type": "ulysses"}
    (study / f"baseline_moe_i2v_480_{name}_seqp1.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )

sla_base = json.loads((study / "baseline_moe_i2v_480_sla_triton_seqp1.json").read_text(encoding="utf-8"))
for p in (2, 4, 8):
    cfg = copy.deepcopy(sla_base)
    cfg["parallel"] = {"seq_p_size": p, "seq_p_attn_type": "ulysses"}
    (study / f"baseline_moe_i2v_480_sla_triton_seqp{p}.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )
print("configs ready")
PY

run_p1() {
    local variant="$1"
    local cfg="${study_dir}/baseline_moe_i2v_480_${variant}_seqp1.json"
    local out="${study_dir}/p1_moe_i2v_480_${variant}_seqp1.json"
    if [[ -f "${out}" ]] && python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
        echo "=== skip variant=${variant} P=1 (exists) ==="
        return 0
    fi
    export CUDA_VISIBLE_DEVICES=0
    echo "=== variant=${variant} P=1 ===" | tee -a "${study_dir}/phase1_sla_480p_runner.log"
    "${python_executable}" "${bench}" \
        --config_json "${cfg}" \
        --model_path "${model_path}" \
        --seq_p_size 1 \
        --inputs_cache "${inputs_cache}" \
        --warmup 1 --measure_iters 1 \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/p1_moe_i2v_480_${variant}_seqp1.log"
}

run_sp() {
    local p="$1"
    local cfg="${study_dir}/baseline_moe_i2v_480_sla_triton_seqp${p}.json"
    local out="${study_dir}/p1_moe_i2v_480_sla_triton_seqp${p}.json"
    if [[ -f "${out}" ]] && python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
        echo "=== skip sla_triton P=${p} (exists) ==="
        return 0
    fi
    local gpus
    gpus=$(seq -s, 0 $((p - 1)))
    export CUDA_VISIBLE_DEVICES="${gpus}"
    echo "=== sla_triton P=${p} gpus=${gpus} ===" | tee -a "${study_dir}/phase1_sla_480p_runner.log"
    "${python_executable}" -m torch.distributed.run \
        --standalone \
        "--nproc_per_node=${p}" \
        "${bench}" \
        --config_json "${cfg}" \
        --model_path "${model_path}" \
        --seq_p_size "${p}" \
        --inputs_cache "${inputs_cache}" \
        --warmup 1 --measure_iters 3 \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/p1_moe_i2v_480_sla_triton_seqp${p}.log"
}

for v in sla_triton sla_flex_block sla_flashinfer sla_fa4 nbhd_sla; do
    run_p1 "${v}" || echo "WARN: variant ${v} failed" | tee -a "${study_dir}/phase1_sla_480p_runner.log"
done

for p in 2 4 8; do
    run_sp "${p}" || echo "WARN: sla_triton P=${p} failed" | tee -a "${study_dir}/phase1_sla_480p_runner.log"
done

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
dense_p1 = 72.75790912200075  # from prior run
dense_sp = {}
for p in (1, 2, 4, 8):
    p3 = study / f"p3_sp_seqp{p}.json"
    if p3.is_file():
        dense_sp[p] = json.loads(p3.read_text())["transformer_compute_s"]

variants = ["sla_triton", "sla_flex_block", "sla_flashinfer", "sla_fa4", "nbhd_sla"]
var_rows = []
for v in variants:
    path = study / f"p1_moe_i2v_480_{v}_seqp1.json"
    row = {"variant": v, "ok": False}
    if path.is_file():
        d = json.loads(path.read_text())
        t = d.get("transformer_compute_s")
        row.update({"ok": t is not None, "transformer_compute_s": t})
        if t:
            row["speedup_vs_dense"] = dense_p1 / t
    var_rows.append(row)

sp_rows = []
sla_p1 = next((r["transformer_compute_s"] for r in var_rows if r["variant"] == "sla_triton" and r.get("transformer_compute_s")), None)
for p in (1, 2, 4, 8):
    row = {"seq_p": p, "dense_s": dense_sp.get(p), "sla_s": None, "sla_speedup_vs_p1": None, "dense_speedup_vs_p1": None}
    if p == 1 and sla_p1:
        row["sla_s"] = sla_p1
    elif p > 1:
        path = study / f"p1_moe_i2v_480_sla_triton_seqp{p}.json"
        if path.is_file():
            row["sla_s"] = json.loads(path.read_text()).get("transformer_compute_s")
    if sla_p1 and row["sla_s"]:
        row["sla_speedup_vs_p1"] = sla_p1 / row["sla_s"]
    if dense_sp.get(1) and row["dense_s"]:
        row["dense_speedup_vs_p1"] = dense_sp[1] / row["dense_s"]
    sp_rows.append(row)

summary = {"variants_p1": var_rows, "sla_sp": sp_rows, "dense_p1_reference": dense_p1, "dense_sp_reference": dense_sp}
out = study / "p1_moe_i2v_480_sla_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    "# 480p MoE I2V — SLA variants & SP scaling",
    "",
    "## P=1 variants (vs dense sage 72.76s)",
    "",
    "| variant | transformer_compute_s | speedup vs dense |",
    "|---|---:|---:|",
]
for r in var_rows:
    t = r.get("transformer_compute_s")
    sp = r.get("speedup_vs_dense", "")
    lines.append(f"| {r['variant']} | {t if t else 'FAIL'} | {f'{sp:.3f}' if sp else ''} |")

lines += [
    "",
    "## SLA triton Ulysses SP (vs dense SP from p3_sp)",
    "",
    "| seq_p | dense (sage) | SLA triton | SLA speedup/P1 | dense speedup/P1 |",
    "|---:|---:|---:|---:|---:|",
]
for r in sp_rows:
    sla_sp = f"{r['sla_speedup_vs_p1']:.3f}" if r.get("sla_speedup_vs_p1") else ""
    dense_sp_up = f"{r['dense_speedup_vs_p1']:.3f}" if r.get("dense_speedup_vs_p1") else ""
    lines.append(
        f"| {r['seq_p']} | {r.get('dense_s', '')} | {r.get('sla_s', '')} | "
        f"{sla_sp} | {dense_sp_up} |"
    )

md = study / "p1_moe_i2v_480_sla_summary.md"
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
