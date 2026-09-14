#!/bin/bash
# Wan2.2-MoE I2V @ 480×832: SLA triton PP=2 and PP×SP scaling vs dense PP/PP×SP.
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
model_path=/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models
inputs_cache="${study_dir}/phase1_encoder_inputs.pt"
force=${FORCE_RERUN:-0}

t1_sla=$(python3 -c "
import json, sys
p='${study_dir}/p1_moe_i2v_480_sla_triton_seqp1.json'
try:
    t=json.load(open(p))['transformer_compute_s']
    print(f'{t:.3f}')
except Exception:
    print('54.429', file=sys.stderr)
    print('54.429')
")

write_sla_pp_config() {
    local dense_cfg="$1"
    local sla_cfg="$2"
    python3 - <<PY
import json
from pathlib import Path
src = Path("${dense_cfg}")
dst = Path("${sla_cfg}")
data = json.loads(src.read_text(encoding="utf-8"))
data["self_attn_1_type"] = "sla_attn"
data["sla_attn_setting"] = {"sparsity_ratio": 0.8, "operator": "triton"}
data.pop("general_sparse_attn_setting", None)
dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

for base in wan22_moe_i2v_pp2_bench wan22_moe_i2v_pp2_sp2_bench wan22_moe_i2v_pp2_sp4_bench; do
    dense="${lightx2v_path}/configs/disagg/baseline/${base}.json"
    sla="${study_dir}/baseline_moe_i2v_480_sla_${base#wan22_moe_i2v_}.json"
    write_sla_pp_config "${dense}" "${sla}"
done

cfg_pp2="${study_dir}/baseline_moe_i2v_480_sla_pp2_bench.json"
cfg_sp2="${study_dir}/baseline_moe_i2v_480_sla_pp2_sp2_bench.json"
cfg_sp4="${study_dir}/baseline_moe_i2v_480_sla_pp2_sp4_bench.json"

run_if_missing() {
    local out="$1"
    shift
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d else 1)"; then
            echo "=== skip (exists) ${out} ==="
            return 0
        fi
    fi
    "$@"
}

echo "=== SLA PP=2 single (2 GPU), T1=${t1_sla} ===" | tee -a "${study_dir}/phase1_moe_pp_sp_sla_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1
run_if_missing "${study_dir}/p1_moe_480_sla_pp2_single.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "${lightx2v_path}/scripts/disagg/run_phase3_pp_pipeline_bench.py" \
    --mode single --measure_iters 1 --warmup 1 \
    --single_gpu_baseline_s "${t1_sla}" \
    --config_json "${cfg_pp2}" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_sla_pp2_single.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_sla_pp2_single.log"

echo "=== SLA PP=2 GPipe dual lps=2 (2 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_sla_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1
run_if_missing "${study_dir}/p1_moe_480_sla_pp2_gpipe_dual_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "${lightx2v_path}/scripts/disagg/run_pp_layers_per_stage_sweep.py" \
    --layers_per_stage_list 2 \
    --single_gpu_baseline_s "${t1_sla}" \
    --config_json "${cfg_pp2}" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_sla_pp2_gpipe_dual_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_sla_pp2_gpipe_dual_lps2.log"

echo "=== SLA PP×SP P=2×2 hybrid m=2 (4 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_sla_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_if_missing "${study_dir}/p1_moe_480_sla_pp2_sp2_hybrid_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_hybrid_bench.py" \
    --microbatch_list 2 \
    --layers_per_stage 2 \
    --single_gpu_baseline_s "${t1_sla}" \
    --config_json "${cfg_sp2}" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_sla_pp2_sp2_hybrid_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_sla_pp2_sp2_hybrid_lps2.log"

echo "=== SLA PP×SP P=2×2 quad (4 GPU, 4 req) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_sla_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_if_missing "${study_dir}/p1_moe_480_sla_pp2_sp2_quad_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_quad_overlap_bench.py" \
    --layers_per_stage 2 \
    --single_gpu_baseline_s "${t1_sla}" \
    --config_json "${cfg_sp2}" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_sla_pp2_sp2_quad_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_sla_pp2_sp2_quad_lps2.log"

echo "=== SLA PP×SP P=2×4 hybrid m=2 (8 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_sla_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run_if_missing "${study_dir}/p1_moe_480_sla_pp2_sp4_hybrid_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_hybrid_bench.py" \
    --microbatch_list 2 \
    --layers_per_stage 2 \
    --per-request-latency \
    --single_gpu_baseline_s "${t1_sla}" \
    --config_json "${cfg_sp4}" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_sla_pp2_sp4_hybrid_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_sla_pp2_sp4_hybrid_lps2.log"

echo "=== SLA PP×SP P=2×4 quad (8 GPU, 8 req) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_sla_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run_if_missing "${study_dir}/p1_moe_480_sla_pp2_sp4_quad_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_quad_overlap_bench.py" \
    --layers_per_stage 2 \
    --quad-layout dual \
    --single_gpu_baseline_s "${t1_sla}" \
    --config_json "${cfg_sp4}" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_sla_pp2_sp4_quad_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_sla_pp2_sp4_quad_lps2.log"

python3 - <<'PY'
import json
import os
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
T1_DENSE = 73.575
T1_SLA = float(os.environ.get("SLA_T1", "54.429"))
t1_path = study / "p1_moe_i2v_480_sla_triton_seqp1.json"
if t1_path.is_file():
    T1_SLA = json.loads(t1_path.read_text()).get("transformer_compute_s", T1_SLA)

def load_json(p):
    return json.loads(p.read_text()) if p.is_file() else {}

def sp_eff(t1, t, n):
    return t1 / (t * n) if t1 and t and n else None

def ov_eff(t1, t, n, reqs=2):
    return reqs * t1 / (t * n) if t1 and t and n else None

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "—"

def f1(x):
    return f"{x:.1f}" if x is not None else "—"

def mark_x(single_eff, ov_eff=None, ov_s=None, t1=None):
    if ov_eff is not None:
        if ov_eff < 0.70 or (t1 and ov_s and ov_s > t1):
            return "×"
        return ""
    if single_eff is not None and single_eff < 0.70:
        return "×"
    return ""

def parse_pp_sp(prefix):
    pp_single = load_json(study / f"{prefix}_pp2_single.json").get("single_transformer_compute_s")
    pp_gpipe = load_json(study / f"{prefix}_pp2_gpipe_dual_lps2.json")
    pp_gpipe_dual = None
    for r in pp_gpipe.get("results", []):
        if r.get("pp_layers_per_stage") == 2:
            pp_gpipe_dual = r.get("dual_pipeline_2req_s")
    hyb22 = load_json(study / f"{prefix}_pp2_sp2_hybrid_lps2.json")
    hyb22_single = hyb22.get("single_transformer_s")
    hyb22_dual = None
    for r in hyb22.get("gpipe_results", []):
        if r.get("num_microbatches") == 2 and r.get("feasible"):
            hyb22_dual = r.get("gpipe_wall_s")
    quad22 = load_json(study / f"{prefix}_pp2_sp2_quad_lps2.json")
    quad22_single = quad22.get("single_transformer_s") or quad22.get("modes", {}).get("single_request", {}).get("wall_s")
    quad22_quad = quad22.get("modes", {}).get("quad_pp_sp_overlap", {}).get("wall_s")
    hyb24 = load_json(study / f"{prefix}_pp2_sp4_hybrid_lps2.json")
    hyb24_single = hyb24.get("single_transformer_s")
    hyb24_dual = None
    for r in hyb24.get("gpipe_results", []):
        if r.get("num_microbatches") == 2 and r.get("feasible"):
            hyb24_dual = r.get("gpipe_wall_s")
    quad24 = load_json(study / f"{prefix}_pp2_sp4_quad_lps2.json")
    quad24_single = quad24.get("single_transformer_s") or quad24.get("modes", {}).get("single_request", {}).get("wall_s")
    quad24_quad = quad24.get("modes", {}).get("quad_pp_sp_overlap", {}).get("wall_s")
    quad24_n = quad24.get("num_quad_requests", 4)
    return {
        "pp_single": pp_single, "pp_gpipe_dual": pp_gpipe_dual,
        "hyb22_single": hyb22_single, "hyb22_dual": hyb22_dual,
        "quad22_single": quad22_single, "quad22_quad": quad22_quad,
        "hyb24_single": hyb24_single, "hyb24_dual": hyb24_dual,
        "quad24_single": quad24_single, "quad24_quad": quad24_quad, "quad24_n": quad24_n,
    }

dense = parse_pp_sp("p1_moe_480")
sla = parse_pp_sp("p1_moe_480_sla")

# SLA SP baselines
sla_sp = {}
for p in (1, 2, 4, 8):
    d = load_json(study / f"p1_moe_i2v_480_sla_triton_seqp{p}.json")
    sla_sp[p] = d.get("transformer_compute_s")
sla_ov = {}
for p in (2, 4, 8):
    d = load_json(study / f"p3_dual_overlap_moe_i2v_480x832_sla_seqp{p}.json")
    sla_ov[p] = d.get("dual_a2a_overlap_s")

def single_rows(t1, data, attn):
    rows = []
    for mode, n, t in [
        ("PP P=2", 2, data["pp_single"]),
        ("PP×SP P=2×2", 4, data["hyb22_single"]),
        ("PP×SP P=2×4", 8, data["hyb24_single"]),
    ]:
        eff = sp_eff(t1, t, n)
        rows.append({"attn": attn, "mode": mode, "n_gpu": n, "single_s": t, "sp_eff": eff, "x": mark_x(eff)})
    return rows

def overlap_rows(t1, data, attn):
    rows = []
    items = [
        ("PP GPipe m=2", 2, 2, data["pp_gpipe_dual"]),
        ("PP×SP hybrid m=2", 4, 2, data["hyb22_dual"]),
        ("PP×SP quad+SP ov", 4, 4, data["quad22_quad"]),
        ("PP×SP hybrid m=2", 8, 2, data["hyb24_dual"]),
        ("PP×SP quad+SP ov", 8, data.get("quad24_n", 4), data["quad24_quad"]),
    ]
    for mode, n, nreq, wall in items:
        eff = ov_eff(t1, wall, n, nreq)
        rows.append({"attn": attn, "mode": mode, "n_gpu": n, "n_req": nreq, "wall_s": wall,
                     "overlap_eff": eff, "x": mark_x(None, eff, wall, t1)})
    return rows

summary = {
    "model": "Wan2.2-MoE I2V",
    "resolution": "480x832",
    "t1_dense_block_offload_s": T1_DENSE,
    "t1_sla_block_offload_s": T1_SLA,
    "config": "SLA triton sparsity=0.8; PP/PP×SP cpu_offload=false, lps=2",
    "sla_sp_scaling": [{"p": p, "single_s": sla_sp.get(p), "sp_eff": sp_eff(T1_SLA, sla_sp.get(p), p) if p > 1 else 1.0} for p in (1,2,4,8)],
    "dense_pp_sp_single": single_rows(T1_DENSE, dense, "稠密"),
    "sla_pp_sp_single": single_rows(T1_SLA, sla, "SLA"),
    "dense_pp_sp_overlap": overlap_rows(T1_DENSE, dense, "稠密"),
    "sla_pp_sp_overlap": overlap_rows(T1_SLA, sla, "SLA"),
}
(study / "p1_moe_480_pp_sp_sla_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Wan2.2-MoE I2V @ 480×832 — SLA PP / PP×SP 扩展效率",
    "",
    f"SLA T₁ = **{T1_SLA:.1f} s**（block-offload SP P=1）；稠密 T₁ = **{T1_DENSE:.1f} s**。PP/PP×SP 路径 **cpu_offload=false**，`pp_layers_per_stage=2`，`sla_attn` triton 0.8。",
    "",
    "**单请求扩展效率** = `T₁/(T×N)`；**overlap 效率** = `n·T₁/(T_overlap×N)`。×：单请求 <70% 或 overlap <70%。",
    "",
    "## 1. 单请求：稠密 vs SLA",
    "",
    "| × | attn | 配置 | GPU | 单请求 (s) | 扩展效率 |",
    "| :---: | :---: | --- | ---: | ---: | ---: |",
]
for r in summary["dense_pp_sp_single"] + summary["sla_pp_sp_single"]:
    lines.append(f"| {r['x']} | {r['attn']} | {r['mode']} | {r['n_gpu']} | {f1(r['single_s'])} | {pct(r['sp_eff'])} |")

lines += [
    "",
    "## 2. SLA SP 对照（同 T₁）",
    "",
    "| × | 配置 | GPU | 单请求 (s) | 扩展效率 |",
    "| :---: | --- | ---: | ---: | ---: |",
]
for p in (1, 2, 4, 8):
    t = sla_sp.get(p)
    eff = 1.0 if p == 1 else sp_eff(T1_SLA, t, p)
    x = mark_x(eff)
    lines.append(f"| {x} | SP P={p} | {p} | {f1(t)} | {pct(eff)} |")

lines += [
    "",
    "## 3. 多请求 overlap：稠密 vs SLA",
    "",
    "| × | attn | 配置 | GPU | 并发 | wall (s) | overlap 效率 |",
    "| :---: | :---: | --- | ---: | ---: | ---: | ---: |",
]
for r in summary["dense_pp_sp_overlap"] + summary["sla_pp_sp_overlap"]:
    lines.append(f"| {r['x']} | {r['attn']} | {r['mode']} | {r['n_gpu']} | {r['n_req']} | {f1(r['wall_s'])} | {pct(r['overlap_eff'])} |")

lines += [
    "",
    "## 4. SLA SP dual-overlap 对照",
    "",
    "| × | 配置 | GPU | wall (s) | overlap 效率 |",
    "| :---: | --- | ---: | ---: | ---: |",
]
for p in (2, 4, 8):
    wall = sla_ov.get(p)
    eff = ov_eff(T1_SLA, wall, p)
    x = mark_x(None, eff, wall, T1_SLA)
    lines.append(f"| {x} | SP dual a2a | {p} | {f1(wall)} | {pct(eff)} |")

(study / "p1_moe_480_pp_sp_sla_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(study / "p1_moe_480_pp_sp_sla_summary.md")
PY
