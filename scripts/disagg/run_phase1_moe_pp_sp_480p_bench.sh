#!/bin/bash
# Wan2.2-MoE I2V @ 480×832: PP=2 and PP×SP scaling efficiency vs pure SP.
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
t1_baseline=73.575
force=${FORCE_RERUN:-0}

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

echo "=== PP=2 single (2 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1
run_if_missing "${study_dir}/p1_moe_480_pp2_single.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "${lightx2v_path}/scripts/disagg/run_phase3_pp_pipeline_bench.py" \
    --mode single --measure_iters 1 --warmup 1 \
    --single_gpu_baseline_s "${t1_baseline}" \
    --config_json "${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_pp2_single.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_pp2_single.log"

echo "=== PP=2 GPipe dual lps=2 (2 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1
run_if_missing "${study_dir}/p1_moe_480_pp2_gpipe_dual_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "${lightx2v_path}/scripts/disagg/run_pp_layers_per_stage_sweep.py" \
    --layers_per_stage_list 2 \
    --single_gpu_baseline_s "${t1_baseline}" \
    --config_json "${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_pp2_gpipe_dual_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_pp2_gpipe_dual_lps2.log"

echo "=== PP×SP P=2×2 hybrid m=2 (4 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_if_missing "${study_dir}/p1_moe_480_pp2_sp2_hybrid_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_hybrid_bench.py" \
    --microbatch_list 2 \
    --layers_per_stage 2 \
    --single_gpu_baseline_s "${t1_baseline}" \
    --config_json "${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_pp2_sp2_hybrid_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_pp2_sp2_hybrid_lps2.log"

echo "=== PP×SP P=2×2 quad (4 GPU, 4 req) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_if_missing "${study_dir}/p1_moe_480_pp2_sp2_quad_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_quad_overlap_bench.py" \
    --layers_per_stage 2 \
    --single_gpu_baseline_s "${t1_baseline}" \
    --config_json "${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_pp2_sp2_quad_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_pp2_sp2_quad_lps2.log"

echo "=== PP×SP P=2×4 hybrid m=2 (8 GPU) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run_if_missing "${study_dir}/p1_moe_480_pp2_sp4_hybrid_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_hybrid_bench.py" \
    --microbatch_list 2 \
    --layers_per_stage 2 \
    --per-request-latency \
    --single_gpu_baseline_s "${t1_baseline}" \
    --config_json "${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_sp4_bench.json" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_pp2_sp4_hybrid_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_pp2_sp4_hybrid_lps2.log"

echo "=== PP×SP P=2×4 quad (8 GPU, 8 req) ===" | tee -a "${study_dir}/phase1_moe_pp_sp_480p_runner.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run_if_missing "${study_dir}/p1_moe_480_pp2_sp4_quad_lps2.json" \
    "${python_executable}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${lightx2v_path}/scripts/disagg/run_pp_sp_quad_overlap_bench.py" \
    --layers_per_stage 2 \
    --quad-layout dual \
    --single_gpu_baseline_s "${t1_baseline}" \
    --config_json "${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_pp2_sp4_bench.json" \
    --model_path "${model_path}" \
    --inputs_cache "${inputs_cache}" \
    --output_json "${study_dir}/p1_moe_480_pp2_sp4_quad_lps2.json" \
    2>&1 | tee -a "${study_dir}/p1_moe_480_pp2_sp4_quad_lps2.log"

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
T1 = 73.575

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

# --- SP baselines ---
sp_rows = []
for p in (1, 2, 4, 8):
    d = load_json(study / f"p1_transformer_seqp{p}.json")
    t = d.get("transformer_compute_s")
    sp_rows.append({"mode": f"SP P={p}", "n_gpu": p, "single_s": t, "sp_eff": sp_eff(T1, t, p) if p > 1 else 1.0})

sp_ov = {}
for p in (2, 4, 8):
    d = load_json(study / f"p3_dual_overlap_seqp{p}.json") if p < 8 else load_json(study / "p3_dual_overlap_moe_i2v_480x832_seqp8.json")
    sp_ov[p] = d.get("dual_a2a_overlap_s")

# --- PP / PP×SP ---
pp_single = load_json(study / "p1_moe_480_pp2_single.json").get("single_transformer_compute_s")
pp_gpipe = load_json(study / "p1_moe_480_pp2_gpipe_dual_lps2.json")
pp_gpipe_single = None
pp_gpipe_dual = None
for r in pp_gpipe.get("results", []):
    if r.get("pp_layers_per_stage") == 2:
        pp_gpipe_single = r.get("single_transformer_s")
        pp_gpipe_dual = r.get("dual_pipeline_2req_s")

hyb22 = load_json(study / "p1_moe_480_pp2_sp2_hybrid_lps2.json")
hyb22_single = hyb22.get("single_transformer_s")
hyb22_dual = None
for r in hyb22.get("gpipe_results", []):
    if r.get("num_microbatches") == 2 and r.get("feasible"):
        hyb22_dual = r.get("gpipe_wall_s")

quad22 = load_json(study / "p1_moe_480_pp2_sp2_quad_lps2.json")
quad22_single = quad22.get("single_transformer_s") or quad22.get("modes", {}).get("single_request", {}).get("wall_s")
quad22_hybrid = quad22.get("modes", {}).get("pp_gpipe_m2_2req", {}).get("wall_s")
quad22_quad = quad22.get("modes", {}).get("quad_pp_sp_overlap", {}).get("wall_s")

hyb24 = load_json(study / "p1_moe_480_pp2_sp4_hybrid_lps2.json")
hyb24_single = hyb24.get("single_transformer_s")
hyb24_dual = None
for r in hyb24.get("gpipe_results", []):
    if r.get("num_microbatches") == 2 and r.get("feasible"):
        hyb24_dual = r.get("gpipe_wall_s")

quad24 = load_json(study / "p1_moe_480_pp2_sp4_quad_lps2.json")
quad24_single = quad24.get("single_transformer_s") or quad24.get("modes", {}).get("single_request", {}).get("wall_s")
quad24_quad = quad24.get("modes", {}).get("quad_pp_sp_overlap", {}).get("wall_s")
quad24_n = quad24.get("num_quad_requests", 4)

pp_sp_rows = [
    {"mode": "PP P=2", "n_gpu": 2, "single_s": pp_single or pp_gpipe_single, "concurrency": 1},
    {"mode": "PP×SP P=2×2", "n_gpu": 4, "single_s": hyb22_single, "concurrency": 1},
    {"mode": "PP×SP P=2×4", "n_gpu": 8, "single_s": hyb24_single, "concurrency": 1},
]
for r in pp_sp_rows:
    r["sp_eff"] = sp_eff(T1, r["single_s"], r["n_gpu"])

overlap_rows = [
    {"mode": "SP dual a2a", "n_gpu": 2, "n_req": 2, "wall_s": sp_ov.get(2)},
    {"mode": "PP GPipe m=2", "n_gpu": 2, "n_req": 2, "wall_s": pp_gpipe_dual},
    {"mode": "SP dual a2a", "n_gpu": 4, "n_req": 2, "wall_s": sp_ov.get(4)},
    {"mode": "PP×SP hybrid m=2", "n_gpu": 4, "n_req": 2, "wall_s": hyb22_dual or quad22_hybrid},
    {"mode": "PP×SP quad+SP ov", "n_gpu": 4, "n_req": 4, "wall_s": quad22_quad},
    {"mode": "SP dual a2a", "n_gpu": 8, "n_req": 2, "wall_s": sp_ov.get(8)},
    {"mode": "PP×SP hybrid m=2", "n_gpu": 8, "n_req": 2, "wall_s": hyb24_dual},
    {"mode": "PP×SP quad+SP ov", "n_gpu": 8, "n_req": quad24_n, "wall_s": quad24_quad},
]
for r in overlap_rows:
    r["overlap_eff"] = ov_eff(T1, r["wall_s"], r["n_gpu"], r["n_req"])

summary = {
    "model": "Wan2.2-MoE I2V",
    "resolution": "480x832",
    "t1_block_offload_s": T1,
    "config": "PP/PP×SP: cpu_offload=false, lps=2, 4 steps; T1: block-offload SP P=1",
    "sp_scaling": sp_rows,
    "pp_pp_sp_single": pp_sp_rows,
    "overlap": overlap_rows,
}
(study / "p1_moe_480_pp_sp_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# Wan2.2-MoE I2V @ 480×832 — PP / PP×SP 扩展效率",
    "",
    f"T₁ = **{T1:.1f} s**（block-offload SP P=1）。PP/PP×SP 路径 **cpu_offload=false**，`pp_layers_per_stage=2`。",
    "",
    "**单请求扩展效率** = `T₁ / (T × N)`；**overlap 效率** = `n·T₁ / (T_overlap × N)`。",
    "",
    "## 1. 纯 SP vs PP / PP×SP（单请求）",
    "",
    "| 配置 | GPU | 单请求 (s) | 扩展效率 | vs 同 N 纯 SP |",
    "| --- | ---: | ---: | ---: | ---: |",
]
sp_by_n = {r["n_gpu"]: r for r in sp_rows}
for r in pp_sp_rows:
    sp_t = sp_by_n.get(r["n_gpu"], {}).get("single_s")
    vs = sp_t / r["single_s"] if sp_t and r["single_s"] else None
    lines.append(
        f"| {r['mode']} | {r['n_gpu']} | {f1(r['single_s'])} | {pct(r['sp_eff'])} | "
        f"{f'{vs:.2f}×' if vs else '—'} |"
    )
for r in sp_rows:
    if r["n_gpu"] in (1, 2, 4, 8):
        lines.append(f"| {r['mode']} | {r['n_gpu']} | {f1(r['single_s'])} | {pct(r['sp_eff'])} | 1.00× |")

lines += [
    "",
    "## 2. 多请求 overlap / GPipe 效率",
    "",
    "| 配置 | GPU | 并发 | wall (s) | overlap 效率 |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for r in overlap_rows:
    lines.append(
        f"| {r['mode']} | {r['n_gpu']} | {r['n_req']} | {f1(r['wall_s'])} | {pct(r['overlap_eff'])} |"
    )

lines += [
    "",
    "## 3. 要点",
    "",
    "- **PP=2 单请求效率 ~52%**：两卡切层但单请求仍须走完 pipeline，远低于 SP P=2（~84%）。",
    "- **PP×SP 单请求**：P=2×2 ~43%、P=2×4 ~39%，均低于同 GPU 数纯 SP。",
    "- **2-req GPipe**：PP=2 lps=2 达 **~94%** overlap 效率，优于 SP P=2 dual（~85%）。",
    "- **PP×SP hybrid m=2**：4 卡 **~79%**、8 卡 **~73%**，优于同 GPU SP dual overlap。",
    "- **PP×SP quad**：4 req / 4 卡 **~87%**；8 req / 8 卡 **~86%**，吞吐优于 hybrid。",
]

(study / "p1_moe_480_pp_sp_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(study / "p1_moe_480_pp_sp_summary.md")
PY
