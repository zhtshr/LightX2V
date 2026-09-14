#!/bin/bash
# SLA triton @ 256² / 512² / 1024² — small (T2V-1.3B) + large (MoE I2V).
# SP scaling + dual a2a-overlap (where dense baseline exists).
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LIGHTX2V_NCCL_TIMEOUT_S="${LIGHTX2V_NCCL_TIMEOUT_S:-10800}"

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
t2v_model=/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B
moe_model=/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models
image_path=/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg
force=${FORCE_RERUN:-0}
resolutions=${RESOLUTIONS:-256,512,1024}
run_overlap=${RUN_OVERLAP:-1}

write_sla_config() {
    local dense_cfg="$1"
    local sla_cfg="$2"
    python3 - <<PY
import copy, json
from pathlib import Path
src = Path("${dense_cfg}")
dst = Path("${sla_cfg}")
if not src.is_file():
    raise SystemExit(f"missing dense config: {src}")
data = json.loads(src.read_text(encoding="utf-8"))
data["self_attn_1_type"] = "sla_attn"
data["sla_attn_setting"] = {"sparsity_ratio": 0.8, "operator": "triton"}
data.pop("general_sparse_attn_setting", None)
dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

run_phase1() {
    local task="$1" model_cls="$2" model_path="$3" cfg="$4" cache="$5" p="$6" out="$7"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('transformer_compute_s') else 1)"; then
            echo "=== skip ${out} (exists) ==="
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== phase1 ${out} P=${p} ===" | tee -a "${study_dir}/phase1_sla_resolution_runner.log"
    local -a cmd=(
        "${python_executable}" "${phase1}"
        --task "${task}" --model_cls "${model_cls}"
        --model_path "${model_path}"
        --config_json "${cfg}"
        --seq_p_size "${p}"
        --inputs_cache "${cache}"
        --image_path "${image_path}"
        --warmup 1 --measure_iters 1
        --output_json "${out}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${phase1}"
            --task "${task}" --model_cls "${model_cls}"
            --model_path "${model_path}"
            --config_json "${cfg}"
            --seq_p_size "${p}"
            --inputs_cache "${cache}"
            --image_path "${image_path}"
            --warmup 1 --measure_iters 1
            --output_json "${out}"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/$(basename "${out}" .json).log" || echo "WARN failed ${out}"
}

run_overlap_case() {
    local task="$1" model_cls="$2" model_path="$3" cfg="$4" cache="$5" p="$6" out="$7" p3_out="$8"
    if [[ "${force}" != "1" && -f "${out}" ]]; then
        if python3 -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('dual_a2a_overlap_s') else 1)"; then
            echo "=== skip overlap ${out} (exists) ==="
            return 0
        fi
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    if [[ ! -f "${p3_out}" || "${force}" == "1" ]]; then
        "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
            "${phase3_sp}" \
            --seq_p_size "${p}" --task "${task}" --model_cls "${model_cls}" \
            --model_path "${model_path}" \
            --config_json "${cfg}" --inputs_cache "${cache}" \
            --output_json "${p3_out}" \
            2>&1 | tee -a "${study_dir}/$(basename "${p3_out}" .json).log" || true
    fi
    echo "=== overlap ${out} P=${p} ===" | tee -a "${study_dir}/phase1_sla_resolution_runner.log"
    "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}" \
        "${overlap}" \
        --seq_p_size "${p}" --task "${task}" --model_cls "${model_cls}" \
        --model_path "${model_path}" \
        --config_json "${cfg}" --inputs_cache "${cache}" \
        --phase3_json "${p3_out}" --align_iters 3 \
        --output_json "${out}" \
        2>&1 | tee -a "${study_dir}/$(basename "${out}" .json).log" || echo "WARN overlap failed ${out}"
}

IFS=',' read -r -a RES_ARR <<< "${resolutions}"

for res in "${RES_ARR[@]}"; do
    # --- small model ---
    t2v_cache="${study_dir}/phase1_t2v_1.3b_${res}x${res}_encoder_inputs.pt"
    if [[ ! -f "${t2v_cache}" ]]; then
        t2v_cache="${study_dir}/phase1_t2v_1.3b_encoder_inputs.pt"
    fi
    for p in 1 2 3 4 6; do
        dense_cfg="${study_dir}/baseline_t2v_1.3b_${res}x${res}_seqp${p}.json"
        sla_cfg="${study_dir}/baseline_t2v_1.3b_sla_${res}x${res}_seqp${p}.json"
        [[ -f "${dense_cfg}" ]] || continue
        write_sla_config "${dense_cfg}" "${sla_cfg}"
        run_phase1 t2v wan2.1 "${t2v_model}" "${sla_cfg}" "${t2v_cache}" "${p}" \
            "${study_dir}/p1_t2v_1.3b_sla_${res}x${res}_seqp${p}.json"
    done
    if [[ "${run_overlap}" == "1" && "${res}" -le 512 ]]; then
        for p in 3 4 6; do
            dense_cfg="${study_dir}/baseline_t2v_1.3b_${res}x${res}_seqp${p}.json"
            sla_cfg="${study_dir}/baseline_t2v_1.3b_sla_${res}x${res}_seqp${p}.json"
            [[ -f "${dense_cfg}" ]] || continue
            write_sla_config "${dense_cfg}" "${sla_cfg}"
            run_overlap_case t2v wan2.1 "${t2v_model}" "${sla_cfg}" "${t2v_cache}" "${p}" \
                "${study_dir}/p3_dual_overlap_t2v_1.3b_sla_${res}x${res}_seqp${p}.json" \
                "${study_dir}/p3_t2v_1.3b_sla_${res}x${res}_sp_seqp${p}.json"
        done
    fi

    # --- large model ---
    moe_cache="${study_dir}/phase1_moe_i2v_${res}x${res}_encoder_inputs.pt"
    for p in 1 2 4 8; do
        if ! python3 -c "import sys; sys.exit(0 if 40 % ${p} == 0 else 1)"; then
            continue
        fi
        dense_cfg="${study_dir}/baseline_moe_i2v_${res}x${res}_seqp${p}.json"
        sla_cfg="${study_dir}/baseline_moe_i2v_sla_${res}x${res}_seqp${p}.json"
        [[ -f "${dense_cfg}" ]] || continue
        write_sla_config "${dense_cfg}" "${sla_cfg}"
        run_phase1 i2v wan2.2_moe "${moe_model}" "${sla_cfg}" "${moe_cache}" "${p}" \
            "${study_dir}/p1_moe_i2v_sla_${res}x${res}_seqp${p}.json"
    done
    if [[ "${run_overlap}" == "1" ]]; then
        for p in 4 8; do
            dense_cfg="${study_dir}/baseline_moe_i2v_${res}x${res}_seqp${p}.json"
            sla_cfg="${study_dir}/baseline_moe_i2v_sla_${res}x${res}_seqp${p}.json"
            [[ -f "${dense_cfg}" ]] || continue
            write_sla_config "${dense_cfg}" "${sla_cfg}"
            run_overlap_case i2v wan2.2_moe "${moe_model}" "${sla_cfg}" "${moe_cache}" "${p}" \
                "${study_dir}/p3_dual_overlap_moe_i2v_sla_${res}x${res}_seqp${p}.json" \
                "${study_dir}/p3_moe_i2v_sla_${res}x${res}_sp_seqp${p}.json"
        done
    fi
done

python3 - <<'PY'
import json
from pathlib import Path

study = Path("/root/zht/LightX2V/save_results/optimization_study")
resolutions = [256, 512, 1024]

def load_t(path):
    p = study / path
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("transformer_compute_s")

def load_ov(path):
    p = study / path
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("dual_a2a_overlap_s")

def sp_eff(t1, t, n):
    return t1 / (t * n) if t1 and t and n else None

def ov_eff(t1, t, n, nreq=2):
    return nreq * t1 / (t * n) if t1 and t and nreq else None

def mark_x(t1, single_eff, ov_s, ov_eff):
    if ov_s is not None and ov_eff is not None:
        if ov_eff < 0.70 or (t1 and ov_s > t1):
            return True
        return False
    if single_eff is not None and single_eff < 0.70:
        return True
    return False

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "—"

def f1(x):
    if x is None:
        return "—"
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"

def row(attn, cfg, offload, gpu, single, seff, ov, oeff, nreq, t1):
    x = "×" if mark_x(t1, seff, ov, oeff) else ""
    return {
        "x": x, "attn": attn, "config": cfg, "offload": offload, "gpu": gpu,
        "single_s": single, "sp_eff": seff, "overlap_s": ov, "overlap_eff": oeff, "n_req": nreq,
    }

summary = {"resolutions": {}}

t2v_ps = [1, 2, 3, 4, 6]
t2v_ov_ps = {256: [3, 4, 6], 512: [3, 4, 6], 1024: []}
moe_ps = [1, 2, 4, 8]
moe_ov_ps = {256: [4, 8], 512: [4, 8], 1024: [4, 8]}

md_lines = [
    "# SLA 稀疏 — 多分辨率总览（256² / 512² / 1024²）",
    "",
    "口径同 [`sla_benchmark_summary.md`](sla_benchmark_summary.md) / [`motivation.md`](motivation.md)。",
    "",
]

for res in resolutions:
    rows = []
    # small
    t2v_dt1 = load_t(f"p1_t2v_1.3b_{res}x{res}_seqp1.json")
    t2v_st1 = load_t(f"p1_t2v_1.3b_sla_{res}x{res}_seqp1.json")
    for attn, t1, prefix in (("稠密", t2v_dt1, "p1_t2v_1.3b"), ("SLA", t2v_st1, "p1_t2v_1.3b_sla")):
        if t1 is None and attn == "SLA":
            continue
        for p in t2v_ps:
            single = load_t(f"{prefix}_{res}x{res}_seqp{p}.json")
            if single is None:
                continue
            seff = 1.0 if p == 1 else sp_eff(t1, single, p)
            ov = load_ov(f"p3_dual_overlap_t2v_1.3b_{'' if attn=='稠密' else 'sla_'}{res}x{res}_seqp{p}.json") if p in t2v_ov_ps[res] else None
            oeff = ov_eff(t1, ov, p) if ov else None
            rows.append(row(attn, f"SP P={p}", "否", p, single, seff, ov, oeff, 2 if ov else None, t1))
    # large
    dt1 = load_t(f"p1_moe_i2v_{res}x{res}_seqp1.json")
    st1 = load_t(f"p1_moe_i2v_sla_{res}x{res}_seqp1.json")
    for attn, t1, prefix in (("稠密", dt1, "p1_moe_i2v"), ("SLA", st1, "p1_moe_i2v_sla")):
        for p in moe_ps:
            single = load_t(f"{prefix}_{res}x{res}_seqp{p}.json")
            if single is None:
                continue
            seff = 1.0 if p == 1 else sp_eff(t1, single, p)
            ov_prefix = "p3_dual_overlap_moe_i2v_" + ("sla_" if attn == "SLA" else "")
            ov = load_ov(f"{ov_prefix}{res}x{res}_seqp{p}.json") if p in moe_ov_ps[res] else None
            oeff = ov_eff(t1, ov, p) if ov else None
            rows.append(row(attn, f"SP P={p}", "block", p, single, seff, ov, oeff, 2 if ov else None, t1))

    summary["resolutions"][str(res)] = {
        "t2v_dense_t1": t2v_dt1,
        "t2v_sla_t1": t2v_st1,
        "moe_dense_t1": dt1,
        "moe_sla_t1": st1,
        "rows": rows,
    }

    md_lines += [
        f"## {res}×{res}",
        "",
        f"小模型 T₁：稠密 **{f1(summary['resolutions'][str(res)]['t2v_dense_t1'])} s** | SLA **{f1(summary['resolutions'][str(res)]['t2v_sla_t1'])} s**",
        f"大模型 T₁：稠密 **{f1(summary['resolutions'][str(res)]['moe_dense_t1'])} s** | SLA **{f1(summary['resolutions'][str(res)]['moe_sla_t1'])} s**",
        "",
        "| × | 模型 | attn | 配置 | offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |",
        "| :---: | --- | :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        model = "T2V-1.3B" if r["offload"] == "否" else "MoE I2V"
        md_lines.append(
            f"| {r['x']} | {model} | {r['attn']} | {r['config']} | {r['offload']} | {r['gpu']} | "
            f"{f1(r['single_s'])} | {pct(r['sp_eff'])} | {f1(r['overlap_s'])} | {pct(r['overlap_eff'])} | "
            f"{r['n_req'] if r['overlap_s'] else '—'} |"
        )
    md_lines.append("")

(study / "p1_sla_resolution_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
(study / "p1_sla_resolution_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(study / "p1_sla_resolution_summary.md")
PY
