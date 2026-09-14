#!/bin/bash
# Sweep SF dual-request a2a-overlap at seq_p in {1,2,3,4,6}.

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
study_dir="${lightx2v_path}/save_results/optimization_study"
mkdir -p "${study_dir}"

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

baseline_conda_env=${BASELINE_CONDA_ENV:-lightx2v}
if [[ "${CONDA_DEFAULT_ENV:-}" != "${baseline_conda_env}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        set +u
        eval "$(conda shell.bash hook)"
        conda activate "${baseline_conda_env}"
        set -u
    fi
fi

python_executable=${BASELINE_PYTHON_EXECUTABLE:-/root/install/miniconda3/envs/lightx2v/bin/python}
bench="${lightx2v_path}/scripts/disagg/run_sf_transformer_dual_overlap_bench.py"
model_path=${MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/self_forcing/wan_t2v_sf_sp_bench.json}
inputs_cache="${study_dir}/sf_phase1_encoder_inputs.pt"
baseline_p1="${study_dir}/sf_transformer_seqp1.json"
force=${FORCE_RERUN:-0}
seq_p_list=${SEQ_P_LIST:-1,2,3,4,6}
warmup=${WARMUP:-1}

run_case() {
    local p="$1"
    local out="${study_dir}/sf_dual_overlap_seqp${p}.json"
    if [[ -f "${out}" && "${force}" != "1" ]]; then
        if "${python_executable}" -c "import json,sys; d=json.load(open('${out}')); sys.exit(0 if d.get('dual_back_to_back_s') else 1)"; then
            echo "=== skip SF dual overlap seq_p=${p} (exists) ==="
            return 0
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))"
    echo "=== SF dual overlap seq_p=${p} gpus=${CUDA_VISIBLE_DEVICES} ==="

    local -a cmd=(
        "${python_executable}" "${bench}"
        --model_path "${model_path}"
        --config_json "${config_json}"
        --seq_p_size "${p}"
        --inputs_cache "${inputs_cache}"
        --baseline_p1_json "${baseline_p1}"
        --warmup "${warmup}"
        --output_json "${out}"
    )
    if [[ "${p}" -gt 1 ]]; then
        cmd=(
            "${python_executable}" -m torch.distributed.run --standalone "--nproc_per_node=${p}"
            "${bench}"
            --model_path "${model_path}"
            --config_json "${config_json}"
            --seq_p_size "${p}"
            --inputs_cache "${inputs_cache}"
            --baseline_p1_json "${baseline_p1}"
            --warmup "${warmup}"
            --output_json "${out}"
        )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${study_dir}/sf_dual_overlap_seqp${p}.log"
}

IFS=',' read -r -a p_values <<< "${seq_p_list}"
for p in "${p_values[@]}"; do
    p="$(echo "${p}" | tr -d ' ')"
    [[ -z "${p}" ]] && continue
    if [[ "${p}" -eq 8 ]]; then
        echo "=== skip P=8: num_heads=12 not divisible by 8 ==="
        continue
    fi
    run_case "${p}"
done

"${python_executable}" - <<PY
import json
from pathlib import Path

study = Path("${study_dir}")
p_values = [int(x.strip()) for x in "${seq_p_list}".split(",") if x.strip()]
baseline_p1 = None
p1_path = study / "sf_transformer_seqp1.json"
if p1_path.is_file():
    baseline_p1 = json.loads(p1_path.read_text()).get("transformer_compute_s")

rows = []
for p in p_values:
    path = study / f"sf_dual_overlap_seqp{p}.json"
    if not path.is_file():
        rows.append({"seq_p": p})
        continue
    d = json.loads(path.read_text())
    single = d.get("single_request_s")
    b2b = d.get("dual_back_to_back_s")
    ov = d.get("dual_a2a_overlap_s")
    ser = d.get("dual_a2a_serial_s")
    t1 = d.get("baseline_p1_single_s") or baseline_p1
    eff = d.get("overlap_efficiency_vs_p1_ideal")
    if eff is None and t1 and ov and p > 0:
        eff = 2 * t1 / (ov * p)
    rows.append({
        "seq_p": p,
        "single_s": single,
        "dual_b2b_s": b2b,
        "dual_a2a_serial_s": ser,
        "dual_a2a_overlap_s": ov,
        "overlap_vs_b2b": d.get("speedup", {}).get("overlap_vs_back_to_back"),
        "overlap_vs_serial": d.get("speedup", {}).get("overlap_vs_a2a_serial"),
        "overlap_eff_vs_p1": eff,
        "overlap_rps": d.get("throughput", {}).get("dual_a2a_overlap_rps"),
        "error": d.get("overlap_error"),
    })

summary = {"model_cls": "wan2.1_sf", "baseline_p1_s": baseline_p1, "results": rows}
summary_path = study / "sf_dual_overlap_scaling.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    "# SF (wan2.1_sf) Dual-Request Overlap vs SP",
    "",
    "Metric: 2 in-flight AR denoise (7 chunk x 4 step + rerun).",
    "P>1: a2a-overlap = comm on comm_stream || other tenant cross_ffn.",
    "Overlap eff vs ideal = 2*T1 / (dual_overlap_wall * P), T1 = P=1 single transformer_s.",
    "",
    "| seq_p | single (s) | dual b2b (s) | dual a2a serial | dual a2a overlap | vs b2b | vs serial | overlap eff |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    def f(v, n=3):
        return "—" if v is None else f"{v:.{n}f}"
    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"
    lines.append(
        f"| {r.get('seq_p')} | {f(r.get('single_s'))} | {f(r.get('dual_b2b_s'))} | "
        f"{f(r.get('dual_a2a_serial_s'))} | {f(r.get('dual_a2a_overlap_s'))} | "
        f"{f(r.get('overlap_vs_b2b'), 2)}x | {f(r.get('overlap_vs_serial'), 2)}x | {pct(r.get('overlap_eff_vs_p1'))} |"
    )
md_path = study / "sf_dual_overlap_scaling.md"
md_path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
print(summary_path)
print(md_path)
PY

echo "Done. See ${study_dir}/sf_dual_overlap_scaling.md"
