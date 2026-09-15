#!/bin/bash
# Wan2.1-T2V-1.3B @ 50 steps, 480×832: DP8 baseline vs Disagg 1:6:1 (8×A10).
#
# Fair 8-GPU closed-loop batch throughput = completed / batch_wall_s.
# Default: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  N_REQ=24  NUM_WORKERS=8
set -euo pipefail

export lightx2v_path=/root/zht/LightX2V
# Full 8-GPU map (override with CUDA_VISIBLE_DEVICES_OVERRIDE if needed).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}
# Prevent generate-mode from overlaying wan22 4-step stages onto T2V-50 config.
export DISAGG_WORKLOAD_STAGES_JSON=${DISAGG_WORKLOAD_STAGES_JSON:-${lightx2v_path}/configs/disagg/wan21_t2v_workload_50step.json}

stamp=${BENCH_STAMP:-$(date +%Y%m%d_%H%M%S)_wan21_t2v50_8gpu}
out_root=${BENCH_OUT_DIR:-${lightx2v_path}/save_results/throughput_${stamp}}
mkdir -p "${out_root}"

n_req=${N_REQ:-24}
prompt=${PROMPT:-"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."}
negative_prompt=${NEGATIVE_PROMPT:-"镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量"}
model_path=${MODEL_PATH:-${lightx2v_path}/models/Wan-AI/Wan2.1-T2V-1.3B}
gpus=${CUDA_VISIBLE_DEVICES}
num_workers=${NUM_WORKERS:-8}
disagg_controller_cfg=${DISAGG_CONTROLLER_CFG_OVERRIDE:-${lightx2v_path}/configs/disagg/single_node/wan21_t2v_1p3b_50step_controller_8gpu.json}

export PYTHONPATH=${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
export model_path

source /root/install/miniconda3/etc/profile.d/conda.sh
conda activate lightx2v

summary=${out_root}/SUMMARY.md
echo "# Wan2.1-T2V-1.3B 50-step throughput (DP vs Disagg)" | tee "${summary}"
echo "" | tee -a "${summary}"
echo "- stamp: ${stamp}" | tee -a "${summary}"
echo "- GPUs: ${gpus} (workers=${num_workers})" | tee -a "${summary}"
echo "- N_REQ: ${n_req}" | tee -a "${summary}"
echo "- model: ${model_path}" | tee -a "${summary}"
echo "" | tee -a "${summary}"

bash "${lightx2v_path}/scripts/disagg/kill_service.sh" || true
bash "${lightx2v_path}/scripts/disagg/kill_base.sh" || true
sleep 2

########################################
# 1) Baseline DP
########################################
dp_out=${out_root}/baseline_dp
mkdir -p "${dp_out}"
echo "===== $(date -Is) START baseline DP =====" | tee -a "${summary}"

export CUDA_VISIBLE_DEVICES="${gpus}"
export model_path="${model_path}"
# base.sh under set -u
export PYTHONPATH=${PYTHONPATH:-}

(
  cd "${lightx2v_path}"
  source "${lightx2v_path}/scripts/base/base.sh"
  python -m lightx2v.disagg.examples.run_controller \
    --mode controller \
    --request_source generate \
    --generate_requests "${n_req}" \
    --num_workers "${num_workers}" \
    --drop_parallel_config \
    --gpus "${gpus}" \
    --dist_master_addr 127.0.0.1 \
    --dist_master_port "${BASELINE_DIST_MASTER_PORT:-29621}" \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --base_config_json "${lightx2v_path}/configs/disagg/baseline/wan21_t2v_1p3b_50step_dp.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path "${dp_out}/video.mp4" \
    --save_dir "${dp_out}" \
    --metrics_output_json "${dp_out}/metrics.json" \
    --completion_timeout_s "${CONTROLLER_WAIT_TIMEOUT_S:-14400}" \
    --slo_timeout_s "${SLO_TIMEOUT_S:-14400}"
) > "${dp_out}/baseline.log" 2>&1
dp_rc=$?
echo "===== $(date -Is) END baseline DP rc=${dp_rc} =====" | tee -a "${summary}"

bash "${lightx2v_path}/scripts/disagg/kill_base.sh" || true
sleep 5

########################################
# 2) Disagg 1:6:1 (8 GPUs)
########################################
dg_out=${out_root}/disagg_161
mkdir -p "${dg_out}"
echo "===== $(date -Is) START disagg 1:6:1 =====" | tee -a "${summary}"

export DISAGG_TOPOLOGY=single_node
export DISAGG_CONTROLLER_CFG=${disagg_controller_cfg}
export DISAGG_BASE_CONFIG_JSON=${DISAGG_CONTROLLER_CFG}
export DISAGG_WORKLOAD_STAGES_JSON=${lightx2v_path}/configs/disagg/wan21_t2v_workload_50step.json
export DISAGG_MODEL_PATH=${model_path}
export DISAGG_MODEL_CLS=wan2.1
export DISAGG_TASK=t2v
export LOAD_FROM_USER=0
export ENABLE_MONITOR=0
export DISAGG_DISABLE_AUTOSCALE=1
export DISAGG_AUTO_REQUEST_COUNT=${n_req}
export DISAGG_CONTROLLER_LOG=${dg_out}/controller.log
export DISAGG_CONTROLLER_METRICS_OUTPUT_JSON=${dg_out}/metrics.json
export SAVE_RESULT_PATH=${dg_out}/video.mp4
export CONTROLLER_WAIT_TIMEOUT_S=${CONTROLLER_WAIT_TIMEOUT_S:-14400}
export PROMPT="${prompt}"
export NEGATIVE_PROMPT="${negative_prompt}"
# Do not set CUDA_VISIBLE_DEVICES globally — slots use physical indices.
unset CUDA_VISIBLE_DEVICES || true

bash "${lightx2v_path}/scripts/disagg/run_dynamic.sh" \
  > "${dg_out}/stdout.log" 2>&1 || true
dg_rc=$?
echo "===== $(date -Is) END disagg rc=${dg_rc} =====" | tee -a "${summary}"

bash "${lightx2v_path}/scripts/disagg/kill_service.sh" || true

########################################
# 3) Summarize
########################################
python3 - <<PY | tee -a "${summary}"
import json, statistics as stats
from pathlib import Path

root = Path("${out_root}")
n_req = int("${n_req}")

def summarize_baseline(metrics_path: Path):
    if not metrics_path.is_file():
        return None
    d = json.loads(metrics_path.read_text())
    reqs = d.get("requests") or []
    ok = [r for r in reqs if int(r.get("return_code", 1)) == 0 and r.get("e2e_latency_s") is not None]
    lats = [float(r["e2e_latency_s"]) for r in ok]
    wall = d.get("batch_total_time_s")
    if wall is None and ok:
        # fallback: max finish - min start
        starts = [float(r.get("start_ts") or r.get("client_send_ts") or 0) for r in ok]
        fins = [float(r.get("finish_ts") or 0) for r in ok]
        if starts and fins and min(starts) and max(fins):
            wall = max(fins) - min(starts)
    thr = (len(ok) / wall) if wall and wall > 0 else None
    return {
        "n_ok": len(ok),
        "n_total": len(reqs),
        "mean_lat": stats.mean(lats) if lats else None,
        "p50_lat": sorted(lats)[len(lats)//2] if lats else None,
        "wall_s": wall,
        "throughput": thr,
    }

def summarize_disagg(metrics_path: Path):
    if not metrics_path.is_file():
        return None
    d = json.loads(metrics_path.read_text())
    reqs = d.get("requests") or []
    lats = []
    for r in reqs:
        rm = r.get("request_metrics") or {}
        if rm.get("slo_is_warmup"):
            continue
        lat = (r.get("latency_summary") or {}).get("end_to_end_delay_s")
        if lat is None:
            # decoder_done - client_send
            a = float(rm.get("client_send_ts") or 0)
            done = float(((rm.get("stages") or {}).get("decoder") or {}).get("output_enqueued_ts") or 0)
            if a and done:
                lat = done - a
        if lat is not None:
            lats.append(float(lat))
    wall = d.get("batch_total_time_s")
    if wall is None and lats and reqs:
        # try controller uptime / timestamps
        ts = []
        te = []
        for r in reqs:
            rm = r.get("request_metrics") or {}
            a = rm.get("client_send_ts") or rm.get("controller_send_ts")
            done = ((rm.get("stages") or {}).get("decoder") or {}).get("output_enqueued_ts")
            if a: ts.append(float(a))
            if done: te.append(float(done))
        if ts and te:
            wall = max(te) - min(ts)
    thr = (len(lats) / wall) if wall and wall > 0 and lats else None
    return {
        "n_ok": len(lats),
        "n_total": len(reqs),
        "mean_lat": stats.mean(lats) if lats else None,
        "p50_lat": sorted(lats)[len(lats)//2] if lats else None,
        "wall_s": wall,
        "throughput": thr,
    }

def fmt(x, nd=3):
    if x is None: return "—"
    return f"{x:.{nd}f}"

rows = [
    ("baseline_dp8", summarize_baseline(root / "baseline_dp" / "metrics.json")),
    ("disagg_161", summarize_disagg(root / "disagg_161" / "metrics.json")),
]
print("")
print("## Results")
print("")
print("| scheme | n_ok/n | mean lat (s) | P50 lat (s) | batch wall (s) | throughput (req/s) |")
print("|---|---:|---:|---:|---:|---:|")
for name, s in rows:
    if not s:
        print(f"| {name} | — | — | — | — | — |")
        continue
    print(
        f"| {name} | {s['n_ok']}/{s['n_total']} | {fmt(s['mean_lat'],1)} | "
        f"{fmt(s['p50_lat'],1)} | {fmt(s['wall_s'],1)} | {fmt(s['throughput'],4)} |"
    )
out = {"n_req": n_req, "results": {k: v for k, v in rows}}
(root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("")
print(f"wrote {root}/summary.json")
PY

echo "done out=${out_root}"
