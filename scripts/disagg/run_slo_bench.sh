#!/bin/bash
# One SLO rate-point run: scheme=disagfusion|lightx2v
# Usage:
#   SCHEME=disagfusion RATE=0.01 bash scripts/disagg/run_slo_bench.sh
#   SCHEME=lightx2v RATE=0.01 N=65 bash scripts/disagg/run_slo_bench.sh
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
scheme=${SCHEME:?SCHEME=disagfusion|lightx2v}
rate=${RATE:?RATE=req/s}
n=${N:-65}
warmup=${WARMUP:-5}
seed=${SEED:-0}
timeout_s=${SLO_TIMEOUT_S:-600}
stamp=${BENCH_STAMP:-$(date +%Y%m%d_%H%M%S)}
rate_tag=$(printf '%s' "${rate}" | tr '.' 'p')
out_dir=${BENCH_OUT_DIR:-${lightx2v_path}/save_results/slo_${scheme}_${rate_tag}_${stamp}}
mkdir -p "${out_dir}"

csv_path=${out_dir}/slo_${scheme}_${rate_tag}.csv
echo "SLO run scheme=${scheme} rate=${rate} n=${n} out=${out_dir}"

bash "${lightx2v_path}/scripts/disagg/kill_service.sh" || true
sleep 3

if [[ "${scheme}" == "disagfusion" ]]; then
    export DISAGG_TOPOLOGY=single_node
    # Fixed 1:6:1 (8 GPUs). Autoscale off for SLO: light-load topology
    # thrashing inflated decoder_communication and inverted latency vs λ.
    export DISAGG_CONTROLLER_CFG=${lightx2v_path}/configs/disagg/single_node/wan22_i2v_distill_controller.json
    export DISAGG_BASE_CONFIG_JSON=${DISAGG_CONTROLLER_CFG}
    export DISAGG_WORKLOAD_STAGES_JSON=${lightx2v_path}/configs/disagg/wan22_i2v_workload_slo_4step.json
    export LOAD_FROM_USER=1
    export ENABLE_MONITOR=0
    export DISAGG_DISABLE_AUTOSCALE=1
    export USER_START_DELAY_S=${USER_START_DELAY_S:-180}
    export USER_MAX_REQUESTS=0
    export DISAGG_USER_MODULE=lightx2v.disagg.examples.run_slo_poisson_user
    export PYTHONPATH=${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}
    export SLO_ARRIVAL_RATE=${rate}
    export SLO_NUM_REQUESTS=${n}
    export SLO_WARMUP_REQUESTS=${warmup}
    export SLO_SEED=${seed}
    export SLO_ARRIVAL_LOG=${out_dir}/arrivals.jsonl
    export DISAGG_CONTROLLER_LOG=${out_dir}/controller.log
    export DISAGG_USER_LOG=${out_dir}/user.log
    export DISAGG_CONTROLLER_METRICS_OUTPUT_JSON=${out_dir}/metrics.json
    export SAVE_RESULT_PATH=${out_dir}/video.mp4
    export CONTROLLER_WAIT_TIMEOUT_S=${CONTROLLER_WAIT_TIMEOUT_S:-7200}

    bash "${lightx2v_path}/scripts/disagg/run_dynamic.sh" \
        > "${out_dir}/stdout.log" 2>&1 || true

    python3 "${lightx2v_path}/scripts/disagg/slo_build_csv.py" \
        --arrivals "${out_dir}/arrivals.jsonl" \
        --metrics "${out_dir}/metrics.json" \
        --out "${csv_path}" \
        --scheme disagfusion \
        --arrival_rate "${rate}" \
        --timeout_s "${timeout_s}"

elif [[ "${scheme}" == "lightx2v" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    # Monolithic DP: 8 independent single-GPU workers (not SP8 cooperative).
    export BASELINE_CONFIG_JSON=${BASELINE_CONFIG_JSON:-${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_baseline_dp8.json}
    export BASELINE_LOG=${out_dir}/baseline.log
    export BASELINE_METRICS_OUTPUT_JSON=${out_dir}/metrics.json
    export BASELINE_SAVE_RESULT_PATH=${out_dir}/video.mp4
    export BASELINE_GENERATE_REQUESTS=${n}
    export BASELINE_REQUEST_SOURCE=generate
    export NUM_GPUS=8
    export DISAGG_WORKLOAD_STAGES_JSON=${lightx2v_path}/configs/disagg/wan22_i2v_workload_slo_4step.json
    # base.sh requires these under `set -u`; mirror run_baseline.sh
    export model_path=${BASELINE_MODEL_PATH:-${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models}
    export PYTHONPATH=${PYTHONPATH:-}
    slo_scheme=${SLO_SCHEME:-lightx2v_dp}
    num_workers=${BASELINE_NUM_WORKERS:-8}

    # Inline invoke so we can pass poisson flags (run_baseline.sh does not expose them).
    source "${lightx2v_path}/scripts/base/base.sh"
    python -m lightx2v.disagg.examples.run_controller \
        --mode controller \
        --request_source generate \
        --generate_requests "${n}" \
        --arrival_rate "${rate}" \
        --arrival_seed "${seed}" \
        --slo_csv "${csv_path}" \
        --slo_warmup_requests "${warmup}" \
        --slo_timeout_s "${timeout_s}" \
        --slo_scheme "${slo_scheme}" \
        --completion_timeout_s "${CONTROLLER_WAIT_TIMEOUT_S:-7200}" \
        --num_workers "${num_workers}" \
        --drop_parallel_config \
        --gpus "${CUDA_VISIBLE_DEVICES}" \
        --dist_master_addr 127.0.0.1 \
        --dist_master_port "${BASELINE_DIST_MASTER_PORT:-29611}" \
        --model_cls wan2.2_moe \
        --task i2v \
        --model_path "${model_path}" \
        --base_config_json "${BASELINE_CONFIG_JSON}" \
        --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
        --negative_prompt "镜头晃动，色调艳丽，过曝，静态" \
        --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
        --save_result_path "${BASELINE_SAVE_RESULT_PATH}" \
        --save_dir "${out_dir}" \
        --metrics_output_json "${BASELINE_METRICS_OUTPUT_JSON}" \
        > "${BASELINE_LOG}" 2>&1
else
    echo "unknown SCHEME=${scheme}" >&2
    exit 2
fi

bash "${lightx2v_path}/scripts/disagg/kill_service.sh" || true
echo "done: ${csv_path}"
ls -la "${out_dir}"
