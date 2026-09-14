#!/bin/bash
# Three-phase workload (all-4 → mix 1:1 → all-1) under static 1:6:1, 1:5:2, then autoscale.
# Ablation: BENCH_ONLY=as_predict|as_feedback|ablation isolates predict vs feedback modes.
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
stamp=${BENCH_STAMP:-$(date +%Y%m%d_%H%M%S)}
only=${BENCH_ONLY:-all}

# Ablation runs land in a separate results tree by default.
if [[ -z "${BENCH_OUT_DIR:-}" ]]; then
    if [[ "${only}" == "as_predict" || "${only}" == "as_feedback" || "${only}" == "ablation" ]]; then
        out_dir=${lightx2v_path}/save_results/mixed_step_ablation_${stamp}
    else
        out_dir=${lightx2v_path}/save_results/mixed_step_topology_${stamp}
    fi
else
    out_dir=${BENCH_OUT_DIR}
fi
mkdir -p "${out_dir}"

stages_json=${DISAGG_WORKLOAD_STAGES_JSON:-${lightx2v_path}/configs/disagg/wan22_i2v_workload_stages_4_mix_1.json}
user_start_delay_s=${USER_START_DELAY_S:-240}
autoscale_cfg=${lightx2v_path}/configs/disagg/multi_node/wan22_i2v_distill_controller_autoscale.json

run_one() {
    local tag="$1"
    local controller_cfg="$2"
    local disable_autoscale="$3"
    local autoscale_mode="${4:-both}"

    echo "===== RUN ${tag} cfg=${controller_cfg} disable_autoscale=${disable_autoscale} mode=${autoscale_mode} ====="
    export DISAGG_TOPOLOGY=multi_node
    export DISAGG_CONTROLLER_CFG="${controller_cfg}"
    export DISAGG_WORKLOAD_STAGES_JSON="${stages_json}"
    export DISAGG_BASE_CONFIG_JSON="${controller_cfg}"
    export LOAD_FROM_USER=1
    export ENABLE_MONITOR=1
    export DISAGG_DISABLE_AUTOSCALE="${disable_autoscale}"
    export DISAGG_AUTOSCALE_MODE="${autoscale_mode}"
    # Fair vs static 1:6:1 / 1:5:2: never run more than 8 instances.
    export DISAGG_AUTOSCALE_MAX_INSTANCES="${DISAGG_AUTOSCALE_MAX_INSTANCES:-8}"
    export USER_START_DELAY_S="${user_start_delay_s}"
    export USER_MAX_REQUESTS=0
    export DISAGG_CONTROLLER_LOG="${out_dir}/${tag}_controller.log"
    export DISAGG_USER_LOG="${out_dir}/${tag}_user.log"
    export DISAGG_CONTROLLER_METRICS_OUTPUT_JSON="${out_dir}/${tag}_metrics.json"
    export SAVE_RESULT_PATH="${out_dir}/${tag}_video.mp4"
    export CONTROLLER_WAIT_TIMEOUT_S=${CONTROLLER_WAIT_TIMEOUT_S:-7200}

    bash "${lightx2v_path}/scripts/disagg/run_dynamic.sh" \
        > "${out_dir}/${tag}_stdout.log" 2>&1 || {
            echo "WARN: ${tag} exited non-zero (see ${out_dir}/${tag}_stdout.log)"
        }

    if [[ -f "${DISAGG_CONTROLLER_METRICS_OUTPUT_JSON}" ]]; then
        python3 "${lightx2v_path}/scripts/disagg/analyze_mixed_step_topology_metrics.py" \
            "${DISAGG_CONTROLLER_METRICS_OUTPUT_JSON}" \
            --out "${out_dir}/${tag}_summary.json" \
            | tee "${out_dir}/${tag}_summary.txt"
    else
        echo "WARN: missing metrics for ${tag}"
    fi

    bash "${lightx2v_path}/scripts/disagg/kill_service.sh" || true
    sleep 10
}

# Sync controller.py autostart change to remote before runs.
export DISAGG_REMOTE_PRE_CLEAN=${DISAGG_REMOTE_PRE_CLEAN:-1}

if [[ "${only}" == "all" || "${only}" == "161" ]]; then
    # Same default multi_node controller used by run_dynamic.sh (1:6:1 on cuda 0-5).
    run_one "static_161" \
        "${lightx2v_path}/configs/disagg/multi_node/wan22_i2v_distill_controller.json" \
        "1" \
        "both"
fi
if [[ "${only}" == "all" || "${only}" == "152" ]]; then
    run_one "static_152" \
        "${lightx2v_path}/configs/disagg/multi_node/wan22_i2v_distill_controller_152.json" \
        "1" \
        "both"
fi
if [[ "${only}" == "all" || "${only}" == "autoscale" ]]; then
    run_one "autoscale" \
        "${autoscale_cfg}" \
        "0" \
        "both"
fi
if [[ "${only}" == "as_predict" || "${only}" == "ablation" ]]; then
    run_one "as_predict" \
        "${autoscale_cfg}" \
        "0" \
        "predict"
fi
if [[ "${only}" == "as_feedback" || "${only}" == "ablation" ]]; then
    run_one "as_feedback" \
        "${autoscale_cfg}" \
        "0" \
        "feedback"
fi

echo "All done. Results under ${out_dir}"
ls -la "${out_dir}"
echo "${out_dir}" > /tmp/mixed_step_bench_out.txt
