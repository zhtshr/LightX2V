#!/bin/bash
# LightX2V monolithic DP8 SLO sweep (8 independent single-GPU workers).
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
stamp=${BENCH_STAMP:-$(date +%Y%m%d_%H%M%S)_lightx2v_dp}
root_out=${BENCH_OUT_DIR:-${lightx2v_path}/save_results/slo_sweep_${stamp}}
mkdir -p "${root_out}"
echo "${root_out}" > /tmp/slo_sweep_out.txt
echo "$$" > /tmp/slo_sweep_pid.txt

if [[ "${CONDA_DEFAULT_ENV:-}" != "lightx2v" ]]; then
    set +u
    # shellcheck disable=SC1091
    source /root/install/miniconda3/etc/profile.d/conda.sh
    conda activate lightx2v
    set -u
fi

rates=(0.010 0.018 0.028 0.045 0.065 0.085)
summary="${root_out}/sweep_status.log"
echo "lightx2v DP8 rerun start $(date -Is) out=${root_out}" | tee -a "${summary}"

for rate in "${rates[@]}"; do
    n=65
    awk "BEGIN{exit !(${rate} <= 0.018)}" && n=45
    awk "BEGIN{exit !(${rate} >= 0.045)}" && n=30

    rate_tag=$(printf '%s' "${rate}" | tr '.' 'p')
    point_out=${root_out}/lightx2v_dp_${rate_tag}
    rm -rf "${point_out}"
    mkdir -p "${point_out}"
    echo "===== $(date -Is) START scheme=lightx2v_dp rate=${rate} n=${n} =====" | tee -a "${summary}"

    set +e
    SCHEME=lightx2v RATE="${rate}" N="${n}" WARMUP=5 SEED=0 \
        BENCH_OUT_DIR="${point_out}" \
        SLO_SCHEME=lightx2v_dp \
        BASELINE_NUM_WORKERS=8 \
        BASELINE_CONFIG_JSON=${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_baseline_dp8.json \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash "${lightx2v_path}/scripts/disagg/run_slo_bench.sh" \
        > "${point_out}/bench_driver.log" 2>&1
    rc=$?
    set -e

    echo "===== $(date -Is) END scheme=lightx2v_dp rate=${rate} rc=${rc} =====" | tee -a "${summary}"
    ls -la "${point_out}" >> "${summary}" || true
    bash "${lightx2v_path}/scripts/disagg/kill_service.sh" >/dev/null 2>&1 || true
    sleep 5
done

echo "lightx2v DP8 rerun done $(date -Is)" | tee -a "${summary}"
ls -la "${root_out}" | tee -a "${summary}"
