#!/bin/bash
# Re-run LightX2V baseline SLO points only (after DisagFusion already succeeded).
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
root_out=${BENCH_OUT_DIR:-${lightx2v_path}/save_results/slo_sweep_20260730_145104_noauto}
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
summary="${root_out}/sweep_status_lightx2v_rerun.log"
echo "lightx2v rerun start $(date -Is) out=${root_out}" | tee -a "${summary}"

for rate in "${rates[@]}"; do
    n=65
    awk "BEGIN{exit !(${rate} <= 0.018)}" && n=45
    awk "BEGIN{exit !(${rate} >= 0.045)}" && n=30

    rate_tag=$(printf '%s' "${rate}" | tr '.' 'p')
    point_out=${root_out}/lightx2v_${rate_tag}
    # Fresh dir for this point
    rm -rf "${point_out}"
    mkdir -p "${point_out}"
    echo "===== $(date -Is) START scheme=lightx2v rate=${rate} n=${n} =====" | tee -a "${summary}"

    set +e
    SCHEME=lightx2v RATE="${rate}" N="${n}" WARMUP=5 SEED=0 \
        BENCH_OUT_DIR="${point_out}" \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash "${lightx2v_path}/scripts/disagg/run_slo_bench.sh" \
        > "${point_out}/bench_driver.log" 2>&1
    rc=$?
    set -e

    echo "===== $(date -Is) END scheme=lightx2v rate=${rate} rc=${rc} =====" | tee -a "${summary}"
    ls -la "${point_out}" >> "${summary}" || true
    bash "${lightx2v_path}/scripts/disagg/kill_service.sh" >/dev/null 2>&1 || true
    sleep 5
done

python3 - <<'PY' "${root_out}" || true
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
header = None
for p in sorted(root.glob("*/slo_*.csv")):
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        continue
    if header is None:
        header = lines[0]
    rows.extend(lines[1:])
if header is not None:
    out = root / "slo_results.csv"
    out.write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print("merged", out, "rows", len(rows))
PY

echo "lightx2v rerun done $(date -Is)" | tee -a "${summary}"
ls -la "${root_out}" | tee -a "${summary}"
