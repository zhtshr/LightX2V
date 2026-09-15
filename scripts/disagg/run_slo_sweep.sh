#!/bin/bash
# Full SLO sweep: 2 schemes × 6 rates (doc defaults for N).
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
stamp=${BENCH_STAMP:-$(date +%Y%m%d_%H%M%S)}
root_out=${BENCH_OUT_DIR:-${lightx2v_path}/save_results/slo_sweep_${stamp}}
mkdir -p "${root_out}"
echo "${root_out}" > /tmp/slo_sweep_out.txt
echo "$$" > /tmp/slo_sweep_pid.txt

# Activate conda like other benches.
if [[ "${CONDA_DEFAULT_ENV:-}" != "lightx2v" ]]; then
    set +u
    # shellcheck disable=SC1091
    source /root/install/miniconda3/etc/profile.d/conda.sh
    conda activate lightx2v
    set -u
fi

rates=(0.010 0.018 0.028 0.045 0.065 0.085)
schemes=(disagfusion lightx2v)

summary="${root_out}/sweep_status.log"
echo "sweep start $(date -Is) out=${root_out}" | tee -a "${summary}"

for scheme in "${schemes[@]}"; do
    for rate in "${rates[@]}"; do
        n=65
        # Doc: low-rate points may use 45; baseline overload points ~30 to confirm collapse.
        awk "BEGIN{exit !(${rate} <= 0.018)}" && n=45
        if [[ "${scheme}" == "lightx2v" ]]; then
            awk "BEGIN{exit !(${rate} >= 0.045)}" && n=30
        fi

        rate_tag=$(printf '%s' "${rate}" | tr '.' 'p')
        point_out=${root_out}/${scheme}_${rate_tag}
        mkdir -p "${point_out}"
        echo "===== $(date -Is) START scheme=${scheme} rate=${rate} n=${n} =====" | tee -a "${summary}"

        set +e
        SCHEME="${scheme}" RATE="${rate}" N="${n}" WARMUP=5 SEED=0 \
            BENCH_OUT_DIR="${point_out}" \
            USER_START_DELAY_S=${USER_START_DELAY_S:-180} \
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            bash "${lightx2v_path}/scripts/disagg/run_slo_bench.sh" \
            > "${point_out}/bench_driver.log" 2>&1
        rc=$?
        set -e

        echo "===== $(date -Is) END scheme=${scheme} rate=${rate} rc=${rc} =====" | tee -a "${summary}"
        ls -la "${point_out}" >> "${summary}" || true

        # Always clean between points.
        bash "${lightx2v_path}/scripts/disagg/kill_service.sh" >/dev/null 2>&1 || true
        sleep 5
    done
done

# Merge CSVs if present.
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

echo "sweep done $(date -Is)" | tee -a "${summary}"
ls -la "${root_out}" | tee -a "${summary}"
