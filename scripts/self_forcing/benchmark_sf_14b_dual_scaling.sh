#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
python_bin="${BENCH_PYTHON:-$repo/.venv/bin/python}"
out="$repo/save_results/sf_14b_480p_dual_overlap"
mkdir -p "$out"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PROFILING_DEBUG_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > "$out/hardware.csv"
git rev-parse HEAD > "$out/commit.txt"
"$python_bin" -m pip freeze > "$out/packages.txt"
cp save_results/sf_14b_480p_scaling/config.json "$out/config.json"
for p in ${SP_LIST:-1 2 4 8}; do
    launch=("$python_bin")
    if ((p > 1)); then launch+=(-m torch.distributed.run --standalone "--nproc_per_node=$p"); fi
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((p - 1)))" "${launch[@]}" \
      scripts/self_forcing/benchmark_sf_14b_dual.py --seq_p_size "$p" \
      --warmup "${WARMUP:-1}" --measure_iters "${MEASURE_ITERS:-3}" \
      --output_dir "$out" > "$out/p$p.log" 2>&1
    echo "Finished SP=$p"
done
"$python_bin" scripts/self_forcing/summarize_sf_14b_dual.py "$out"
