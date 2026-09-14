#!/bin/bash
# Single-GPU smoke test: Wan SF 14B (Krea Realtime checkpoint) with block offload.
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
model_path="${lightx2v_path}/models/Wan-AI/Wan2.1-T2V-14B"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH:-}"

source "${lightx2v_path}/scripts/base/base.sh"

python "${lightx2v_path}/scripts/self_forcing/benchmark_wan_t2v_sf.py" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/self_forcing/wan_t2v_sf_14b_local.json" \
  --prompt 'A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections.' \
  --save_path "${lightx2v_path}/save_results/output_wan_t2v_sf_14b.mp4" \
  --summary_path "${lightx2v_path}/save_results/wan_t2v_sf_14b_smoke_summary.json"
