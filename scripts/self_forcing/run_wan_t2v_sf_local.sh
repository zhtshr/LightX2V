#!/bin/bash
set -euo pipefail

lightx2v_path=/root/zht/LightX2V
model_path=/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B
export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls wan2.1_sf \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/self_forcing/wan_t2v_sf_local.json" \
  --prompt 'A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections.' \
  --save_result_path "${lightx2v_path}/save_results/output_wan_t2v_sf.mp4"
