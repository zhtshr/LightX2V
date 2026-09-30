#!/bin/bash
# Download Wan2.1-T2V-14B base (no DIT shards) + Krea Realtime 14B SF checkpoint.
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

LIGHTX2V_PATH="${LIGHTX2V_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
WAN14B_DIR="${LIGHTX2V_PATH}/models/Wan-AI/Wan2.1-T2V-14B"
SF_CKPT_DIR="${LIGHTX2V_PATH}/models/Self-Forcing/checkpoints"

mkdir -p "${WAN14B_DIR}" "${SF_CKPT_DIR}"

echo "=== 1/2 Wan2.1-T2V-14B base (config + T5 + VAE only, no DIT shards) ==="
hf download Wan-AI/Wan2.1-T2V-14B \
  config.json \
  Wan2.1_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json \
  --local-dir "${WAN14B_DIR}"

echo "=== 2/2 Krea Realtime 14B SF checkpoint (~28.6 GB) ==="
hf download krea/krea-realtime-video \
  krea-realtime-video-14b.safetensors \
  --local-dir "${SF_CKPT_DIR}"

echo "=== Done ==="
ls -lh "${SF_CKPT_DIR}/krea-realtime-video-14b.safetensors"
ls -lh "${WAN14B_DIR}/config.json" "${WAN14B_DIR}/Wan2.1_VAE.pth"
