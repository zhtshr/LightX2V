#!/usr/bin/env bash
# Krea SF 14B BF16, batch=1, full AR denoise throughput (excludes T5/VAE/load).
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
python_bin="${BENCH_PYTHON:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-$repo/models/Wan-AI/Wan2.1-T2V-14B}"
sf_ckpt="${SF_CKPT:-$repo/models/Self-Forcing/checkpoints/krea-realtime-video-14b.safetensors}"
out="${BENCH_OUTPUT:-$repo/save_results/sf_14b_480p_scaling}"
mkdir -p "$out"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PROFILING_DEBUG_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$python_bin" - "$repo" "$out" "$sf_ckpt" <<'PYCONFIG'
import json, sys
from pathlib import Path
repo, out, ckpt = map(Path, sys.argv[1:])
if not ckpt.is_file():
    raise SystemExit(f"Missing SF checkpoint: {ckpt}")
cfg = json.loads((repo / 'configs/self_forcing/wan_t2v_sf_14b_local.json').read_text())
cfg.update(target_height=480, target_width=832, target_video_length=81,
           cpu_offload=False, dit_original_ckpt=str(ckpt.resolve()),
           self_attn_1_type='torch_sdpa', cross_attn_1_type='torch_sdpa',
           cross_attn_2_type='torch_sdpa')
(Path(out) / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
PYCONFIG
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > "$out/hardware.csv"
git rev-parse HEAD > "$out/commit.txt"
"$python_bin" -m pip freeze > "$out/packages.txt"
for p in 1 2 4 8; do
    devices="$(seq -s, 0 $((p - 1)))"
    launch=("$python_bin")
    if ((p > 1)); then
        launch+=(-m torch.distributed.run --standalone "--nproc_per_node=$p")
    fi
    CUDA_VISIBLE_DEVICES="$devices" "${launch[@]}" scripts/disagg/run_sf_transformer_sp_bench.py \
      --model_path "$model_path" --config_json "$out/config.json" \
      --seq_p_size "$p" --seq_p_attn_type ulysses \
      --cuda_devices "$devices" --allow_all_gpus \
      --warmup "${WARMUP:-1}" --measure_iters "${MEASURE_ITERS:-3}" \
      --inputs_cache "$out/encoder_inputs.pt" \
      --output_json "$out/p$p.json" 2>&1 | tee "$out/p$p.log"
done
"$python_bin" - "$out" <<'PYSUMMARY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
base = None
for p in [1, 2, 4, 8]:
    d = json.loads((out / f'p{p}.json').read_text())
    latency = d['transformer_compute_s']
    frames = d['output_video_frames']
    if p == 1:
        base = latency
    rows.append(dict(gpus=p, seconds=latency, frames=frames, fps=frames/latency,
                     videos_per_minute=60/latency, speedup=base/latency,
                     efficiency=base/latency/p,
                     peak_gib=d['peak_allocated_mb_max_across_ranks']/1024))
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
lines = ['Krea SF 14B BF16, 832x480, batch=1, Ulysses SP, torch_sdpa attention.',
         'Full autoregressive denoise including KV reruns; excludes loading, T5 and VAE.',
         '', '| GPUs | seconds | frames | FPS | videos/min | speedup | efficiency | peak GiB |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['gpus']} | {r['seconds']:.3f} | {r['frames']} | {r['fps']:.2f} | {r['videos_per_minute']:.2f} | {r['speedup']:.2f} | {r['efficiency']:.1%} | {r['peak_gib']:.2f} |")
text = '\n'.join(lines) + '\n'
(out / 'summary.md').write_text(text)
print(text)
PYSUMMARY
