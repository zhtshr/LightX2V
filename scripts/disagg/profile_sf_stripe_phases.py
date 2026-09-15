#!/usr/bin/env python3
"""Profile stripe self-attn phase breakdown for SF transformer SP (P=2 default).

Phases timed inside ``sp_kvcache_attn_stripe``:
  all_gather_q / flash / all_gather_out / all_gather_lse / merge / slice
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.kvcache.base import enable_stripe_attn_profiler, get_stripe_attn_profiler
from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)

# GPU3 (pci 0000:72:00.0) is excluded: repeated Xid-79 bus falloff.
SAFE_CUDA_DEVICES_P2 = "0,2"
SAFE_CUDA_DEVICES_P4 = "0,2,4,5"


def _run_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _move_tensor_tree(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_tensor_tree(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tensor_tree(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(value, device) for value in obj)
    return obj


def _latent_shape_from_config(config: dict[str, Any]) -> list[int]:
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def _prepare_inputs_cache(config: dict[str, Any], cache_path: Path, prompt: str, seed: int) -> dict[str, Any]:
    if cache_path.is_file():
        if is_main_process():
            print(f"Loaded encoder inputs cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if dist.is_initialized():
            dist.barrier()
        return payload

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Preparing T5 encoder inputs...")
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    latent_shape = _latent_shape_from_config(config)
    payload = {
        "seed": seed,
        "latent_shape": latent_shape,
        "inputs": {
            "text_encoder_output": {"context": context, "context_null": None},
            "image_encoder_output": None,
            "latent_shape": latent_shape,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    if dist.is_initialized():
        dist.barrier()
    return payload


def _run_once(model, scheduler, config, payload, include_rerun: bool) -> dict[str, Any]:
    inputs = payload["inputs"]
    latent_shape = list(payload["latent_shape"])
    seed = int(payload["seed"])
    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    try:
        for seg_idx in range(num_chunks):
            for step_index in range(infer_steps):
                model.kv_cache_manager.current_step = step_index
                scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
                model.infer(inputs)
                scheduler.step_post()
            if include_rerun:
                scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
                model.infer(inputs)
    finally:
        DisaggSFKVCacheManager.teardown(model)
    _sync_device()
    return {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "num_output_frames": num_output_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_p_size", type=int, default=2)
    parser.add_argument("--seq_p_attn_type", default="stripe")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_phase1_encoder_inputs.pt")
    parser.add_argument(
        "--cuda_devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", SAFE_CUDA_DEVICES_P2),
        help=f"Physical GPU indices (default {SAFE_CUDA_DEVICES_P2}; never use GPU1/GPU3)",
    )
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_stripe_p2_phase_profile.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["LIGHTX2V_STRIPE_PROFILE"] = "1"

    config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    if args.seq_p_size > 1:
        config["cpu_offload"] = False
        config["parallel"] = {
            "seq_p_size": int(args.seq_p_size),
            "seq_p_attn_type": args.seq_p_attn_type,
        }
    else:
        config["parallel"] = False

    seed_all(args.seed)
    if config.get("parallel"):
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        if platform_device is None:
            raise RuntimeError("platform device registry is unavailable")
        platform_device.init_parallel_env()
        set_parallel_config(config)

    payload = _prepare_inputs_cache(config, Path(args.inputs_cache), args.prompt, args.seed)
    payload = _move_tensor_tree(payload, _run_device())
    include_rerun = not args.no_rerun

    model = load_wan_transformer(config)
    scheduler = load_wan_scheduler(config)
    model.set_scheduler(scheduler)

    for _ in range(args.warmup):
        enable_stripe_attn_profiler(True)
        _run_once(model, scheduler, config, payload, include_rerun)

    enable_stripe_attn_profiler(True)
    _sync_device()
    t0 = time.perf_counter()
    meta = None
    for _ in range(args.measure_iters):
        meta = _run_once(model, scheduler, config, payload, include_rerun)
    wall_s = (time.perf_counter() - t0) / max(args.measure_iters, 1)

    prof = get_stripe_attn_profiler()
    stripe = prof.snapshot() if prof is not None else {}

    result = {
        "metric": "sf_stripe_phase_profile",
        "seq_p_size": args.seq_p_size,
        "seq_p_attn_type": args.seq_p_attn_type,
        "transformer_compute_s": round(wall_s, 4),
        "num_chunks": meta.get("num_chunks") if meta else None,
        "infer_steps_per_chunk": meta.get("infer_steps_per_chunk") if meta else None,
        "stripe_phase": stripe,
        "note": (
            "stripe_phase times are summed over all self-attn calls in the measured run "
            "(layers × chunks × steps(+rerun)). wall_s is end-to-end transformer compute."
        ),
    }

    if is_main_process():
        print(json.dumps(result, indent=2))
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
