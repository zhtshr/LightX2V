#!/usr/bin/env python3
"""SF (wan2.1_sf) transformer denoise micro-benchmark with Ulysses seq parallel.

Measures KV-cache AR denoise only: all chunks × infer_steps (+ optional rerun),
excluding model load, T5 encoder, and VAE decoder.
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

SAFE_CUDA_DEVICES = {1: "0", 2: "0,2", 4: "0,2,4,5", 6: "0,2,4,5,6,7"}
FORBIDDEN_PHYSICAL_GPUS = {"1", "3"}


def _assert_no_gpu3(cuda_devices: str) -> None:
    ids = {x.strip() for x in cuda_devices.split(",") if x.strip()}
    bad = ids & FORBIDDEN_PHYSICAL_GPUS
    if bad:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cuda_devices!r} includes forbidden physical GPU(s) "
            f"{sorted(bad)}; use 0,2,4,5 for P=4 (skip 1 and 3)."
        )


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


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    pipe_p = int(getattr(args, "pipe_p_size", 1) or 1)
    seq_p = int(args.seq_p_size)
    # CLI parallel sizes override JSON; both 1 → disable distributed.
    if pipe_p > 1 and seq_p > 1:
        config["cpu_offload"] = False
        config["parallel"] = {
            "pipe_p_size": pipe_p,
            "seq_p_size": seq_p,
            "seq_p_attn_type": args.seq_p_attn_type,
        }
    elif pipe_p > 1:
        config["cpu_offload"] = False
        config["parallel"] = {"pipe_p_size": pipe_p}
    elif seq_p > 1:
        config["cpu_offload"] = False
        config["parallel"] = {
            "seq_p_size": seq_p,
            "seq_p_attn_type": args.seq_p_attn_type,
        }
    else:
        config["parallel"] = False
    return config


def _init_distributed(config: dict[str, Any]) -> None:
    if not config.get("parallel"):
        return
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _prepare_inputs_cache(
    config: dict[str, Any],
    cache_path: Path,
    prompt: str,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    if cache_path.is_file() and not force:
        if is_main_process():
            print(f"Loaded encoder inputs cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if dist.is_initialized():
            dist.barrier()
        return payload

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        # Rank0 may still be flushing the file around the barrier; poll briefly.
        for _ in range(60):
            if cache_path.is_file():
                break
            time.sleep(0.5)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Preparing T5 encoder inputs (excluded from transformer bench timing)...")
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder

    latent_shape = _latent_shape_from_config(config)
    text_encoder_output = {"context": context, "context_null": None}
    inputs = {
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": None,
        "latent_shape": latent_shape,
    }
    payload = {
        "seed": seed,
        "latent_shape": latent_shape,
        "inputs": inputs,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"Wrote encoder inputs cache: {cache_path}")

    _sync_device()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return payload


def _run_sf_transformer_compute(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
) -> dict[str, Any]:
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
        "include_rerun": include_rerun,
        "num_output_frames": num_output_frames,
        "latent_shape": latent_shape_adj,
    }


def _bench_once(model: Any, scheduler: Any, config: dict[str, Any], payload: dict[str, Any], include_rerun: bool) -> tuple[float, dict[str, Any]]:
    _sync_device()
    start = time.perf_counter()
    meta = _run_sf_transformer_compute(model, scheduler, config, payload, include_rerun=include_rerun)
    elapsed = time.perf_counter() - start
    return elapsed, meta


def _validate_seq_p(config: dict[str, Any], seq_p_size: int, seq_p_attn_type: str) -> None:
    if seq_p_size <= 1:
        return
    if seq_p_attn_type in ("stripe", "stripe_kv", "ring_kv_cache"):
        return
    num_heads = int(config.get("num_heads", 0))
    if num_heads <= 0:
        return
    if num_heads % seq_p_size != 0:
        raise ValueError(
            f"Ulysses SP requires num_heads ({num_heads}) % seq_p_size ({seq_p_size}) == 0"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="SF transformer SP latency micro-benchmark")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_p_size", type=int, default=1)
    parser.add_argument("--pipe_p_size", type=int, default=1, help="Layer pipeline parallel size (PP)")
    parser.add_argument("--seq_p_attn_type", default="ulysses")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true", help="Skip per-chunk rerun forward pass")
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/sf_phase1_encoder_inputs.pt",
    )
    parser.add_argument("--refresh_inputs_cache", action="store_true")
    parser.add_argument("--output_json", default="")
    parser.add_argument(
        "--cuda_devices",
        default="",
        help="Physical GPU list (e.g. 0,1). Default picks safe set excluding GPU3.",
    )
    parser.add_argument(
        "--encoder_only",
        action="store_true",
        help="Only build encoder input cache; skip transformer benchmark",
    )
    args = parser.parse_args()

    world_need = max(int(args.seq_p_size), 1) * max(int(args.pipe_p_size), 1)
    if args.pipe_p_size > 1 and args.seq_p_size <= 1:
        world_need = int(args.pipe_p_size)
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    elif world_need in SAFE_CUDA_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = SAFE_CUDA_DEVICES[world_need]
    _assert_no_gpu3(os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    config = _load_config(args)
    # Resolve pipe_p from CLI or JSON after _load_config.
    pipe_p_eff = int(args.pipe_p_size)
    if pipe_p_eff <= 1 and isinstance(config.get("parallel"), dict):
        pipe_p_eff = int(config["parallel"].get("pipe_p_size", 1) or 1)
    _validate_seq_p(config, args.seq_p_size, args.seq_p_attn_type)
    seed_all(args.seed)
    _init_distributed(config)

    payload = _prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=args.prompt,
        seed=args.seed,
        force=args.refresh_inputs_cache,
    )
    if args.encoder_only:
        if is_main_process():
            print(f"Encoder-only mode; cache ready at {args.inputs_cache}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    payload = _move_tensor_tree(payload, _run_device())
    include_rerun = not args.no_rerun

    load_start = time.perf_counter()
    model = load_wan_transformer(config)
    scheduler = load_wan_scheduler(config)
    model.set_scheduler(scheduler)
    _sync_device()
    model_load_s = time.perf_counter() - load_start
    if is_main_process():
        print(f"Transformer load time (excluded): {model_load_s:.3f}s")

    def _cuda_mem_mb() -> dict[str, float | None]:
        if not torch.cuda.is_available():
            return {"allocated_mb": None, "reserved_mb": None, "max_allocated_mb": None}
        torch.cuda.synchronize()
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        }

    mem_after_load = _cuda_mem_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_meta: dict[str, Any] = {}
    for _ in range(args.warmup):
        _, run_meta = _bench_once(model, scheduler, config, payload, include_rerun)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    samples: list[float] = []
    for _ in range(args.measure_iters):
        elapsed, run_meta = _bench_once(model, scheduler, config, payload, include_rerun)
        samples.append(elapsed)

    mem_after_measure = _cuda_mem_mb()
    avg_s = sum(samples) / len(samples) if samples else None
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Gather per-rank peak for SP runs.
    peak_local = float(mem_after_measure["max_allocated_mb"] or 0.0)
    peak_all: list[float] | None = None
    if dist.is_initialized() and torch.cuda.is_available():
        t = torch.tensor([peak_local], device=_run_device(), dtype=torch.float32)
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        peak_all = [round(float(x.item()), 1) for x in gathered]

    if pipe_p_eff > 1 and args.seq_p_size > 1:
        parallel_mode = "pipe_p_x_seq_p"
    elif pipe_p_eff > 1:
        parallel_mode = "pipe_p"
    elif args.seq_p_size > 1:
        parallel_mode = "seq_p"
    else:
        parallel_mode = "none"

    result = {
        "metric": "sf_transformer_compute_s",
        "description": "SF chunk AR denoise (KV cache): all chunks × infer_steps"
        + (" + rerun" if include_rerun else ""),
        "model_cls": "wan2.1_sf",
        "parallel_mode": parallel_mode,
        "seq_p_size": int(args.seq_p_size),
        "pipe_p_size": int(pipe_p_eff),
        "seq_p_attn_type": args.seq_p_attn_type if args.seq_p_size > 1 else None,
        "world_size": world_size,
        "num_heads": int(config.get("num_heads", 0)),
        "num_layers": int(config.get("num_layers", 0)),
        "model_load_s_excluded": round(model_load_s, 4),
        "warmup_iters": args.warmup,
        "measure_iters": args.measure_iters,
        "transformer_compute_samples_s": [round(x, 4) for x in samples],
        "transformer_compute_s": round(avg_s, 4) if avg_s is not None else None,
        "num_chunks": run_meta.get("num_chunks"),
        "infer_steps_per_chunk": run_meta.get("infer_steps_per_chunk"),
        "num_output_frames": run_meta.get("num_output_frames"),
        "resolution": f"{config.get('target_width')}x{config.get('target_height')}",
        "target_video_length": config.get("target_video_length"),
        "config_json": args.config_json,
        "dit_original_ckpt": config.get("dit_original_ckpt"),
        "dit_quantized": bool(config.get("dit_quantized", False)),
        "dit_quant_scheme": config.get("dit_quant_scheme"),
        "kv_quant": bool((config.get("ar_config") or {}).get("kv_quant")),
        "kv_offload": bool((config.get("ar_config") or {}).get("kv_offload")),
        "cuda_mem_after_load_mb": mem_after_load,
        "cuda_mem_after_measure_mb": mem_after_measure,
        "peak_allocated_mb_per_rank": peak_all,
        "peak_allocated_mb_max_across_ranks": max(peak_all) if peak_all else mem_after_measure.get("max_allocated_mb"),
    }

    if is_main_process():
        if pipe_p_eff > 1 and args.seq_p_size > 1:
            label = f"pipe_p={pipe_p_eff},seq_p={args.seq_p_size}"
        elif pipe_p_eff > 1:
            label = f"pipe_p={pipe_p_eff}"
        elif args.seq_p_size > 1:
            label = f"seq_p={args.seq_p_size}"
        else:
            label = "P=1"
        print(
            f"{label}: sf_transformer_compute_s={result['transformer_compute_s']}s "
            f"chunks={result['num_chunks']} steps/chunk={result['infer_steps_per_chunk']} "
            f"peak_mb={result.get('peak_allocated_mb_max_across_ranks')} "
            f"per_rank={result.get('peak_allocated_mb_per_rank')} "
            f"samples={result['transformer_compute_samples_s']}"
        )
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
