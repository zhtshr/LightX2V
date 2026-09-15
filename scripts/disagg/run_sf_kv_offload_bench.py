#!/usr/bin/env python3
"""Compare SF transformer latency with kv_offload on vs off."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.utils.utils import seed_all

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)


def _latent_shape_from_config(config: dict[str, Any]) -> list[int]:
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def _move_tensor_tree(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_tensor_tree(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensor_tree(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(v, device) for v in obj)
    return obj


def _prepare_inputs(config: dict[str, Any], prompt: str, cache_path: Path) -> dict[str, Any]:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        payload["latent_shape"] = _latent_shape_from_config(config)
        return payload
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    latent_shape = _latent_shape_from_config(config)
    payload = {
        "seed": 42,
        "latent_shape": latent_shape,
        "inputs": {
            "text_encoder_output": {"context": context, "context_null": None},
            "image_encoder_output": None,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _pool_tensor_bytes(pool: Any) -> dict[str, float]:
    gpu_b = cpu_b = 0
    for name in vars(pool):
        if not name.startswith("_"):
            continue
        val = getattr(pool, name, None)
        if not torch.is_tensor(val):
            continue
        nbytes = int(val.nbytes)
        if val.device.type == "cuda":
            gpu_b += nbytes
        elif val.device.type == "cpu":
            cpu_b += nbytes
    return {
        "gpu_mb": round(gpu_b / (1024**2), 3),
        "cpu_mb": round(cpu_b / (1024**2), 3),
    }


def _measure_kv_memory(model: Any, config: dict[str, Any], latent_shape: list[int]) -> dict[str, Any]:
    torch.cuda.synchronize()
    mem_before = int(torch.cuda.memory_allocated())
    DisaggSFKVCacheManager.setup(model, config, latent_shape)
    torch.cuda.synchronize()
    mem_after = int(torch.cuda.memory_allocated())
    kv = model.kv_cache_manager
    sa = _pool_tensor_bytes(kv.self_attn_kv_cache)
    ca = _pool_tensor_bytes(kv.cross_attn_kv_cache)
    out = {
        "self_attn_kv_gpu_mb": sa["gpu_mb"],
        "self_attn_kv_cpu_mb": sa["cpu_mb"],
        "cross_attn_kv_gpu_mb": ca["gpu_mb"],
        "cuda_delta_mb": round((mem_after - mem_before) / (1024**2), 3),
        "cuda_after_kv_mb": round(mem_after / (1024**2), 3),
    }
    DisaggSFKVCacheManager.teardown(model)
    return out


def _run_denoise(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
    collect_profile: bool,
) -> tuple[float, dict[str, Any]]:
    inputs = payload["inputs"]
    seed = int(payload["seed"])
    latent_shape = list(payload["latent_shape"])

    ti = model.transformer_infer
    if collect_profile and hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
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

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated())

    profile = None
    if collect_profile and hasattr(ti, "finalize_kv_store_profile"):
        profile = ti.finalize_kv_store_profile()

    meta = {
        "transformer_s": round(elapsed, 4),
        "cuda_peak_mb": round(peak / (1024**2), 3),
        "num_chunks": num_chunks,
        "num_forwards": num_chunks * infer_steps + (num_chunks if include_rerun else 0),
        "kv_profile": profile,
    }
    return elapsed, meta


def _apply_kv_offload(config: dict[str, Any], kv_offload: bool, profile: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    ar = dict(cfg.get("ar_config", {}))
    ar["kv_offload"] = kv_offload
    ar["profile_kv_store"] = profile
    cfg["ar_config"] = ar
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="SF kv_offload latency comparison")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/disagg/wan/wan_t2v_sf_disagg_transformer.json")
    parser.add_argument("--target_height", type=int, default=720)
    parser.add_argument("--target_width", type=int, default=1280)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_phase1_encoder_inputs.pt")
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_kv_offload_720p.json")
    parser.add_argument("--output_md", default="save_results/optimization_study/sf_kv_offload_720p.md")
    args = parser.parse_args()

    base_config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    base_config["target_height"] = args.target_height
    base_config["target_width"] = args.target_width
    base_config["parallel"] = False

    payload = _prepare_inputs(base_config, args.prompt, Path(args.inputs_cache))
    device = torch.device("cuda")
    payload = _move_tensor_tree(payload, device)
    include_rerun = not args.no_rerun
    latent_shape = list(payload["latent_shape"])

    results: dict[str, Any] = {
        "resolution": f"{args.target_height}x{args.target_width}",
        "include_rerun": include_rerun,
        "modes": {},
    }

    for mode, kv_offload in [("gpu_kv", False), ("kv_offload", True)]:
        cfg = _apply_kv_offload(base_config, kv_offload, profile=True)
        seed_all(args.seed)
        model = load_wan_transformer(cfg)
        scheduler = load_wan_scheduler(cfg)
        model.set_scheduler(scheduler)

        memory = _measure_kv_memory(model, cfg, latent_shape)

        for _ in range(args.warmup):
            _run_denoise(model, scheduler, cfg, payload, include_rerun=include_rerun, collect_profile=False)

        samples: list[float] = []
        profiles: list[dict[str, Any]] = []
        for _ in range(args.measure_iters):
            elapsed, meta = _run_denoise(
                model, scheduler, cfg, payload, include_rerun=include_rerun, collect_profile=True,
            )
            samples.append(elapsed)
            if meta.get("kv_profile"):
                profiles.append(meta["kv_profile"])

        avg_s = sum(samples) / len(samples)
        prof = profiles[-1] if profiles else {}
        transformer_ms = avg_s * 1000.0
        self_attn_ms = float(prof.get("self_attn_ms", 0.0))
        store_kv_ms = float(prof.get("store_kv_ms", 0.0))
        kv_read_ms = float(prof.get("kv_read_ms", 0.0))

        results["modes"][mode] = {
            "kv_offload": kv_offload,
            "transformer_s": round(avg_s, 4),
            "transformer_ms": round(transformer_ms, 2),
            "cuda_peak_mb": meta["cuda_peak_mb"],
            "memory": memory,
            "kv_profile": prof,
            "self_attn_ms": round(self_attn_ms, 3),
            "store_kv_ms": round(store_kv_ms, 3),
            "kv_read_ms": round(kv_read_ms, 3),
            "self_attn_share_pct": round(100.0 * self_attn_ms / transformer_ms, 2) if transformer_ms > 0 else None,
        }
        del model
        torch.cuda.empty_cache()

    gpu = results["modes"]["gpu_kv"]
    off = results["modes"]["kv_offload"]
    slowdown = off["transformer_s"] / gpu["transformer_s"] if gpu["transformer_s"] > 0 else None
    delta_s = off["transformer_s"] - gpu["transformer_s"]
    delta_ms = delta_s * 1000.0
    results["comparison"] = {
        "slowdown_x": round(slowdown, 4) if slowdown is not None else None,
        "delta_s": round(delta_s, 4),
        "delta_ms": round(delta_ms, 2),
        "delta_pct": round(100.0 * delta_s / gpu["transformer_s"], 2) if gpu["transformer_s"] > 0 else None,
        "gpu_kv_peak_mb": gpu["cuda_peak_mb"],
        "kv_offload_peak_mb": off["cuda_peak_mb"],
        "peak_saved_mb": round(gpu["cuda_peak_mb"] - off["cuda_peak_mb"], 3),
        "self_attn_delta_ms": round(off["self_attn_ms"] - gpu["self_attn_ms"], 3),
        "kv_read_delta_ms": round(off["kv_read_ms"] - gpu["kv_read_ms"], 3),
        "store_kv_delta_ms": round(off["store_kv_ms"] - gpu["store_kv_ms"], 3),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    cmp_ = results["comparison"]
    md = "\n".join([
        f"# SF kv_offload @ {results['resolution']}",
        "",
        f"- Config: `{args.config_json}`, rerun={'on' if include_rerun else 'off'}",
        "",
        "## Latency",
        "",
        "| Mode | Transformer (s) | Self-attn (ms) | kv_read (ms) | store_kv (ms) | Peak GPU (MB) |",
        "|------|-----------------|----------------|--------------|---------------|---------------|",
        f"| GPU KV (`kv_offload=false`) | {gpu['transformer_s']} | {gpu['self_attn_ms']} | {gpu['kv_read_ms']} | {gpu['store_kv_ms']} | {gpu['cuda_peak_mb']} |",
        f"| CPU offload (`kv_offload=true`) | {off['transformer_s']} | {off['self_attn_ms']} | {off['kv_read_ms']} | {off['store_kv_ms']} | {off['cuda_peak_mb']} |",
        "",
        f"- **Slowdown: {cmp_['slowdown_x']}× (+{cmp_['delta_s']} s / +{cmp_['delta_ms']} ms, +{cmp_['delta_pct']}%)**",
        f"- Peak GPU saved: **{cmp_['peak_saved_mb']} MB**",
        f"- Self-attn KV GPU: {gpu['memory']['self_attn_kv_gpu_mb']} MB → {off['memory']['self_attn_kv_gpu_mb']} MB",
        f"- Self-attn KV CPU: {gpu['memory']['self_attn_kv_cpu_mb']} MB → {off['memory']['self_attn_kv_cpu_mb']} MB",
        "",
    ])
    Path(args.output_md).write_text(md, encoding="utf-8")

    print(
        f"720p kv_offload: gpu_kv={gpu['transformer_s']}s offload={off['transformer_s']}s "
        f"slowdown={cmp_['slowdown_x']}x (+{cmp_['delta_pct']}%) peak_saved={cmp_['peak_saved_mb']}MB"
    )
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
