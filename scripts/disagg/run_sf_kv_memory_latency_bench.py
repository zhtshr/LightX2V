#!/usr/bin/env python3
"""Benchmark SF Self-attn KV vs Cross-attn KV GPU memory and compute latency."""

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
        return torch.load(cache_path, map_location="cpu", weights_only=False)
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
    """Sum tensor nbytes on a KV pool, split by device."""
    gpu_b = 0
    cpu_b = 0
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
        "gpu_bytes": gpu_b,
        "cpu_bytes": cpu_b,
        "gpu_mb": round(gpu_b / (1024**2), 3),
        "cpu_mb": round(cpu_b / (1024**2), 3),
        "total_mb": round((gpu_b + cpu_b) / (1024**2), 3),
    }


def _kv_cache_layout(kv_manager: Any) -> dict[str, Any]:
    sa = kv_manager.self_attn_kv_cache
    ca = kv_manager.cross_attn_kv_cache
    sa_k = sa._k_buffer if hasattr(sa, "_k_buffer") else None
    ca_k = ca._k_buffer if hasattr(ca, "_k_buffer") else None
    layout: dict[str, Any] = {
        "frame_seq_length": int(kv_manager.frame_seq_length),
        "num_output_frames": int(kv_manager.num_output_frames),
        "kv_cache_size_tokens": int(kv_manager.kv_size),
        "text_len": int(kv_manager.config.get("text_len", 512)),
        "num_layers": int(kv_manager.config["num_layers"]),
        "num_heads": int(kv_manager.config["num_heads"]),
        "cache_num_heads": int(getattr(kv_manager, "cache_num_heads", kv_manager.config["num_heads"])),
        "head_dim": int(kv_manager.config["dim"] // kv_manager.config["num_heads"]),
        "dtype": str(kv_manager.dtype).replace("torch.", ""),
        "kv_offload": bool(kv_manager.ar_config.get("kv_offload", False)),
        "sp_head_sharded_kv": bool(getattr(kv_manager, "sp_head_sharded_kv", False)),
    }
    if sa_k is not None:
        layout["self_attn_buffer_shape"] = list(sa_k.shape)
    if ca_k is not None:
        layout["cross_attn_buffer_shape"] = list(ca_k.shape)
    return layout


def _measure_kv_memory(model: Any, config: dict[str, Any], latent_shape: list[int]) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before_kv = int(torch.cuda.memory_allocated())

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    torch.cuda.synchronize()
    mem_after_kv = int(torch.cuda.memory_allocated())
    peak_after_kv = int(torch.cuda.max_memory_allocated())

    kv_manager = model.kv_cache_manager
    sa_mem = _pool_tensor_bytes(kv_manager.self_attn_kv_cache)
    ca_mem = _pool_tensor_bytes(kv_manager.cross_attn_kv_cache)
    layout = _kv_cache_layout(kv_manager)

    result = {
        "layout": layout,
        "num_chunks": num_chunks,
        "num_output_frames": num_output_frames,
        "latent_shape_adjusted": list(latent_shape_adj),
        "self_attn_kv": sa_mem,
        "cross_attn_kv": ca_mem,
        "kv_total_gpu_mb": round(sa_mem["gpu_mb"] + ca_mem["gpu_mb"], 3),
        "kv_total_cpu_mb": round(sa_mem["cpu_mb"] + ca_mem["cpu_mb"], 3),
        "cuda_allocated_before_kv_mb": round(mem_before_kv / (1024**2), 3),
        "cuda_allocated_after_kv_mb": round(mem_after_kv / (1024**2), 3),
        "cuda_kv_delta_mb": round((mem_after_kv - mem_before_kv) / (1024**2), 3),
        "cuda_peak_after_kv_setup_mb": round(peak_after_kv / (1024**2), 3),
    }
    DisaggSFKVCacheManager.teardown(model)
    return result


def _run_sf_denoise_profiled(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
) -> tuple[float, dict[str, Any]]:
    inputs = payload["inputs"]
    seed = int(payload["seed"])
    latent_shape = list(payload["latent_shape"])

    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    num_forwards = num_chunks * infer_steps + (num_chunks if include_rerun else 0)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
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
    peak_during_run = int(torch.cuda.max_memory_allocated())

    profile = None
    if hasattr(ti, "finalize_kv_store_profile"):
        profile = ti.finalize_kv_store_profile() or {}

    transformer_ms = elapsed * 1000.0
    self_attn_ms = float(profile.get("self_attn_ms", 0.0))
    cross_attn_ms = float(profile.get("cross_attn_ms", 0.0))
    store_kv_ms = float(profile.get("store_kv_ms", 0.0))
    cross_kv_store_ms = float(profile.get("cross_kv_store_ms", 0.0))
    kv_read_ms = float(profile.get("kv_read_ms", 0.0))
    sp_a2a_ms = float(profile.get("sp_a2a_ms", 0.0))

    meta = {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "include_rerun": include_rerun,
        "num_forwards": num_forwards,
        "transformer_s": round(elapsed, 4),
        "transformer_ms": round(transformer_ms, 2),
        "cuda_peak_during_run_mb": round(peak_during_run / (1024**2), 3),
        "kv_profile": profile,
        "latency": {
            "self_attn_total_ms": round(self_attn_ms, 3),
            "cross_attn_total_ms": round(cross_attn_ms, 3),
            "store_kv_total_ms": round(store_kv_ms, 3),
            "cross_kv_store_total_ms": round(cross_kv_store_ms, 3),
            "kv_read_total_ms": round(kv_read_ms, 3),
            "sp_a2a_total_ms": round(sp_a2a_ms, 3),
            "self_attn_per_forward_ms": round(self_attn_ms / num_forwards, 4) if num_forwards else None,
            "cross_attn_per_forward_ms": round(cross_attn_ms / num_forwards, 4) if num_forwards else None,
            "self_attn_share_pct": round(100.0 * self_attn_ms / transformer_ms, 2) if transformer_ms > 0 else None,
            "cross_attn_share_pct": round(100.0 * cross_attn_ms / transformer_ms, 2) if transformer_ms > 0 else None,
            "store_kv_share_of_self_attn_pct": round(100.0 * store_kv_ms / self_attn_ms, 2) if self_attn_ms > 0 else None,
            "cross_kv_store_share_of_cross_attn_pct": round(100.0 * cross_kv_store_ms / cross_attn_ms, 2)
            if cross_attn_ms > 0
            else None,
        },
    }
    return elapsed, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="SF Self-attn vs Cross-attn KV memory + latency bench")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1, help="Ulysses SP world size (1 = no SP)")
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/sf_phase1_encoder_inputs.pt",
    )
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_kv_memory_latency.json")
    parser.add_argument("--output_md", default="save_results/optimization_study/sf_kv_memory_latency.md")
    args = parser.parse_args()

    base_config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    base_config["cpu_offload"] = False
    base_config["parallel"] = args.sp_size > 1
    if args.sp_size > 1:
        base_config["parallel"] = True
        base_config["seq_parallel"] = True
        base_config["ulysses_degree"] = args.sp_size

    ar = dict(base_config.get("ar_config", {}))
    ar["profile_kv_store"] = True
    base_config["ar_config"] = ar

    payload = _prepare_inputs(base_config, args.prompt, Path(args.inputs_cache))
    device = torch.device("cuda")
    payload = _move_tensor_tree(payload, device)
    include_rerun = not args.no_rerun
    latent_shape = list(payload["latent_shape"])

    seed_all(args.seed)
    model = load_wan_transformer(base_config)
    scheduler = load_wan_scheduler(base_config)
    model.set_scheduler(scheduler)

    memory = _measure_kv_memory(model, base_config, latent_shape)

    for _ in range(args.warmup):
        _run_sf_denoise_profiled(model, scheduler, base_config, payload, include_rerun=include_rerun)

    samples: list[float] = []
    run_metas: list[dict[str, Any]] = []
    for _ in range(args.measure_iters):
        elapsed, meta = _run_sf_denoise_profiled(
            model, scheduler, base_config, payload, include_rerun=include_rerun,
        )
        samples.append(elapsed)
        run_metas.append(meta)

    avg_s = sum(samples) / len(samples)
    last = run_metas[-1]
    lat = last["latency"]

    results: dict[str, Any] = {
        "model_cls": "wan2.1_sf",
        "sp_size": args.sp_size,
        "include_rerun": include_rerun,
        "memory": memory,
        "latency": {
            **lat,
            "transformer_s": round(avg_s, 4),
            "transformer_ms": round(avg_s * 1000.0, 2),
            "cuda_peak_during_run_mb": last["cuda_peak_during_run_mb"],
            "num_forwards": last["num_forwards"],
            "num_chunks": last["num_chunks"],
            "infer_steps_per_chunk": last["infer_steps_per_chunk"],
        },
        "kv_profile_raw": last.get("kv_profile"),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    mem = memory
    sa = mem["self_attn_kv"]
    ca = mem["cross_attn_kv"]
    lay = mem["layout"]
    md_lines = [
        "# SF Self-attn vs Cross-attn KV: Memory & Latency",
        "",
        f"- Config: `{args.config_json}`, SP={args.sp_size}, rerun={'on' if include_rerun else 'off'}",
        f"- Resolution: {base_config['target_height']}x{base_config['target_width']}, "
        f"{base_config['target_video_length']} frames",
        "",
        "## KV cache layout",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| frame_seq_length | {lay['frame_seq_length']} |",
        f"| num_output_frames | {lay['num_output_frames']} |",
        f"| self-attn tokens (kv_size) | {lay['kv_cache_size_tokens']} |",
        f"| cross-attn tokens (text_len) | {lay['text_len']} |",
        f"| layers | {lay['num_layers']} |",
        f"| heads (cache) | {lay['cache_num_heads']} |",
        f"| head_dim | {lay['head_dim']} |",
        f"| dtype | {lay['dtype']} |",
        "",
        "## GPU memory (KV buffers)",
        "",
        "| Pool | GPU (MB) | CPU (MB) | Notes |",
        "|------|----------|----------|-------|",
        f"| Self-attn KV | {sa['gpu_mb']} | {sa['cpu_mb']} | K+V rolling cache |",
        f"| Cross-attn KV | {ca['gpu_mb']} | {ca['cpu_mb']} | K+V text context |",
        f"| **KV total** | **{mem['kv_total_gpu_mb']}** | {mem['kv_total_cpu_mb']} | buffer nbytes sum |",
        "",
        f"- CUDA allocated delta after KV setup: **{mem['cuda_kv_delta_mb']} MB**",
        f"- CUDA peak during full denoise: **{results['latency']['cuda_peak_during_run_mb']} MB**",
        "",
        "## Compute latency (full run)",
        "",
        f"- Transformer total: **{results['latency']['transformer_s']} s** ({last['num_forwards']} forwards)",
        "",
        "| Path | Total (ms) | Per-forward (ms) | Share of transformer |",
        "|------|------------|------------------|----------------------|",
        f"| Self-attn (w/ KV) | {lat['self_attn_total_ms']} | {lat['self_attn_per_forward_ms']} | {lat['self_attn_share_pct']}% |",
        f"| Cross-attn (w/ KV) | {lat['cross_attn_total_ms']} | {lat['cross_attn_per_forward_ms']} | {lat['cross_attn_share_pct']}% |",
        f"| └ store_kv (self) | {lat['store_kv_total_ms']} | — | {lat['store_kv_share_of_self_attn_pct']}% of self-attn |",
        f"| └ store_kv (cross) | {lat['cross_kv_store_total_ms']} | — | {lat['cross_kv_store_share_of_cross_attn_pct']}% of cross-attn |",
        f"| └ kv_read (self hist) | {lat['kv_read_total_ms']} | — | — |",
        f"| └ SP all2all | {lat['sp_a2a_total_ms']} | — | — |",
        "",
    ]
    out_md = Path(args.output_md)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Self-attn KV: {sa['gpu_mb']} MB GPU | Cross-attn KV: {ca['gpu_mb']} MB GPU")
    print(
        f"Latency: transformer={results['latency']['transformer_s']}s "
        f"self_attn={lat['self_attn_total_ms']}ms ({lat['self_attn_share_pct']}%) "
        f"cross_attn={lat['cross_attn_total_ms']}ms ({lat['cross_attn_share_pct']}%)"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
