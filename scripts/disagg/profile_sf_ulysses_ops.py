#!/usr/bin/env python3
"""Profile Ulysses SF self-attn + whole-forward CUDA kernels (P>=2).

Reports:
  1) UlyssesAttnProfiler phases: all2all_q / flash / all2all_out
  2) torch.profiler top CUDA kernels + coarse categories
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.kvcache.base import enable_ulysses_attn_profiler, get_ulysses_attn_profiler
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
SAFE_CUDA = {2: "0,2", 4: "0,2,4,5"}

KERNEL_CATEGORIES = (
    ("nccl_comm", re.compile(r"nccl|all_to_all|alltoall|all_gather|broadcast|reduce", re.I)),
    ("flash_attn", re.compile(r"flash|fmha|fwd_kvcache|attention.*fwd|scaled_dot", re.I)),
    ("gemm", re.compile(r"cublas|cutlass|gemm|_mm_|matmul|linear|gemm_nt|gemm_tn", re.I)),
    ("int8_quant", re.compile(r"int8|quantize|dequant|triton.*quant|w8a", re.I)),
    ("kivi_kv", re.compile(r"kivi|kv_quant|pack.*kv|unpack.*kv", re.I)),
    ("rope", re.compile(r"rope|rotary|triton.*causal", re.I)),
    ("norm_modulate", re.compile(r"norm|layer_norm|rms|modulate|scale_shift", re.I)),
    ("elementwise", re.compile(r"elementwise|mul|add|silu|gelu|softmax|copy|cat|slice", re.I)),
)


def _cat(name: str) -> str:
    for cat, pat in KERNEL_CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


def _cuda_ms(evt) -> float:
    for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total", "self_device_time_total"):
        val = getattr(evt, attr, None)
        if val:
            return float(val) / 1000.0
    return 0.0


def _device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _move(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move(v, device) for v in obj)
    return obj


def _latent_shape(cfg: dict[str, Any]) -> list[int]:
    return [
        cfg.get("num_channels_latents", 16),
        (cfg["target_video_length"] - 1) // cfg["vae_stride"][0] + 1,
        cfg["target_height"] // cfg["vae_stride"][1],
        cfg["target_width"] // cfg["vae_stride"][2],
    ]


def _load_payload(cfg: dict[str, Any], cache: Path, prompt: str, seed: int) -> dict[str, Any]:
    latent = _latent_shape(cfg)
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        payload["latent_shape"] = latent
        payload["inputs"]["latent_shape"] = latent
        if dist.is_initialized():
            dist.barrier()
        return payload
    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        payload["latent_shape"] = latent
        payload["inputs"]["latent_shape"] = latent
        return payload
    text_encoder = load_wan_text_encoder(cfg)[0]
    text_len = int(cfg.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    payload = {
        "seed": seed,
        "latent_shape": latent,
        "inputs": {
            "text_encoder_output": {"context": context, "context_null": None},
            "image_encoder_output": None,
            "latent_shape": latent,
        },
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    if dist.is_initialized():
        dist.barrier()
    return payload


def _warmup_to(model, scheduler, cfg, inputs, latent, seg_idx: int) -> tuple[list[int], int, int]:
    DisaggSFKVCacheManager.teardown(model)
    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent)
    scheduler.num_output_frames = num_out
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=42, latent_shape=list(ls_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    for s in range(seg_idx):
        for st in range(infer_steps):
            model.kv_cache_manager.current_step = st
            scheduler.step_pre(seg_index=s, step_index=st, is_rerun=False)
            model.infer(inputs)
            scheduler.step_post()
        scheduler.step_pre(seg_index=s, step_index=infer_steps - 1, is_rerun=True)
        model.infer(inputs)
    return ls_adj, num_out, num_chunks


def _profile_one_forward(model, scheduler, inputs, seg_idx: int, step_index: int = 0) -> dict[str, Any]:
    enable_ulysses_attn_profiler(True)
    prof_u = get_ulysses_attn_profiler()
    if prof_u is not None:
        prof_u.reset()

    model.kv_cache_manager.current_step = step_index
    scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
    _sync()
    t0 = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        model.infer(inputs)
    _sync()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    scheduler.step_post()

    ulysses = prof_u.snapshot() if prof_u is not None else {}

    cat_ms: dict[str, float] = defaultdict(float)
    top: list[dict[str, Any]] = []
    total_k = 0.0
    for evt in prof.key_averages():
        ms = _cuda_ms(evt)
        if ms <= 0:
            continue
        total_k += ms
        cat_ms[_cat(evt.key)] += ms
        top.append({"name": evt.key, "cuda_ms": round(ms, 3), "category": _cat(evt.key)})
    top.sort(key=lambda x: -x["cuda_ms"])
    top = top[:20]

    q_len = 3 * model.kv_cache_manager.frame_seq_length
    k_len = (seg_idx + 1) * q_len
    return {
        "seg_index": seg_idx,
        "step_index": step_index,
        "q_tokens": q_len,
        "k_tokens": k_len,
        "wall_ms": round(wall_ms, 2),
        "ulysses_self_attn_phases": ulysses,
        "kernel_category_ms": {k: round(v, 3) for k, v in sorted(cat_ms.items(), key=lambda x: -x[1])},
        "kernel_category_pct": {
            k: round(100.0 * v / total_k, 2) for k, v in sorted(cat_ms.items(), key=lambda x: -x[1])
        }
        if total_k > 0
        else {},
        "profiler_kernel_ms": round(total_k, 3),
        "top_kernels": top,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_14b_int8_sp_5s_kivi.json")
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--chunks", default="0,3,6", help="comma-separated chunk indices to profile")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_14b_int8_encoder_inputs_f81.pt")
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_14b_ulysses_p4_5s_op_profile.json")
    parser.add_argument("--cuda_devices", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    elif args.seq_p_size in SAFE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = SAFE_CUDA[args.seq_p_size]
    os.environ["LIGHTX2V_ULYSSES_PROFILE"] = "1"

    cfg = set_config(model_path=args.model_path, task="t2v", model_cls="wan2.1_sf", config_path=args.config_json)
    cfg["cpu_offload"] = False
    ar = dict(cfg.get("ar_config", {}))
    ar["kv_offload"] = False
    cfg["ar_config"] = ar
    cfg["parallel"] = {"seq_p_size": int(args.seq_p_size), "seq_p_attn_type": "ulysses"}

    seed_all(args.seed)
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    platform_device.init_parallel_env()
    set_parallel_config(cfg)

    payload = _load_payload(cfg, Path(args.inputs_cache), args.prompt, args.seed)
    payload = _move(payload, _device())
    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)
    _sync()

    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]
    profiles: list[dict[str, Any]] = []
    for seg in chunks:
        if is_main_process():
            print(f"\n=== profile chunk {seg} (single forward, step0) ===")
        _warmup_to(model, scheduler, cfg, payload["inputs"], list(payload["latent_shape"]), seg)
        row = _profile_one_forward(model, scheduler, payload["inputs"], seg, 0)
        profiles.append(row)
        if is_main_process():
            u = row.get("ulysses_self_attn_phases") or {}
            print(
                f"wall={row['wall_ms']:.1f}ms K={row['k_tokens']} "
                f"a2a_q={u.get('all2all_q_ms', 0):.1f} flash={u.get('flash_ms', 0):.1f} "
                f"a2a_out={u.get('all2all_out_ms', 0):.1f} | cats={row['kernel_category_pct']}"
            )
            for k in row["top_kernels"][:8]:
                print(f"  {k['cuda_ms']:7.1f}ms  [{k['category']}] {k['name'][:90]}")
        DisaggSFKVCacheManager.teardown(model)

    result = {
        "metric": "sf_ulysses_operator_profile",
        "model_cls": "wan2.1_sf",
        "seq_p_size": args.seq_p_size,
        "seq_p_attn_type": "ulysses",
        "config_json": args.config_json,
        "dit_quantized": bool(cfg.get("dit_quantized")),
        "kv_quant": bool((cfg.get("ar_config") or {}).get("kv_quant")),
        "kv_offload": bool((cfg.get("ar_config") or {}).get("kv_offload")),
        "target_video_length": cfg.get("target_video_length"),
        "chunks": profiles,
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
