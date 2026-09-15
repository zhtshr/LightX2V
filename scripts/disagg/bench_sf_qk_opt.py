#!/usr/bin/env python3
"""Benchmark chunk0 (q=k) forward for self-attn / Q=K optimization experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config

DEFAULT_PROMPT = "A stylish woman strolls down a bustling Tokyo street."


def _prepare(cfg: dict[str, Any], prompt: str) -> tuple[Any, Any, dict[str, Any], list[int]]:
    text_encoder = load_wan_text_encoder(cfg)[0]
    text_len = int(cfg.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    torch.cuda.empty_cache()

    vae_stride = cfg["vae_stride"]
    h, w = cfg["target_height"], cfg["target_width"]
    latent_shape = [
        16,
        (cfg["target_video_length"] - 1) // vae_stride[0] + 1,
        h // vae_stride[1],
        w // vae_stride[2],
    ]
    inputs = {
        "text_encoder_output": {"context": context.cuda(), "context_null": None},
        "image_encoder_output": None,
    }
    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)
    return model, scheduler, inputs, latent_shape


def _run_once(model, scheduler, cfg, inputs, latent_shape, seg_idx: int, step_index: int, seed: int) -> dict[str, float]:
    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    ls_adj, num_out, _ = DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
    scheduler.num_output_frames = num_out
    scheduler.prepare(seed=seed, latent_shape=list(ls_adj), image_encoder_output=None)

    model.kv_cache_manager.current_step = step_index
    scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.infer(inputs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    scheduler.step_post()

    kv = ti.finalize_kv_store_profile() if hasattr(ti, "finalize_kv_store_profile") else {}
    DisaggSFKVCacheManager.teardown(model)

    return {
        "forward_ms": elapsed_ms,
        "self_attn_ms": float(kv.get("self_attn_ms", 0) if kv else 0),
        "cross_attn_ms": float(kv.get("cross_attn_ms", 0) if kv else 0),
        "store_kv_ms": float(kv.get("store_kv_ms", 0) if kv else 0),
        "kv_read_ms": float(kv.get("kv_read_ms", 0) if kv else 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--seg", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config,
    )
    cfg["parallel"] = False
    ar = dict(cfg.get("ar_config", {}))
    ar["profile_kv_store"] = True
    cfg["ar_config"] = ar

    model, scheduler, inputs, latent_shape = _prepare(cfg, DEFAULT_PROMPT)

    for _ in range(args.warmup):
        _run_once(model, scheduler, cfg, inputs, latent_shape, args.seg, args.step, args.seed)

    samples = [
        _run_once(model, scheduler, cfg, inputs, latent_shape, args.seg, args.step, args.seed)
        for _ in range(args.iters)
    ]

    def avg(key: str) -> float:
        return sum(s[key] for s in samples) / len(samples)

    q_len = 3 * model.kv_cache_manager.frame_seq_length if hasattr(model, "kv_cache_manager") and model.kv_cache_manager else 4680
    k_len = (args.seg + 1) * q_len

    out = {
        "label": args.label,
        "config": args.config,
        "seg_index": args.seg,
        "step_index": args.step,
        "q_tokens": q_len,
        "k_tokens": k_len,
        "forward_ms_avg": round(avg("forward_ms"), 3),
        "self_attn_ms_avg": round(avg("self_attn_ms"), 3),
        "cross_attn_ms_avg": round(avg("cross_attn_ms"), 3),
        "samples": [{k: round(v, 3) for k, v in s.items()} for s in samples],
    }
    print(json.dumps(out, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
