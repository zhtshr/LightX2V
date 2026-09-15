#!/usr/bin/env python3
"""Profile SF self-attn store_kv cost and validate store_kv_only_on_rerun."""

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


def _run_sf_denoise(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
    collect_profile: bool,
) -> tuple[float, dict[str, Any], torch.Tensor]:
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

    profile = None
    if collect_profile and hasattr(ti, "finalize_kv_store_profile"):
        profile = ti.finalize_kv_store_profile()

    meta = {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "include_rerun": include_rerun,
        "kv_profile": profile,
    }
    return elapsed, meta, scheduler.latents.detach().cpu()


def _apply_mode(config: dict[str, Any], mode: str, profile: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    ar = dict(cfg.get("ar_config", {}))
    ar["profile_kv_store"] = profile
    ar["store_kv_only_on_rerun"] = mode == "rerun_only_store"
    cfg["ar_config"] = ar
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="SF KV store_kv profile + rerun-only store validation")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/sf_phase1_encoder_inputs.pt",
    )
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_kv_store_profile.json")
    args = parser.parse_args()

    base_config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    base_config["cpu_offload"] = False
    base_config["parallel"] = False

    payload = _prepare_inputs(base_config, args.prompt, Path(args.inputs_cache))
    device = torch.device("cuda")
    payload = _move_tensor_tree(payload, device)
    include_rerun = not args.no_rerun
    modes = ["baseline", "rerun_only_store"]

    results: dict[str, Any] = {
        "model_cls": "wan2.1_sf",
        "include_rerun": include_rerun,
        "modes": {},
    }
    baseline_latents: torch.Tensor | None = None

    for mode in modes:
        cfg = _apply_mode(base_config, mode, profile=True)
        seed_all(args.seed)
        model = load_wan_transformer(cfg)
        scheduler = load_wan_scheduler(cfg)
        model.set_scheduler(scheduler)

        for _ in range(args.warmup):
            _run_sf_denoise(model, scheduler, cfg, payload, include_rerun=include_rerun, collect_profile=False)

        samples: list[float] = []
        profiles: list[dict[str, Any]] = []
        latents_last: torch.Tensor | None = None
        for _ in range(args.measure_iters):
            elapsed, meta, latents = _run_sf_denoise(
                model, scheduler, cfg, payload, include_rerun=include_rerun, collect_profile=True,
            )
            samples.append(elapsed)
            if meta.get("kv_profile"):
                profiles.append(meta["kv_profile"])
            latents_last = latents

        avg_s = sum(samples) / len(samples)
        prof = profiles[-1] if profiles else {}
        transformer_ms = avg_s * 1000.0
        store_ms = float(prof.get("store_kv_ms", 0.0))
        self_attn_ms = float(prof.get("self_attn_ms", 0.0))
        sp_a2a_ms = float(prof.get("sp_a2a_ms", 0.0))
        kv_read_ms = float(prof.get("kv_read_ms", 0.0))

        mode_result = {
            "transformer_s": round(avg_s, 4),
            "transformer_ms": round(transformer_ms, 2),
            "kv_profile": prof,
            "store_kv_share_of_self_attn_pct": round(100.0 * store_ms / self_attn_ms, 2) if self_attn_ms > 0 else None,
            "store_kv_share_of_transformer_pct": round(100.0 * store_ms / transformer_ms, 2) if transformer_ms > 0 else None,
            "self_attn_share_of_transformer_pct": round(100.0 * self_attn_ms / transformer_ms, 2) if transformer_ms > 0 else None,
            "sp_a2a_share_of_transformer_pct": round(100.0 * sp_a2a_ms / transformer_ms, 2) if transformer_ms > 0 else None,
            "kv_read_share_of_transformer_pct": round(100.0 * kv_read_ms / transformer_ms, 2) if transformer_ms > 0 else None,
        }

        if mode == "baseline":
            baseline_latents = latents_last
        elif baseline_latents is not None and latents_last is not None:
            diff = (latents_last - baseline_latents).abs()
            mode_result["latent_max_abs_diff_vs_baseline"] = float(diff.max().item())
            mode_result["latent_mean_abs_diff_vs_baseline"] = float(diff.mean().item())

        if mode == "rerun_only_store" and baseline_latents is not None:
            mode_result["speedup_vs_baseline"] = round(
                results["modes"]["baseline"]["transformer_s"] / avg_s, 4,
            ) if avg_s > 0 else None

        results["modes"][mode] = mode_result
        del model
        torch.cuda.empty_cache()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    base = results["modes"]["baseline"]
    opt = results["modes"]["rerun_only_store"]
    print(f"baseline: {base['transformer_s']}s store_kv={base['kv_profile'].get('store_kv_ms')}ms "
          f"({base['store_kv_share_of_transformer_pct']}% of transformer, "
          f"{base['store_kv_share_of_self_attn_pct']}% of self_attn)")
    print(f"rerun_only_store: {opt['transformer_s']}s store_kv={opt['kv_profile'].get('store_kv_ms')}ms "
          f"speedup={opt.get('speedup_vs_baseline')}x "
          f"latent_max_diff={opt.get('latent_max_abs_diff_vs_baseline')}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
