#!/usr/bin/env python3
"""Compare SF denoise with vs without per-chunk rerun (latency + latent diff)."""

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


def _prepare_inputs(config: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([DEFAULT_PROMPT])
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
) -> tuple[float, dict[str, Any], torch.Tensor]:
    inputs = payload["inputs"]
    seed = int(payload["seed"])
    latent_shape = list(payload["latent_shape"])

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    num_forwards = num_chunks * (infer_steps + (1 if include_rerun else 0))

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
    meta = {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "include_rerun": include_rerun,
        "num_forwards": num_forwards,
    }
    return elapsed, meta, scheduler.latents.detach().cpu()


def _run_mode(
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
    seed: int,
    warmup: int,
    measure_iters: int,
) -> dict[str, Any]:
    seed_all(seed)
    model = load_wan_transformer(config)
    scheduler = load_wan_scheduler(config)
    model.set_scheduler(scheduler)

    for _ in range(warmup):
        _run_sf_denoise(model, scheduler, config, payload, include_rerun=include_rerun)

    samples: list[float] = []
    latents_last: torch.Tensor | None = None
    meta: dict[str, Any] = {}
    for _ in range(measure_iters):
        elapsed, meta, latents = _run_sf_denoise(model, scheduler, config, payload, include_rerun=include_rerun)
        samples.append(elapsed)
        latents_last = latents

    avg_s = sum(samples) / len(samples)
    del model
    torch.cuda.empty_cache()
    return {
        "transformer_s": round(avg_s, 4),
        "transformer_samples_s": [round(x, 4) for x in samples],
        "num_forwards": meta.get("num_forwards"),
        "latents": latents_last,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SF no_rerun vs baseline")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/sf_phase1_encoder_inputs.pt",
    )
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_no_rerun_validate.json")
    parser.add_argument("--save_latents_dir", default="")
    args = parser.parse_args()

    config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    config["cpu_offload"] = False
    config["parallel"] = False

    payload = _prepare_inputs(config, Path(args.inputs_cache))
    payload = _move_tensor_tree(payload, torch.device("cuda"))

    with_rerun = _run_mode(
        config, payload, include_rerun=True, seed=args.seed,
        warmup=args.warmup, measure_iters=args.measure_iters,
    )
    no_rerun = _run_mode(
        config, payload, include_rerun=False, seed=args.seed,
        warmup=args.warmup, measure_iters=args.measure_iters,
    )

    base_lat = with_rerun.pop("latents")
    no_lat = no_rerun.pop("latents")
    diff = (no_lat - base_lat).abs()

    result = {
        "model_cls": "wan2.1_sf",
        "seed": args.seed,
        "with_rerun": with_rerun,
        "no_rerun": no_rerun,
        "comparison": {
            "speedup_no_rerun_vs_with_rerun": round(
                with_rerun["transformer_s"] / no_rerun["transformer_s"], 4,
            ) if no_rerun["transformer_s"] > 0 else None,
            "saved_s": round(with_rerun["transformer_s"] - no_rerun["transformer_s"], 4),
            "saved_pct": round(
                100.0 * (with_rerun["transformer_s"] - no_rerun["transformer_s"]) / with_rerun["transformer_s"],
                2,
            ) if with_rerun["transformer_s"] > 0 else None,
            "forwards_with_rerun": with_rerun["num_forwards"],
            "forwards_no_rerun": no_rerun["num_forwards"],
            "latent_max_abs_diff": float(diff.max().item()),
            "latent_mean_abs_diff": float(diff.mean().item()),
            "latent_rel_l2_diff": float(
                diff.norm() / (base_lat.abs().norm() + 1e-12)
            ),
        },
    }

    if args.save_latents_dir:
        out_dir = Path(args.save_latents_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(base_lat, out_dir / "latents_with_rerun.pt")
        torch.save(no_lat, out_dir / "latents_no_rerun.pt")
        result["saved_latents"] = str(out_dir)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    c = result["comparison"]
    print(
        f"with_rerun: {with_rerun['transformer_s']}s ({with_rerun['num_forwards']} forwards)\n"
        f"no_rerun:   {no_rerun['transformer_s']}s ({no_rerun['num_forwards']} forwards)\n"
        f"speedup:    {c['speedup_no_rerun_vs_with_rerun']}x  saved {c['saved_s']}s ({c['saved_pct']}%)\n"
        f"latent diff max={c['latent_max_abs_diff']:.6f} mean={c['latent_mean_abs_diff']:.6f} "
        f"rel_l2={c['latent_rel_l2_diff']:.6f}\n"
        f"wrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
