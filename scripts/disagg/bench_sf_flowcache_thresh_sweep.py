#!/usr/bin/env python3
"""Sweep FlowCache rel_l1_thresh: reuse ratio, speed, latent + optional video quality."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lightx2v.common.flowcache import SFFlowCacheManager
from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_transformer, set_config
from lightx2v.utils.utils import seed_all
from scripts.disagg.bench_sf_flowcache_quality import (
    PRESETS,
    _compare_videos,
    _generate_video,
    _make_cfg,
    DEFAULT_PROMPT,
)

DEFAULT_PROMPT_CACHE = Path("save_results/optimization_study/sf_phase1_encoder_inputs.pt")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_latents(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    flowcache: SFFlowCacheManager | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs = payload["inputs"]
    latent_shape = list(payload["latent_shape"])
    seed = int(payload["seed"])

    if flowcache is not None:
        flowcache.reset()
        model.flowcache_manager = flowcache
        model.transformer_infer.flowcache_manager = flowcache

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)

    infer_steps = int(scheduler.infer_steps)
    try:
        for seg_idx in range(num_chunks):
            if flowcache is not None:
                flowcache.begin_chunk(seg_idx)
            for step_index in range(infer_steps):
                model.kv_cache_manager.current_step = step_index
                scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)

                if flowcache is not None and flowcache.uses_feature_cache:
                    metric = flowcache.compute_metric(model, inputs)
                    if flowcache.should_skip_transformer(seg_idx, step_index, metric, is_rerun=False):
                        flowcache.apply_cached_noise_pred(scheduler, seg_idx)
                    else:
                        model.infer(inputs)
                        seg_start = seg_idx * scheduler.num_frame_per_chunk
                        seg_end = min((seg_idx + 1) * scheduler.num_frame_per_chunk, scheduler.num_output_frames)
                        noise_pred = scheduler.noise_pred[:, seg_start:seg_end]
                        flowcache.on_forward(seg_idx, step_index, metric, noise_pred)
                else:
                    model.infer(inputs)

                scheduler.step_post()

            scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
            model.infer(inputs)
    finally:
        DisaggSFKVCacheManager.teardown(model)

    _sync()
    latents = scheduler.latents.detach().float().clone()
    stats = flowcache.feature_cache.stats() if flowcache is not None and flowcache.feature_cache is not None else {}
    return latents, stats


def _latent_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    r = ref.flatten()
    t = test.flatten()
    mse = float(torch.mean((r - t) ** 2).item())
    rel_l1 = float((r - t).abs().mean() / (r.abs().mean() + 1e-8))
    cos = float(torch.nn.functional.cosine_similarity(r.unsqueeze(0), t.unsqueeze(0)).item())
    max_abs = float((r - t).abs().max().item())
    return {
        "latent_mse": round(mse, 6),
        "latent_rel_l1": round(rel_l1, 6),
        "latent_cosine": round(cos, 6),
        "latent_max_abs": round(max_abs, 6),
    }


def _parse_thresholds(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--inputs_cache", default=str(DEFAULT_PROMPT_CACHE))
    parser.add_argument(
        "--thresholds",
        default="0.01,0.02,0.03,0.05,0.07,0.1,0.12,0.15,0.18,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
    )
    parser.add_argument("--output_json", default="save_results/sf_flowcache/thresh_sweep.json")
    parser.add_argument("--video_validate", action="store_true", help="Run video SSIM for promising thresholds")
    parser.add_argument("--video_ssim_min", type=float, default=0.95)
    parser.add_argument("--latent_cosine_min", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--video_dir", default="save_results/sf_flowcache/thresh_sweep_videos")
    args = parser.parse_args()

    cache_path = Path(args.inputs_cache)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing inputs cache: {cache_path}")

    payload = torch.load(cache_path, map_location="cuda", weights_only=False)
    seed_all(int(payload["seed"]))

    base_cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.base_config,
    )

    thresholds = _parse_thresholds(args.thresholds)
    print("Loading model...")
    model = load_wan_transformer(base_cfg)
    scheduler = load_wan_scheduler(base_cfg)
    model.set_scheduler(scheduler)

    _sync()
    t0 = time.perf_counter()
    ref_latents, _ = _run_latents(model, scheduler, base_cfg, payload, None)
    baseline_s = time.perf_counter() - t0
    print(f"baseline latent run: {baseline_s:.3f}s")

    rows: list[dict[str, Any]] = []
    for thresh in thresholds:
        fc_cfg = copy.deepcopy(base_cfg)
        fc_cfg.setdefault("ar_config", {}).setdefault("flowcache", {}).update(
            {
                "enable": True,
                "enable_feature_cache": True,
                "enable_kv_compress": False,
                "rel_l1_thresh": thresh,
            }
        )
        flowcache = SFFlowCacheManager(fc_cfg)
        _sync()
        t0 = time.perf_counter()
        latents, stats = _run_latents(model, scheduler, fc_cfg, payload, flowcache)
        elapsed = time.perf_counter() - t0
        metrics = _latent_metrics(ref_latents, latents)
        reuse_ratio = float(stats.get("reuse_ratio", 0.0))
        row = {
            "rel_l1_thresh": thresh,
            "transformer_s": round(elapsed, 4),
            "speedup_vs_baseline": round(baseline_s / elapsed, 4) if elapsed else 0.0,
            "reuse_count": stats.get("reuse_count", 0),
            "total_steps": stats.get("total_steps", 0),
            "reuse_ratio": round(reuse_ratio, 4),
            **metrics,
        }
        rows.append(row)
        print(
            f"thresh={thresh:5.2f} reuse={reuse_ratio:5.1%} "
            f"t={elapsed:6.3f}s ({baseline_s/elapsed:.3f}x) "
            f"cos={metrics['latent_cosine']:.6f} rel_l1={metrics['latent_rel_l1']:.6f}"
        )

    del model, scheduler
    torch.cuda.empty_cache()

    # Find knee: last threshold with near-lossless latents before big jump
    lossless = [r for r in rows if r["latent_cosine"] >= 0.99999 and r["latent_rel_l1"] < 1e-5]
    promising = [
        r
        for r in rows
        if r["reuse_ratio"] > 0
        and (r["latent_cosine"] >= args.latent_cosine_min or r["latent_rel_l1"] < 0.001)
    ]

    summary: dict[str, Any] = {
        "seed": int(payload["seed"]),
        "baseline_transformer_s": round(baseline_s, 4),
        "thresholds": rows,
        "lossless_thresholds": [r["rel_l1_thresh"] for r in lossless],
        "max_lossless_thresh": max((r["rel_l1_thresh"] for r in lossless), default=None),
        "promising_for_video": [r["rel_l1_thresh"] for r in promising],
    }

    if args.video_validate and promising:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        ref_video = video_dir / "baseline.mp4"
        if not ref_video.is_file():
            print(f"Generating baseline video -> {ref_video}")
            _generate_video(_make_cfg(args.base_config, PRESETS["baseline"]), args.prompt, args.seed, ref_video)

        video_rows = []
        # Always validate max lossless + promising band
        validate_thresh = sorted(
            set(summary.get("lossless_thresholds", []) + [r["rel_l1_thresh"] for r in promising])
        )
        for thresh in validate_thresh:
            cfg = _make_cfg(
                args.base_config,
                {
                    "enable": True,
                    "enable_feature_cache": True,
                    "enable_kv_compress": False,
                    "rel_l1_thresh": thresh,
                },
            )
            vpath = video_dir / f"thresh_{thresh:.4f}.mp4".replace(".", "p", 1)
            print(f"Video validate thresh={thresh} -> {vpath}")
            _, pipe_s, fc_stats = _generate_video(cfg, args.prompt, args.seed, vpath)
            quality = _compare_videos(ref_video, vpath)
            vr = {
                "rel_l1_thresh": thresh,
                "pipeline_s": round(pipe_s, 3),
                "flowcache_stats": fc_stats,
                "quality_vs_baseline": quality,
            }
            video_rows.append(vr)
            print(
                f"  PSNR={quality['psnr_mean']} SSIM={quality['ssim_mean']} "
                f"reuse={fc_stats.get('reuse_ratio', 0):.1%}"
            )

        summary["video_validation"] = video_rows
        good = [v for v in video_rows if v["quality_vs_baseline"]["ssim_mean"] >= args.video_ssim_min]
        summary["recommended_thresholds"] = [v["rel_l1_thresh"] for v in good]

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== Sweep summary ===")
    print(f"baseline: {baseline_s:.3f}s")
    print(f"max lossless thresh (latent): {summary['max_lossless_thresh']}")
    print("| thresh | reuse | speedup | latent_cos | latent_rel_l1 |")
    print("|--------|-------|---------|------------|---------------|")
    for r in rows:
        print(
            f"| {r['rel_l1_thresh']:.2f} | {r['reuse_ratio']:.1%} | {r['speedup_vs_baseline']:.3f}x | "
            f"{r['latent_cosine']:.6f} | {r['latent_rel_l1']:.6f} |"
        )
    if summary.get("recommended_thresholds"):
        print(f"recommended (SSIM>={args.video_ssim_min}): {summary['recommended_thresholds']}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
