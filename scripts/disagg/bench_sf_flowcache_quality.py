#!/usr/bin/env python3
"""Benchmark FlowCache feature cache vs KV compression: speed + video quality vs baseline."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.utils import seed_all

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)

PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"enable": False},
    "kv_only": {
        "enable": True,
        "enable_feature_cache": False,
        "enable_kv_compress": True,
        "kv_compress_once": True,
        "kv_budget_chunks": 4,
        "similarity_mode": "blocked",
        "similarity_block_size": 256,
    },
    "feat_0.01": {
        "enable": True,
        "enable_feature_cache": True,
        "enable_kv_compress": False,
        "rel_l1_thresh": 0.01,
    },
    "feat_1.0": {
        "enable": True,
        "enable_feature_cache": True,
        "enable_kv_compress": False,
        "rel_l1_thresh": 1.0,
    },
    "feat_1.5": {
        "enable": True,
        "enable_feature_cache": True,
        "enable_kv_compress": False,
        "rel_l1_thresh": 1.5,
    },
    "feat_2.0": {
        "enable": True,
        "enable_feature_cache": True,
        "enable_kv_compress": False,
        "rel_l1_thresh": 2.0,
    },
    "feat_3.0": {
        "enable": True,
        "enable_feature_cache": True,
        "enable_kv_compress": False,
        "rel_l1_thresh": 3.0,
    },
    "both_default": {
        "enable": True,
        "enable_feature_cache": True,
        "enable_kv_compress": True,
        "kv_compress_once": True,
        "rel_l1_thresh": 0.01,
    },
}


def _make_cfg(config_path: str, flowcache: dict[str, Any] | None) -> dict[str, Any]:
    cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=config_path,
    )
    cfg["parallel"] = False
    cfg["cpu_offload"] = False
    if flowcache is not None:
        cfg.setdefault("ar_config", {}).setdefault("flowcache", {}).update(flowcache)
    return cfg


def _generate_video(cfg: dict[str, Any], prompt: str, seed: int, save_path: Path) -> tuple[Path, float, dict[str, Any]]:
    from argparse import Namespace

    seed_all(seed)
    args = Namespace(
        model_cls="wan2.1_sf",
        task="t2v",
        model_path=cfg["model_path"],
        config_json="",
        seed=seed,
        support_tasks=[],
        parallel=False,
        num_iterations=None,
    )
    runner_cfg = copy.deepcopy(cfg)
    input_info = init_empty_input_info("t2v")
    update_input_info_from_dict(input_info, {"prompt": prompt, "save_result_path": str(save_path), "seed": seed})
    runner = WanSFRunner(runner_cfg)
    torch.cuda.synchronize()
    runner.init_modules()
    runner.input_info = input_info
    runner.inputs = runner.run_input_encoder()
    if hasattr(runner, "text_encoders") and runner.text_encoders:
        del runner.text_encoders[0]
        runner.text_encoders = None
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runner.run_main()
    torch.cuda.synchronize()
    pipeline_s = time.perf_counter() - t0
    fc_stats = {}
    fc = getattr(runner, "flowcache_manager", None)
    if fc is not None and fc.feature_cache is not None:
        fc_stats = fc.feature_cache.stats()
    del runner
    torch.cuda.empty_cache()
    return save_path, pipeline_s, fc_stats


def _bench_transformer(cfg: dict[str, Any], prompt: str, seed: int, warmup: int = 0) -> dict[str, Any]:
    from lightx2v.common.flowcache import SFFlowCacheManager

    text_encoder = load_wan_text_encoder(cfg)[0]
    text_len = int(cfg.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    torch.cuda.empty_cache()

    vae_stride = cfg["vae_stride"]
    h, w = cfg["target_height"], cfg["target_width"]
    lat_f = (cfg["target_video_length"] - 1) // vae_stride[0] + 1
    latent_shape = [16, lat_f, h // vae_stride[1], w // vae_stride[2]]
    inputs = {"text_encoder_output": {"context": context.cuda(), "context_null": None}, "image_encoder_output": None}

    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)
    flowcache = SFFlowCacheManager(cfg)
    model.flowcache_manager = flowcache
    model.transformer_infer.flowcache_manager = flowcache

    def run_once() -> float:
        flowcache.reset()
        t0 = time.perf_counter()
        ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
        scheduler.num_output_frames = num_out
        scheduler.prepare(seed=seed, latent_shape=list(ls_adj), image_encoder_output=None)
        infer_steps = int(scheduler.infer_steps)
        try:
            for seg_idx in range(num_chunks):
                flowcache.begin_chunk(seg_idx)
                for step_index in range(infer_steps):
                    model.kv_cache_manager.current_step = step_index
                    scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
                    if flowcache.uses_feature_cache:
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
                if flowcache.uses_kv_compress:
                    flowcache.mark_chunk_completed(seg_idx)
                    flowcache.maybe_compress_kv(model, seg_idx)
        finally:
            DisaggSFKVCacheManager.teardown(model)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        run_once()
    samples = [run_once() for _ in range(2)]
    avg = sum(samples) / len(samples)
    fc_stats = flowcache.feature_cache.stats() if flowcache.feature_cache is not None else {}
    del model, scheduler
    torch.cuda.empty_cache()
    return {
        "transformer_s": round(avg, 4),
        "samples_s": [round(x, 4) for x in samples],
        "flowcache_stats": fc_stats,
    }


def _load_video_frames(path: Path) -> np.ndarray:
    import torchvision.io as io

    video, _, _ = io.read_video(str(path), pts_unit="sec")
    return video.numpy().astype(np.float32) / 255.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim

        return float(ssim(a, b, channel_axis=2, data_range=1.0))
    except ImportError:
        ag, bg = a.mean(axis=2), b.mean(axis=2)
        va, vb = ag.var(), bg.var()
        if va < 1e-12 or vb < 1e-12:
            return 1.0 if np.allclose(ag, bg) else 0.0
        return float(np.corrcoef(ag.flatten(), bg.flatten())[0, 1])


def _compare_videos(ref_path: Path, test_path: Path) -> dict[str, Any]:
    ref = _load_video_frames(ref_path)
    test = _load_video_frames(test_path)
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    psnrs = [_psnr(ref[i], test[i]) for i in range(n)]
    ssims = [_ssim_frame(ref[i], test[i]) for i in range(n)]
    return {
        "num_frames": n,
        "psnr_mean": round(float(np.mean(psnrs)), 3),
        "psnr_min": round(float(np.min(psnrs)), 3),
        "ssim_mean": round(float(np.mean(ssims)), 4),
        "ssim_min": round(float(np.min(ssims)), 4),
        "pixel_mae": round(float(np.mean(np.abs(ref - test))), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", default="save_results/sf_flowcache/quality")
    parser.add_argument(
        "--presets",
        default="baseline,kv_only,feat_0.01,feat_1.5,feat_2.0,feat_3.0,both_default",
        help="Comma-separated preset names",
    )
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--skip_speed", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
    for name in preset_names:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {list(PRESETS)}")

    rows: list[dict[str, Any]] = []
    ref_path: Path | None = None

    for name in preset_names:
        fc = PRESETS[name]
        cfg = _make_cfg(args.base_config, fc)
        row: dict[str, Any] = {"preset": name, "flowcache": fc}

        if not args.skip_speed:
            print(f"[{name}] benchmarking transformer...")
            speed = _bench_transformer(cfg, args.prompt, args.seed)
            row.update(speed)
            print(f"  transformer_s={speed['transformer_s']} reuse={speed.get('flowcache_stats', {})}")

        if not args.skip_video:
            video_path = out_dir / f"{name}.mp4"
            print(f"[{name}] generating video -> {video_path}")
            _, pipeline_s, fc_stats = _generate_video(cfg, args.prompt, args.seed, video_path)
            row["pipeline_s"] = round(pipeline_s, 3)
            row["flowcache_stats_video"] = fc_stats
            row["video"] = str(video_path)
            if name == "baseline":
                ref_path = video_path
            elif ref_path is not None:
                quality = _compare_videos(ref_path, video_path)
                row["quality_vs_baseline"] = quality
                print(
                    f"  vs baseline: PSNR={quality['psnr_mean']} dB, "
                    f"SSIM={quality['ssim_mean']}, MAE={quality['pixel_mae']}"
                )

        rows.append(row)

    summary = {
        "seed": args.seed,
        "prompt": args.prompt,
        "base_config": args.base_config,
        "results": rows,
    }
    out_json = out_dir / "quality_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== Quality vs baseline ===")
    print("| preset | transformer_s | reuse_ratio | PSNR | SSIM | MAE |")
    print("|--------|---------------|-------------|------|------|-----|")
    for row in rows:
        q = row.get("quality_vs_baseline", {})
        stats = row.get("flowcache_stats_video") or row.get("flowcache_stats") or {}
        reuse = stats.get("reuse_ratio", "")
        if isinstance(reuse, float):
            reuse = f"{reuse:.1%}"
        print(
            f"| {row['preset']} | {row.get('transformer_s', '-')} | {reuse} | "
            f"{q.get('psnr_mean', '-')} | {q.get('ssim_mean', '-')} | {q.get('pixel_mae', '-')} |"
        )
    print(f"\nSaved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
