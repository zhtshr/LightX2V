#!/usr/bin/env python3
"""Benchmark KV compress (budget=4) on longer SF videos: speed + quality vs baseline."""

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
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.utils import seed_all

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_cfg(base_config: str, target_video_length: int, flowcache: dict[str, Any] | None) -> dict[str, Any]:
    cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=base_config,
    )
    cfg["parallel"] = False
    cfg["cpu_offload"] = False
    cfg["target_video_length"] = int(target_video_length)
    if flowcache is not None:
        cfg.setdefault("ar_config", {}).setdefault("flowcache", {}).update(flowcache)
    return cfg


def _prepare_inputs(cfg: dict[str, Any], prompt: str) -> tuple[dict, list[int]]:
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
    return inputs, latent_shape


def _run_transformer(
    cfg: dict[str, Any],
    inputs: dict,
    latent_shape: list[int],
    seed: int,
    flowcache: SFFlowCacheManager | None,
) -> tuple[float, dict[str, Any]]:
    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)
    if flowcache is not None:
        flowcache.reset()
        model.flowcache_manager = flowcache
        model.transformer_infer.flowcache_manager = flowcache

    _sync()
    t0 = time.perf_counter()
    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
    scheduler.num_output_frames = num_out
    scheduler.prepare(seed=seed, latent_shape=list(ls_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    compress_events = 0
    frame_seq_length = 0
    try:
        frame_seq_length = model.kv_cache_manager.frame_seq_length
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
                        flowcache.on_forward(seg_idx, step_index, metric, scheduler.noise_pred[:, seg_start:seg_end])
                else:
                    model.infer(inputs)
                scheduler.step_post()
            scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
            model.infer(inputs)
            if flowcache is not None and flowcache.uses_kv_compress:
                flowcache.mark_chunk_completed(seg_idx)
                if flowcache.maybe_compress_kv(model, seg_idx):
                    compress_events += 1
    finally:
        DisaggSFKVCacheManager.teardown(model)
    _sync()
    elapsed = time.perf_counter() - t0
    meta = {
        "num_chunks": num_chunks,
        "num_output_frames": num_out,
        "tokens_per_chunk": frame_seq_length * cfg["ar_config"].get("num_frame_per_chunk", 3),
        "kv_compress_events": compress_events,
        "kv_compress_count": getattr(flowcache, "kv_compress_count", 0) if flowcache else 0,
    }
    del model, scheduler
    torch.cuda.empty_cache()
    return elapsed, meta


def _generate_video(cfg: dict[str, Any], prompt: str, seed: int, save_path: Path) -> tuple[float, dict[str, Any]]:
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
    _sync()
    runner.init_modules()
    runner.input_info = input_info
    runner.inputs = runner.run_input_encoder()
    if hasattr(runner, "text_encoders") and runner.text_encoders:
        del runner.text_encoders[0]
        runner.text_encoders = None
        torch.cuda.empty_cache()
    _sync()
    t0 = time.perf_counter()
    runner.run_main()
    _sync()
    pipeline_s = time.perf_counter() - t0
    fc = getattr(runner, "flowcache_manager", None)
    meta = {"kv_compress_count": getattr(fc, "kv_compress_count", 0) if fc else 0}
    del runner
    torch.cuda.empty_cache()
    return pipeline_s, meta


def _load_video_frames(path: Path) -> np.ndarray:
    import torchvision.io as io

    video, _, _ = io.read_video(str(path), pts_unit="sec")
    return video.numpy().astype(np.float32) / 255.0


def _ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    return float(ssim(a, b, channel_axis=2, data_range=1.0))


def _compare_videos(ref_path: Path, test_path: Path) -> dict[str, Any]:
    ref = _load_video_frames(ref_path)
    test = _load_video_frames(test_path)
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    ssims = [_ssim_frame(ref[i], test[i]) for i in range(n)]
    mae = float(np.mean(np.abs(ref - test)))
    return {
        "num_frames": n,
        "ssim_mean": round(float(np.mean(ssims)), 4),
        "ssim_min": round(float(np.min(ssims)), 4),
        "pixel_mae": round(mae, 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--target_video_length", type=int, default=161, help="~10s at 16fps (161 frames)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", default="save_results/sf_flowcache/long_video_10s")
    parser.add_argument("--skip_video", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    # ~10s: 161 frames @ 16fps; 81 frames ≈ 5s baseline reference
    fps_assume = 16
    duration_s = (args.target_video_length - 1) / fps_assume

    kv_fc = {
        "enable": True,
        "enable_feature_cache": False,
        "enable_kv_compress": True,
        "kv_budget_chunks": 4,
        "kv_compress_once": True,
        "similarity_mode": "blocked",
        "similarity_block_size": 256,
    }
    kv_roll_fc = {**kv_fc, "kv_compress_once": False}

    base_cfg = _make_cfg(args.base_config, args.target_video_length, None)
    kv_cfg = _make_cfg(args.base_config, args.target_video_length, kv_fc)
    kv_roll_cfg = _make_cfg(args.base_config, args.target_video_length, kv_roll_fc)

    print(f"target_video_length={args.target_video_length} (~{duration_s:.1f}s @ {fps_assume}fps)")
    inputs, latent_shape = _prepare_inputs(base_cfg, args.prompt)
    print(f"latent_shape={latent_shape}")

    results: dict[str, Any] = {
        "target_video_length": args.target_video_length,
        "approx_duration_s": round(duration_s, 2),
        "seed": args.seed,
        "latent_shape": latent_shape,
        "quality_threshold_ssim": 0.95,
    }

    for label, cfg, fc in [
        ("baseline", base_cfg, None),
        ("kv_budget4_once", kv_cfg, SFFlowCacheManager(kv_cfg)),
        ("kv_budget4_rolling", kv_roll_cfg, SFFlowCacheManager(kv_roll_cfg)),
    ]:
        print(f"\n[{label}] transformer bench...")
        t, meta = _run_transformer(cfg, inputs, latent_shape, args.seed, fc)
        row = {"transformer_s": round(t, 3), **meta}
        print(f"  transformer_s={row['transformer_s']} chunks={meta['num_chunks']} compress_events={meta['kv_compress_events']}")
        results[label] = row

    base_t = results["baseline"]["transformer_s"]
    for label in ("kv_budget4_once", "kv_budget4_rolling"):
        t = results[label]["transformer_s"]
        results[label]["speedup_vs_baseline"] = round(base_t / t, 4)
        results[label]["delta_s"] = round(t - base_t, 3)
        print(f"  {label}: {base_t/t:.3f}x ({t-base_t:+.3f}s)")

    if not args.skip_video:
        ref_path = out_dir / "baseline.mp4"
        print(f"\n[video] baseline -> {ref_path}")
        pipe_b, _ = _generate_video(base_cfg, args.prompt, args.seed, ref_path)
        results["baseline"]["pipeline_s"] = round(pipe_b, 3)

        for label, cfg in [("kv_budget4_once", kv_cfg), ("kv_budget4_rolling", kv_roll_cfg)]:
            vpath = out_dir / f"{label}.mp4"
            print(f"[video] {label} -> {vpath}")
            pipe_s, meta = _generate_video(cfg, args.prompt, args.seed, vpath)
            q = _compare_videos(ref_path, vpath)
            results[label]["pipeline_s"] = round(pipe_s, 3)
            results[label]["quality_vs_baseline"] = q
            results[label]["kv_compress_count_video"] = meta["kv_compress_count"]
            ok = q["ssim_mean"] >= 0.95
            print(f"  SSIM={q['ssim_mean']} MAE={q['pixel_mae']} pipeline={pipe_s:.1f}s pass_5%={ok}")

    out_json = out_dir / "long_video_summary.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
