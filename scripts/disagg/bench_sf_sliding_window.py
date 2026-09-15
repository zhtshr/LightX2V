#!/usr/bin/env python3
"""Benchmark SF sliding window (local_attn_size) for speed vs quality."""

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


def _make_config(base_path: str, local_attn_size: int, cpu_offload: bool) -> dict[str, Any]:
    cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=base_path,
    )
    cfg["cpu_offload"] = cpu_offload
    cfg["parallel"] = False
    ar = dict(cfg.get("ar_config", {}))
    ar["local_attn_size"] = local_attn_size
    cfg["ar_config"] = ar
    return cfg


def _bench_transformer(cfg: dict[str, Any], prompt: str, seed: int, warmup: int = 1) -> dict[str, Any]:
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

    def run_once() -> float:
        t0 = time.perf_counter()
        ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
        scheduler.num_output_frames = num_out
        scheduler.prepare(seed=seed, latent_shape=list(ls_adj), image_encoder_output=None)
        infer_steps = int(scheduler.infer_steps)
        try:
            for seg_idx in range(num_chunks):
                for step_index in range(infer_steps):
                    model.kv_cache_manager.current_step = step_index
                    scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
                    model.infer(inputs)
                    scheduler.step_post()
                scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
                model.infer(inputs)
        finally:
            DisaggSFKVCacheManager.teardown(model)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        run_once()

    samples = [run_once() for _ in range(3)]

    avg = sum(samples) / len(samples)
    kv = model.kv_cache_manager if hasattr(model, "kv_cache_manager") and model.kv_cache_manager else None
    fsl = None
    kv_tokens = None
    if kv is None:
        DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
        kv = model.kv_cache_manager
    if kv is not None:
        fsl = int(kv.frame_seq_length)
        kv_tokens = int(kv.kv_size)
        DisaggSFKVCacheManager.teardown(model)

    ar = cfg["ar_config"]
    la = ar.get("local_attn_size", -1)
    max_k = (la if la != -1 else lat_f) * (fsl or 0)

    del model, scheduler
    torch.cuda.empty_cache()

    return {
        "local_attn_size_frames": la,
        "max_k_tokens": max_k,
        "kv_cache_tokens": kv_tokens,
        "transformer_s": round(avg, 4),
        "samples_s": [round(x, 4) for x in samples],
    }


def _generate_video(cfg: dict[str, Any], prompt: str, seed: int, save_path: Path) -> Path:
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
    # WanSFRunner expects config dict via set_config path; build manually
    runner_cfg = copy.deepcopy(cfg)
    input_info = init_empty_input_info("t2v")
    update_input_info_from_dict(
        input_info,
        {"prompt": prompt, "save_result_path": str(save_path), "seed": seed},
    )
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
    del runner
    torch.cuda.empty_cache()
    return save_path, pipeline_s


def _load_video_frames(path: Path, max_frames: int | None = None) -> np.ndarray:
    import torchvision.io as io

    video, _, info = io.read_video(str(path), pts_unit="sec")
    frames = video.numpy().astype(np.float32) / 255.0
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: H,W,C in [0,1]
    try:
        from skimage.metrics import structural_similarity as ssim

        return float(ssim(a, b, channel_axis=2, data_range=1.0))
    except ImportError:
        # fallback: grayscale SSIM-ish via correlation
        ag = a.mean(axis=2)
        bg = b.mean(axis=2)
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
    mae = float(np.mean(np.abs(ref - test)))
    return {
        "num_frames": n,
        "psnr_mean": round(float(np.mean(psnrs)), 3),
        "psnr_min": round(float(np.min(psnrs)), 3),
        "psnr_per_frame": [round(x, 2) for x in psnrs[:: max(1, n // 8)]],
        "ssim_mean": round(float(np.mean(ssims)), 4),
        "ssim_min": round(float(np.min(ssims)), 4),
        "ssim_per_frame_sample": [round(x, 4) for x in ssims[:: max(1, n // 8)]],
        "pixel_mae": round(mae, 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--window_frames", default="-1,15,12,9,6,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", default="save_results/sf_sliding_window")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [int(x.strip()) for x in args.window_frames.split(",") if x.strip()]

    perf_rows = []
    quality_rows = []
    baseline_video = out_dir / "baseline_full_la-1.mp4"

    for la in windows:
        label = "full" if la == -1 else f"la{la}f"
        print(f"\n=== local_attn_size={la} ({label}) ===")
        cfg = _make_config(args.base_config, la, cpu_offload=args.cpu_offload)
        perf = _bench_transformer(cfg, args.prompt, args.seed)
        perf_rows.append(perf)
        print(f"  transformer: {perf['transformer_s']}s, max_k={perf['max_k_tokens']} tokens")

        if args.skip_video:
            continue

        vid_path = out_dir / f"video_{label}.mp4"
        _, pipeline_s = _generate_video(cfg, args.prompt, args.seed, vid_path)
        perf["pipeline_s"] = round(pipeline_s, 3)
        print(f"  pipeline: {pipeline_s:.3f}s -> {vid_path}")

        if la == -1:
            baseline_video = vid_path
            quality_rows.append({"local_attn_size_frames": la, "label": label, "vs_baseline": "self"})
        else:
            q = _compare_videos(baseline_video, vid_path)
            q["local_attn_size_frames"] = la
            q["label"] = label
            quality_rows.append(q)
            print(f"  vs baseline: PSNR={q['psnr_mean']} dB, SSIM={q['ssim_mean']}, MAE={q['pixel_mae']}")

    baseline_s = next(r["transformer_s"] for r in perf_rows if r["local_attn_size_frames"] == -1)
    for r in perf_rows:
        r["speedup_vs_full"] = round(baseline_s / r["transformer_s"], 3) if r["transformer_s"] > 0 else None

    summary = {
        "config": {
            "resolution": "832x480",
            "cpu_offload": args.cpu_offload,
            "seed": args.seed,
            "prompt": args.prompt[:80],
            "window_frames_tested": windows,
            "note": "local_attn_size is in latent frames; fsl=1560 tokens/frame at 480p",
        },
        "performance": perf_rows,
        "quality_vs_full": quality_rows,
    }
    json_path = out_dir / "sliding_window_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# SF Sliding Window Benchmark",
        "",
        f"- Seed={args.seed}, cpu_offload={args.cpu_offload}",
        "",
        "## Performance (transformer denoise only)",
        "",
        "| local_attn (frames) | max K tokens | transformer_s | speedup vs full |",
        "|---------------------|--------------|---------------|-----------------|",
    ]
    for r in perf_rows:
        la = r["local_attn_size_frames"]
        md.append(
            f"| {la} | {r['max_k_tokens']} | {r['transformer_s']} | {r['speedup_vs_full']}x |"
        )
    if quality_rows:
        md += [
            "",
            "## Quality vs full attention (-1)",
            "",
            "| local_attn (frames) | PSNR mean (dB) | SSIM mean | pixel MAE |",
            "|---------------------|------------------|-----------|-----------|",
        ]
        for q in quality_rows:
            if q.get("vs_baseline") == "self":
                continue
            md.append(
                f"| {q['local_attn_size_frames']} | {q['psnr_mean']} | {q['ssim_mean']} | {q['pixel_mae']} |"
            )
    md_path = out_dir / "sliding_window_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
