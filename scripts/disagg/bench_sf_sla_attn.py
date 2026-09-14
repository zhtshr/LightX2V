#!/usr/bin/env python3
"""Benchmark SF sla_attn vs flash_attn2 baseline (speed + optional video quality)."""

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


def _make_cfg(config_path: str) -> dict[str, Any]:
    cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=config_path,
    )
    cfg["parallel"] = False
    cfg["cpu_offload"] = False
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
    del model, scheduler
    torch.cuda.empty_cache()
    return {
        "self_attn_type": cfg.get("self_attn_1_type"),
        "sla_setting": cfg.get("sla_attn_setting"),
        "transformer_s": round(avg, 4),
        "samples_s": [round(x, 4) for x in samples],
    }


def _generate_video(cfg: dict[str, Any], prompt: str, seed: int, save_path: Path) -> tuple[Path, float]:
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
    del runner
    torch.cuda.empty_cache()
    return save_path, pipeline_s


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
    parser.add_argument("--baseline_config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--sla_config", default="configs/self_forcing/wan_t2v_sf_sla_ar.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", default="save_results/sf_sla_attn")
    parser.add_argument("--skip_video", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = _make_cfg(args.baseline_config)
    sla_cfg = _make_cfg(args.sla_config)

    print("Benchmarking baseline (flash_attn2)...")
    baseline = _bench_transformer(baseline_cfg, args.prompt, args.seed)
    print("Benchmarking sla_attn...")
    sla = _bench_transformer(sla_cfg, args.prompt, args.seed)

    speedup = baseline["transformer_s"] / sla["transformer_s"]
    summary: dict[str, Any] = {
        "seed": args.seed,
        "prompt": args.prompt,
        "baseline": baseline,
        "sla_attn": sla,
        "speedup_vs_baseline": round(speedup, 4),
    }

    if not args.skip_video:
        ref_path = out_dir / "baseline_flash.mp4"
        sla_path = out_dir / "sla_triton.mp4"
        print("Generating baseline video...")
        _, baseline_pipe = _generate_video(baseline_cfg, args.prompt, args.seed, ref_path)
        print("Generating sla_attn video...")
        _, sla_pipe = _generate_video(sla_cfg, args.prompt, args.seed, sla_path)
        quality = _compare_videos(ref_path, sla_path)
        summary["baseline_pipeline_s"] = round(baseline_pipe, 3)
        summary["sla_pipeline_s"] = round(sla_pipe, 3)
        summary["quality_vs_baseline"] = quality
        summary["videos"] = {"baseline": str(ref_path), "sla": str(sla_path)}

    out_json = out_dir / "sla_attn_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
