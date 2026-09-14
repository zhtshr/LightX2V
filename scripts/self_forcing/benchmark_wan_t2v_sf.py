#!/usr/bin/env python3
"""Wan2.1 Self-Forcing T2V smoke + latency benchmark."""

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import seed_all

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)


def build_config(model_path: str, config_json: str, seed: int):
    args = Namespace(
        model_cls="wan2.1_sf",
        task="t2v",
        model_path=model_path,
        config_json=config_json,
        seed=seed,
        support_tasks=[],
        parallel=False,
        num_iterations=None,
    )
    return set_config(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_local.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--save_path", default="save_results/wan_t2v_sf_bench.mp4")
    parser.add_argument("--summary_path", default="save_results/wan_t2v_sf_benchmark_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = build_config(args.model_path, args.config_json, args.seed)
    seed_all(args.seed)

    input_info = init_empty_input_info("t2v")
    update_input_info_from_dict(
        input_info,
        {
            "prompt": args.prompt,
            "save_result_path": args.save_path,
            "seed": args.seed,
        },
    )

    runner = WanSFRunner(config)

    torch.cuda.synchronize()
    t_load0 = time.perf_counter()
    runner.init_modules()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load0

    runner.input_info = input_info
    torch.cuda.synchronize()
    t_enc0 = time.perf_counter()
    runner.inputs = runner.run_input_encoder()
    torch.cuda.synchronize()
    encoder_s = time.perf_counter() - t_enc0

    if hasattr(runner, "text_encoders") and runner.text_encoders:
        del runner.text_encoders[0]
        runner.text_encoders = None
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runner.run_main()
    torch.cuda.synchronize()
    pipeline_s = time.perf_counter() - t0

    out = Path(args.save_path)
    ar_cfg = config.get("ar_config", {})
    vae_stride_t = config.get("vae_stride", (4, 8, 8))[0]
    latent_frames = (config.get("target_video_length", 81) - 1) // vae_stride_t + 1
    num_chunks = latent_frames // ar_cfg.get("num_frame_per_chunk", 3)
    summary = {
        "model_cls": "wan2.1_sf",
        "model_path": args.model_path,
        "sf_ckpt": config.get("dit_original_ckpt"),
        "success": out.exists(),
        "output": str(out) if out.exists() else None,
        "load_latency_s": round(load_s, 3),
        "encoder_latency_s": round(encoder_s, 3),
        "pipeline_latency_s": round(pipeline_s, 3),
        "total_latency_s": round(load_s + encoder_s + pipeline_s, 3),
        "infer_steps_per_chunk": config.get("infer_steps", 4),
        "num_chunks": num_chunks,
        "latent_frames": latent_frames,
        "target_video_length": config.get("target_video_length"),
        "resolution": f"{config.get('target_width')}x{config.get('target_height')}",
        "num_frame_per_chunk": ar_cfg.get("num_frame_per_chunk"),
        "seconds_per_chunk": round(pipeline_s / max(num_chunks, 1), 3),
    }

    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
