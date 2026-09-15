#!/usr/bin/env python3
"""Minimal BAGEL T2I/I2I smoke + latency benchmark."""

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.models.runners.bagel.bagel_runner import BagelRunner  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import seed_all


def build_config(task: str, model_path: str, config_json: str, seed: int):
    args = Namespace(
        model_cls="bagel",
        task=task,
        model_path=model_path,
        config_json=config_json,
        seed=seed,
        support_tasks=[],
        parallel=False,
        num_iterations=None,
        cpu_offload=True,
        offload_granularity="block",
    )
    return set_config(args)


def run_case(task: str, model_path: str, config_json: str, prompt: str, image_path: str, save_path: str, seed: int):
    config = build_config(task, model_path, config_json, seed)
    seed_all(seed)

    input_info = init_empty_input_info(task)
    payload = {
        "prompt": prompt,
        "save_result_path": save_path,
        "seed": seed,
    }
    if task == "i2i":
        payload["image_path"] = image_path
    if task == "t2i":
        payload["aspect_ratio"] = "1:1"
    update_input_info_from_dict(input_info, payload)

    runner = BagelRunner(config)
    torch.cuda.synchronize()
    t_load0 = time.perf_counter()
    runner.init_modules()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = runner.run_pipeline(input_info)
    torch.cuda.synchronize()
    pipeline_s = time.perf_counter() - t0

    out = Path(save_path)
    return {
        "task": task,
        "success": out.exists() or result.get("images") is not None,
        "output": str(out) if out.exists() else None,
        "load_latency_s": round(load_s, 3),
        "pipeline_latency_s": round(pipeline_s, 3),
        "total_latency_s": round(load_s + pipeline_s, 3),
        "infer_steps": config.get("infer_steps", 50),
        "image_shape": getattr(input_info, "image_shapes", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--t2i_config", default="configs/bagel/bagel_t2i.json")
    parser.add_argument("--i2i_config", default="configs/bagel/bagel_i2i.json")
    parser.add_argument("--image_path", default="assets/inputs/imgs/img_0.jpg")
    parser.add_argument("--out_dir", default="save_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task, cfg, prompt, out_name in [
        (
            "t2i",
            args.t2i_config,
            "A small cabin beside a lake at sunrise, cinematic lighting",
            "bagel_t2i_bench.png",
        ),
        (
            "i2i",
            args.i2i_config,
            "Change the scene to golden hour while preserving the main subject.",
            "bagel_i2i_bench.png",
        ),
    ]:
        print(f"\n=== Running BAGEL {task.upper()} ===", flush=True)
        try:
            r = run_case(
                task=task,
                model_path=args.model_path,
                config_json=cfg,
                prompt=prompt,
                image_path=args.image_path,
                save_path=str(out_dir / out_name),
                seed=42,
            )
        except Exception as exc:
            r = {"task": task, "success": False, "error": str(exc)}
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        results.append(r)

    summary_path = out_dir / "bagel_benchmark_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSummary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
