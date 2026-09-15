#!/usr/bin/env python3
"""PP=2 pipeline throughput benchmark (no CPU offload).

Modes:
  single          - one request transformer denoise loop
  dual_b2b        - two requests back-to-back on the same PP group
  dual_pipeline   - two requests with stage overlap (rank0 B || rank1 A)
  dual_serial     - two requests same step, no stage overlap (sanity)
  all             - run single + dual_b2b + dual_pipeline (+ dual_serial)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Reuse phase1 input cache helpers
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from scripts.disagg.pp2_dual_pipeline import PpTenantCtx, run_pp2_dual_pipeline
from scripts.disagg.run_phase1_transformer_bench import (  # noqa: E402
    _bench_transformer_once,
    _init_distributed,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        config_path=args.config_json,
    )
    cfg_path = Path(args.config_json)
    cfg_parallel = {}
    if cfg_path.is_file():
        cfg_parallel = json.loads(cfg_path.read_text(encoding="utf-8")).get("parallel", {}) or {}

    pipe_p = int(args.pipe_p_size or cfg_parallel.get("pipe_p_size", 0) or 0)
    if pipe_p > 1:
        config["parallel"] = {"pipe_p_size": pipe_p}
    elif cfg_parallel.get("pipe_p_size", 1) > 1:
        config["parallel"] = dict(cfg_parallel)
    else:
        config["parallel"] = False
    return config


def _run_transformer_compute(scheduler: WanScheduler, model: Any, payload: dict[str, Any]) -> None:
    seed = payload["seed"]
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]
    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    infer_steps = scheduler.infer_steps
    for step_index in range(infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()
    _sync_device()


def _bench_dual_b2b(scheduler: WanScheduler, model: Any, payload_a: dict, payload_b: dict) -> float:
    _sync_device()
    start = time.perf_counter()
    _run_transformer_compute(scheduler, model, payload_a)
    _run_transformer_compute(scheduler, model, payload_b)
    return time.perf_counter() - start


def _bench_dual_pipeline(
    runner: MultiModelStruct,
    payload_a: dict,
    payload_b: dict,
    pp_group,
    pp_rank: int,
    pp_size: int,
    *,
    overlap: bool,
) -> float:
    sched_a = WanScheduler(runner.config)
    sched_b = WanScheduler(runner.config)
    tenant_a = PpTenantCtx(scheduler=sched_a, inputs=payload_a["inputs"])
    tenant_b = PpTenantCtx(scheduler=sched_b, inputs=payload_b["inputs"])
    _sync_device()
    start = time.perf_counter()
    run_pp2_dual_pipeline(
        runner,
        tenant_a,
        tenant_b,
        payload_a,
        payload_b,
        pp_group,
        pp_rank,
        pp_size,
        overlap=overlap,
    )
    _sync_device()
    return time.perf_counter() - start


def _per_gpu_rps(total_req: int, wall_s: float, num_gpus: int) -> float | None:
    if wall_s <= 0:
        return None
    return (total_req / wall_s) / num_gpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    parser.add_argument("--prompt", default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    parser.add_argument("--pipe_p_size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--mode", choices=("single", "dual_b2b", "dual_pipeline", "dual_serial", "all"), default="all")
    parser.add_argument("--single_gpu_baseline_s", type=float, default=72.67, help="Reference single-GPU transformer_compute_s")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    args = parser.parse_args()

    config = _load_config(args)
    seed_all(args.seed)
    _init_distributed(config)

    payload_a = _prepare_payload_on_device(
        _prepare_inputs_cache(
            config=config,
            cache_path=Path(args.inputs_cache),
            prompt=args.prompt,
            image_path=args.image_path,
            seed=args.seed,
            task=args.task,
            negative_prompt=args.negative_prompt,
        )
    )
    import copy

    payload_b = copy.deepcopy(payload_a)
    payload_b["seed"] = args.seed_b

    load_start = time.perf_counter()
    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)
    model_load_s = time.perf_counter() - load_start
    if not isinstance(model, MultiModelStruct):
        raise TypeError("PP=2 MoE bench expects MultiModelStruct")
    pp_group = config["device_mesh"].get_group(mesh_dim="pipe_p")
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    if is_main_process():
        print(f"Transformer load time (excluded): {model_load_s:.3f}s")

    result: dict[str, Any] = {
        "parallel_mode": "pipe_p",
        "pipe_p_size": config.get("pp_size", args.pipe_p_size),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "model_load_s_excluded": model_load_s,
        "config_json": args.config_json,
        "cpu_offload": config.get("cpu_offload", False),
        "target_height": config.get("target_height"),
        "target_width": config.get("target_width"),
    }

    if args.mode in ("single", "all"):
        for _ in range(args.warmup):
            _bench_transformer_once(scheduler, model, payload_a)
        single_samples = [_bench_transformer_once(scheduler, model, payload_a) for _ in range(args.measure_iters)]
        single_avg = sum(single_samples) / len(single_samples)
        result["single_transformer_compute_s"] = single_avg
        result["single_samples_s"] = single_samples
        result["single_req_per_s"] = 1.0 / single_avg if single_avg > 0 else None
        result["single_per_gpu_req_per_s"] = _per_gpu_rps(1, single_avg, pp_size)
        if is_main_process():
            print(f"PP=2 single: {single_avg:.3f}s ({1.0/single_avg:.4f} req/s, per-GPU {_per_gpu_rps(1, single_avg, pp_size):.4f})")

    if args.mode in ("dual_b2b", "all"):
        for _ in range(args.warmup):
            _bench_dual_b2b(scheduler, model, payload_a, payload_b)
        dual_samples = [_bench_dual_b2b(scheduler, model, payload_a, payload_b) for _ in range(args.measure_iters)]
        dual_avg = sum(dual_samples) / len(dual_samples)
        result["dual_b2b_transformer_compute_s"] = dual_avg
        result["dual_b2b_samples_s"] = dual_samples
        result["dual_b2b_req_per_s"] = 2.0 / dual_avg if dual_avg > 0 else None
        result["dual_b2b_per_gpu_req_per_s"] = _per_gpu_rps(2, dual_avg, pp_size)
        if is_main_process():
            print(
                f"PP=2 dual b2b (2 req): {dual_avg:.3f}s "
                f"({2.0/dual_avg:.4f} req/s, per-GPU {_per_gpu_rps(2, dual_avg, pp_size):.4f})"
            )

    if args.mode in ("dual_pipeline", "all"):
        for _ in range(args.warmup):
            _bench_dual_pipeline(model, payload_a, payload_b, pp_group, pp_rank, pp_size, overlap=True)
        pipe_samples = [
            _bench_dual_pipeline(model, payload_a, payload_b, pp_group, pp_rank, pp_size, overlap=True)
            for _ in range(args.measure_iters)
        ]
        pipe_avg = sum(pipe_samples) / len(pipe_samples)
        result["dual_pipeline_transformer_compute_s"] = pipe_avg
        result["dual_pipeline_samples_s"] = pipe_samples
        result["dual_pipeline_req_per_s"] = 2.0 / pipe_avg if pipe_avg > 0 else None
        result["dual_pipeline_per_gpu_req_per_s"] = _per_gpu_rps(2, pipe_avg, pp_size)
        if is_main_process():
            print(
                f"PP=2 dual pipeline overlap (2 req): {pipe_avg:.3f}s "
                f"({2.0/pipe_avg:.4f} req/s, per-GPU {_per_gpu_rps(2, pipe_avg, pp_size):.4f})"
            )

    if args.mode in ("dual_serial", "all"):
        for _ in range(args.warmup):
            _bench_dual_pipeline(model, payload_a, payload_b, pp_group, pp_rank, pp_size, overlap=False)
        serial_samples = [
            _bench_dual_pipeline(model, payload_a, payload_b, pp_group, pp_rank, pp_size, overlap=False)
            for _ in range(args.measure_iters)
        ]
        serial_avg = sum(serial_samples) / len(serial_samples)
        result["dual_serial_transformer_compute_s"] = serial_avg
        result["dual_serial_samples_s"] = serial_samples
        result["dual_serial_req_per_s"] = 2.0 / serial_avg if serial_avg > 0 else None
        result["dual_serial_per_gpu_req_per_s"] = _per_gpu_rps(2, serial_avg, pp_size)
        if is_main_process():
            print(
                f"PP=2 dual same-step serial (2 req): {serial_avg:.3f}s "
                f"({2.0/serial_avg:.4f} req/s, per-GPU {_per_gpu_rps(2, serial_avg, pp_size):.4f})"
            )

    baseline_s = args.single_gpu_baseline_s
    result["single_gpu_baseline_transformer_s"] = baseline_s
    result["single_gpu_baseline_per_gpu_req_per_s"] = 1.0 / baseline_s if baseline_s > 0 else None
    pipe_per_gpu = result.get("dual_pipeline_per_gpu_req_per_s")
    if pipe_per_gpu and baseline_s > 0:
        result["dual_pipeline_vs_single_gpu_per_gpu_ratio"] = pipe_per_gpu / (1.0 / baseline_s)
    b2b_per_gpu = result.get("dual_b2b_per_gpu_req_per_s")
    if pipe_per_gpu and b2b_per_gpu:
        result["dual_pipeline_vs_dual_b2b_per_gpu_ratio"] = pipe_per_gpu / b2b_per_gpu

    if is_main_process():
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
