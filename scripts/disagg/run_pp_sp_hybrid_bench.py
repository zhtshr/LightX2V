#!/usr/bin/env python3
"""Benchmark PP×SP hybrid (pipe_p=2, seq_p=2) vs PP-only and SP-only baselines."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.networks.wan.pp_utils import pp_num_stages
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

from scripts.disagg.pp_interleaved_pipeline import (
    PpTenantCtx,
    max_feasible_microbatches,
    run_gpipe_dual_pipeline,
    theoretical_utilization,
)
from scripts.disagg.run_phase1_transformer_bench import (
    _bench_transformer_once,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)


def _init_distributed(config: dict[str, Any]) -> None:
    import os

    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        config_path=args.config_json,
    )
    if args.layers_per_stage is not None:
        config["pp_layers_per_stage"] = int(args.layers_per_stage)
    return config


def _bench_single(runner: MultiModelStruct, scheduler: WanScheduler, payload: dict) -> float:
    _sync_device()
    t0 = time.perf_counter()
    _bench_transformer_once(scheduler, runner, payload)
    _sync_device()
    return time.perf_counter() - t0


def _bench_gpipe(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int,
    *,
    seq_parallel: bool,
    measure_per_request_latency: bool = False,
) -> tuple[float, dict[int, float] | None]:
    scheds = [WanScheduler(runner.config) for _ in payloads]
    tenants = [PpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads)]
    latencies: dict[int, float] | None = {} if measure_per_request_latency else None
    _sync_device()
    t0 = time.perf_counter()
    run_gpipe_dual_pipeline(
        runner,
        tenants,
        payloads,
        pp_group,
        pp_rank,
        pp_size,
        num_layers,
        layers_per_stage,
        seq_parallel=seq_parallel,
        per_request_latency_s=latencies,
    )
    _sync_device()
    wall_s = time.perf_counter() - t0
    if latencies is not None and pp_rank != 0:
        latencies = None
    return wall_s, latencies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--layers_per_stage", type=int, default=None)
    parser.add_argument("--microbatch_list", default="2")
    parser.add_argument("--per-request-latency", action="store_true", default=False)
    parser.add_argument("--single_gpu_baseline_s", type=float, default=72.67)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    config = _load_config(args)
    seed_all(args.base_seed)
    _init_distributed(config)

    pp_group = config["device_mesh"].get_group(mesh_dim="pipe_p")
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    seq_parallel = bool(config.get("seq_parallel"))
    seq_p_size = int(config["parallel"]["seq_p_size"])
    n_gpu = pp_size * seq_p_size
    num_layers = int(config["num_layers"])
    lps = config.get("pp_layers_per_stage")
    num_stages = pp_num_stages(num_layers, pp_size, lps)
    max_m = max_feasible_microbatches(num_stages, pp_size)

    cache_path = Path(args.inputs_cache)
    payload0 = _prepare_payload_on_device(
        _prepare_inputs_cache(
            config=config,
            cache_path=cache_path,
            prompt="bench",
            image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
            seed=args.base_seed,
        )
    )
    microbatch_list = [int(x) for x in args.microbatch_list.split(",") if x.strip()]

    runner = load_wan_transformer(config)
    assert isinstance(runner, MultiModelStruct)
    scheduler = WanScheduler(config)
    runner.set_scheduler(scheduler)

    if pp_rank == 0:
        print(
            f"PP×SP hybrid: pp={pp_size} seq_p={config['parallel']['seq_p_size']} "
            f"lps={lps} stages={num_stages} seq_parallel={seq_parallel} max_feasible_m={max_m}"
        )

    single_s = _bench_single(runner, scheduler, payload0)

    pipeline_results: list[dict[str, Any]] = []
    for m in microbatch_list:
        if m > max_m:
            if is_main_process():
                print(f"skip m={m}: infeasible for GPipe (need m<={max_m} with {num_stages} stages on {pp_size} PP ranks)")
            pipeline_results.append(
                {
                    "num_microbatches": m,
                    "feasible": False,
                    "max_feasible_microbatches": max_m,
                    "reason": f"m*S={m * num_stages} > pp_size*(m+S-1)={pp_size * (m + num_stages - 1)}",
                }
            )
            continue

        payloads = []
        for i in range(m):
            p = copy.deepcopy(payload0)
            p["seed"] = args.base_seed + i
            payloads.append(p)

        wall_s, per_req_lat = _bench_gpipe(
            runner,
            payloads,
            pp_group,
            pp_rank,
            pp_size,
            num_layers,
            lps,
            seq_parallel=seq_parallel,
            measure_per_request_latency=args.per_request_latency and m == 2,
        )

        theory = theoretical_utilization(m, num_stages, pp_size)
        cluster_rps = m / wall_s if wall_s > 0 else 0.0
        per_gpu_rps = cluster_rps / n_gpu if n_gpu else 0.0
        ideal_rps = n_gpu / args.single_gpu_baseline_s
        row: dict[str, Any] = {
            "num_microbatches": m,
            "feasible": True,
            "gpipe_wall_s": round(wall_s, 3),
            "cluster_throughput_rps": round(cluster_rps, 5),
            "per_gpu_throughput_rps": round(per_gpu_rps, 5),
            f"ratio_vs_{n_gpu}x_single_gpu_ideal": round(cluster_rps / ideal_rps, 3),
            "theoretical_stage_utilization": round(theory, 3),
            "naive_b2b_s_estimate": round(m * single_s, 3),
        }
        if per_req_lat:
            row["per_request_latency_s"] = {str(k): round(v, 3) for k, v in sorted(per_req_lat.items())}
            vals = list(per_req_lat.values())
            row["per_request_latency_summary"] = {
                "min_s": round(min(vals), 3),
                "max_s": round(max(vals), 3),
                "mean_s": round(sum(vals) / len(vals), 3),
                "wall_s": round(wall_s, 3),
                "latency_vs_single_ratio": round(max(vals) / single_s, 3) if single_s > 0 else None,
            }
        pipeline_results.append(row)
        if pp_rank == 0:
            ratio = cluster_rps / ideal_rps
            print(
                f"m={m}: wall={wall_s:.1f}s rps={cluster_rps:.4f} "
                f"vs {n_gpu}×P1 ideal={ideal_rps:.4f} ({100 * ratio:.1f}%) "
                f"theory_util={theory:.3f}"
            )
            if per_req_lat:
                s = row["per_request_latency_summary"]
                print(f"  per-request latency: min={s['min_s']:.1f}s max={s['max_s']:.1f}s")

    ideal_rps = n_gpu / args.single_gpu_baseline_s
    out = {
        "description": f"PP×SP hybrid bench (pipe_p={pp_size}, seq_p={seq_p_size}, GPipe m=2, no SP overlap)",
        "config_json": args.config_json,
        "pp_size": pp_size,
        "seq_p_size": seq_p_size,
        "num_gpus": n_gpu,
        "pp_layers_per_stage": lps,
        "num_stages": num_stages,
        "seq_parallel": seq_parallel,
        "single_gpu_baseline_s": args.single_gpu_baseline_s,
        "single_transformer_s": round(single_s, 3),
        f"ideal_{n_gpu}gpu_rps": round(ideal_rps, 5),
        "max_feasible_microbatches": max_m,
        "gpipe_results": pipeline_results,
        "references": {
            "sp4_dual_overlap_rps": 0.0396,
            "pp2_lps2_dual_rps": 0.0260,
        },
    }

    if is_main_process():
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
