#!/usr/bin/env python3
"""Sweep quad pp+sp optimizations: defer orch sync, slot barrier, async activation P2P."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

from scripts.disagg.pp_sp_quad_overlap import QuadPipelineOpts, QuadPpTenantCtx, run_gpipe_quad_sp_pipeline
from scripts.disagg.run_phase1_transformer_bench import (
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


def _bench_quad(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    lps: int,
    *,
    seq_parallel: bool,
    pipeline_opts: QuadPipelineOpts,
) -> float:
    scheds = [WanScheduler(runner.config) for _ in payloads[:4]]
    tenants = [QuadPpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads[:4])]
    _sync_device()
    t0 = time.perf_counter()
    run_gpipe_quad_sp_pipeline(
        runner,
        tenants,
        payloads[:4],
        pp_group,
        pp_rank,
        pp_size,
        num_layers,
        lps,
        seq_parallel=seq_parallel,
        sp_overlap=True,
        minimal_step_sync=True,
        pipeline_opts=pipeline_opts,
    )
    _sync_device()
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--layers_per_stage", type=int, default=2)
    parser.add_argument("--single_gpu_baseline_s", type=float, default=72.67)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--only", default="", help="comma-separated variant names to run")
    args = parser.parse_args()

    config = set_config(
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=args.config_json,
    )
    config["pp_layers_per_stage"] = args.layers_per_stage
    seed_all(args.base_seed)
    _init_distributed(config)

    pp_group = config["device_mesh"].get_group(mesh_dim="pipe_p")
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    seq_parallel = bool(config.get("seq_parallel"))
    num_layers = int(config["num_layers"])
    lps = int(config["pp_layers_per_stage"])
    ideal_4gpu = 4.0 / args.single_gpu_baseline_s

    payload0 = _prepare_payload_on_device(
        _prepare_inputs_cache(
            config=config,
            cache_path=Path("save_results/optimization_study/phase1_encoder_inputs.pt"),
            prompt="bench",
            image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
            seed=args.base_seed,
        )
    )
    payloads = []
    for i in range(4):
        p = copy.deepcopy(payload0)
        p["seed"] = args.base_seed + i
        payloads.append(p)

    runner = load_wan_transformer(config)
    assert isinstance(runner, MultiModelStruct)

    variants: list[tuple[str, QuadPipelineOpts]] = [
        ("baseline", QuadPipelineOpts()),
        ("defer_orch_sync", QuadPipelineOpts(defer_orch_stream_sync=True)),
        ("orch_sync_per_step", QuadPipelineOpts(orch_sync_per_step=True, defer_orch_stream_sync=True)),
        ("bubble_barrier", QuadPipelineOpts(slot_barrier_mode="bubble_only")),
        ("no_slot_barrier", QuadPipelineOpts(slot_barrier_mode="none")),
        ("barrier_after_p2p", QuadPipelineOpts(barrier_after_p2p=True)),
        ("async_act_p2p", QuadPipelineOpts(async_activation_p2p=True)),
        (
            "best_combo",
            QuadPipelineOpts(
                defer_orch_stream_sync=True,
                orch_sync_per_step=True,
                slot_barrier_mode="bubble_only",
                async_activation_p2p=True,
            ),
        ),
    ]

    if args.only.strip():
        only = {x.strip() for x in args.only.split(",") if x.strip()}
        variants = [(n, o) for n, o in variants if n in only]

    results: dict[str, Any] = {}
    baseline_wall: float | None = None

    for name, opts in variants:
        wall = _bench_quad(
            runner, payloads, pp_group, pp_rank, pp_size, num_layers, lps,
            seq_parallel=seq_parallel, pipeline_opts=opts,
        )
        rps = 4.0 / wall if wall > 0 else 0.0
        row = {
            "wall_s": round(wall, 3),
            "cluster_throughput_rps": round(rps, 5),
            "ratio_vs_4x_ideal": round(rps / ideal_4gpu, 3),
            "opts": {
                "defer_orch_stream_sync": opts.defer_orch_stream_sync,
                "slot_barrier_mode": opts.slot_barrier_mode,
                "async_activation_p2p": opts.async_activation_p2p,
            },
        }
        if baseline_wall is None:
            baseline_wall = wall
            row["speedup_vs_baseline"] = 1.0
        else:
            row["speedup_vs_baseline"] = round(baseline_wall / wall, 4)
            row["saved_s_vs_baseline"] = round(baseline_wall - wall, 3)
        results[name] = row
        if pp_rank == 0:
            print(
                f"{name}: {wall:.1f}s rps={rps:.4f} "
                f"({100 * row['ratio_vs_4x_ideal']:.1f}% ideal) "
                f"speedup={row['speedup_vs_baseline']:.3f}x"
            )

    out = {
        "description": "Quad pp+sp optimization sweep (lps=2, 4 req)",
        "pp_layers_per_stage": lps,
        "ideal_4gpu_rps": round(ideal_4gpu, 5),
        "variants": results,
    }

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
