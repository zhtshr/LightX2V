#!/usr/bin/env python3
"""Benchmark PP×SP with 4 requests: GPipe (PP) + optional SP a2a overlap."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

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
from scripts.disagg.pp_sp_quad_overlap import (
    QuadPpTenantCtx,
    QuadTenantLayout,
    num_quad_tenants,
    run_gpipe_quad_sp_pipeline,
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


def _bench_pp_gpipe_m2(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    lps: int,
    *,
    seq_parallel: bool,
) -> float:
    scheds = [WanScheduler(runner.config) for _ in payloads[:2]]
    tenants = [PpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads[:2])]
    _sync_device()
    t0 = time.perf_counter()
    run_gpipe_dual_pipeline(
        runner,
        tenants,
        payloads[:2],
        pp_group,
        pp_rank,
        pp_size,
        num_layers,
        lps,
        seq_parallel=seq_parallel,
    )
    _sync_device()
    return time.perf_counter() - t0


def _bench_quad(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    lps: int,
    *,
    seq_p_size: int,
    layout: QuadTenantLayout,
    seq_parallel: bool,
    sp_overlap: bool,
    minimal_step_sync: bool = False,
    measure_per_request_latency: bool = False,
) -> tuple[float, dict[str, int] | None, dict[int, float] | None]:
    n_req = num_quad_tenants(seq_p_size, layout)
    scheds = [WanScheduler(runner.config) for _ in payloads[:n_req]]
    tenants = [QuadPpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads[:n_req])]
    latencies: dict[int, float] | None = {} if measure_per_request_latency else None
    _sync_device()
    t0 = time.perf_counter()
    stats = run_gpipe_quad_sp_pipeline(
        runner,
        tenants,
        payloads[:n_req],
        pp_group,
        pp_rank,
        pp_size,
        num_layers,
        lps,
        seq_p_size=seq_p_size,
        layout=layout,
        seq_parallel=seq_parallel,
        sp_overlap=sp_overlap,
        minimal_step_sync=minimal_step_sync,
        per_request_latency_s=latencies,
    )
    _sync_device()
    wall_s = time.perf_counter() - t0
    if latencies is not None and pp_rank != 0:
        latencies = None
    return wall_s, stats, latencies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--layers_per_stage", type=int, default=2)
    parser.add_argument("--minimal-step-sync", action="store_true", default=False)
    parser.add_argument("--legacy-step-sync", action="store_true", default=False)
    parser.add_argument("--quad-only", action="store_true", default=False)
    parser.add_argument("--quad-layout", choices=("oct", "dual"), default="oct")
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
    seq_p_size = int(config["parallel"].get("seq_p_size", 1))
    num_layers = int(config["num_layers"])
    lps = config.get("pp_layers_per_stage")
    layout: QuadTenantLayout = args.quad_layout
    if layout == "oct" and seq_p_size >= 4:
        layout = "dual"
    n_quad = num_quad_tenants(seq_p_size, layout)
    n_gpu = pp_size * seq_p_size
    minimal_step_sync = bool(args.minimal_step_sync and not args.legacy_step_sync)
    num_stages = pp_num_stages(num_layers, pp_size, lps)
    max_m = max_feasible_microbatches(num_stages, pp_size)

    payload0 = _prepare_payload_on_device(
        _prepare_inputs_cache(
            config=config,
            cache_path=Path(args.inputs_cache),
            prompt="bench",
            image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
            seed=args.base_seed,
        )
    )
    payloads = []
    for i in range(n_quad):
        p = copy.deepcopy(payload0)
        p["seed"] = args.base_seed + i
        payloads.append(p)

    runner = load_wan_transformer(config)
    assert isinstance(runner, MultiModelStruct)
    scheduler = WanScheduler(config)
    runner.set_scheduler(scheduler)

    if pp_rank == 0:
        print(
            f"PP×SP quad ({layout}): lps={lps} stages={num_stages} seq_p={seq_p_size} n_req={n_quad} "
            f"seq_parallel={seq_parallel} pp_gpipe_max_m={max_m} minimal_step_sync={minimal_step_sync}"
        )

    single_s = None
    if not args.quad_only:
        single_s = _bench_single(runner, scheduler, payload0)
    ideal_rps = float(n_gpu) / args.single_gpu_baseline_s

    pp_m2_s = None
    quad_serial_s = None
    if not args.quad_only:
        pp_m2_s = _bench_pp_gpipe_m2(
            runner, payloads, pp_group, pp_rank, pp_size, num_layers, lps, seq_parallel=seq_parallel,
        )
        quad_serial_s, _, _ = _bench_quad(
            runner, payloads, pp_group, pp_rank, pp_size, num_layers, lps,
            seq_p_size=seq_p_size, layout=layout, seq_parallel=seq_parallel, sp_overlap=False,
            minimal_step_sync=minimal_step_sync,
        )
    quad_overlap_s, overlap_stats, per_req_lat = _bench_quad(
        runner, payloads, pp_group, pp_rank, pp_size, num_layers, lps,
        seq_p_size=seq_p_size, layout=layout, seq_parallel=seq_parallel, sp_overlap=True,
        minimal_step_sync=minimal_step_sync,
        measure_per_request_latency=args.per_request_latency,
    )

    theory_pp = theoretical_utilization(2, num_stages, pp_size)

    def _row(wall_s: float, n_req: int) -> dict[str, Any]:
        rps = n_req / wall_s if wall_s > 0 else 0.0
        return {
            "wall_s": round(wall_s, 3),
            "cluster_throughput_rps": round(rps, 5),
            f"ratio_vs_{n_gpu}x_single_gpu_ideal": round(rps / ideal_rps, 3),
        }

    pair_note = (
        f"{n_quad} requests, GPipe + SP a2a overlap, layout={layout} "
        f"({seq_p_size // 2 if layout == 'oct' else 1} pair(s)/mb)"
    )
    out = {
        "description": f"PP×SP quad bench ({layout}): {n_quad} requests, GPipe m=2 + optional SP a2a overlap",
        "quad_layout": layout,
        "pp_layers_per_stage": lps,
        "num_stages": num_stages,
        "seq_p_size": seq_p_size,
        "num_quad_requests": n_quad,
        "num_gpus": n_gpu,
        "seq_parallel": seq_parallel,
        "minimal_step_sync": minimal_step_sync,
        f"ideal_{n_gpu}gpu_rps": round(ideal_rps, 5),
        "theoretical_pp_stage_util_m2": round(theory_pp, 3),
        "modes": {
            "quad_pp_sp_overlap": {
                **_row(quad_overlap_s, n_quad),
                "note": pair_note,
                "a2a_overlap_stats": overlap_stats,
            },
        },
    }
    if single_s is not None:
        out["single_transformer_s"] = round(single_s, 3)
        out["modes"]["single_request"] = _row(single_s, 1)
        out["modes"]["pp_gpipe_m2_2req"] = {**_row(pp_m2_s, 2), "note": "existing PP-only GPipe overlap"}
        out["modes"]["quad_pp_only"] = {
            **_row(quad_serial_s, n_quad),
            "note": f"{n_quad} requests, GPipe m=2 without SP a2a overlap",
        }
        out[f"naive_{n_quad}x_serial_s"] = round(n_quad * single_s, 3)

    if per_req_lat:
        out["per_request_latency_s"] = {str(k): round(v, 3) for k, v in sorted(per_req_lat.items())}
        vals = list(per_req_lat.values())
        out["per_request_latency_summary"] = {
            "min_s": round(min(vals), 3),
            "max_s": round(max(vals), 3),
            "mean_s": round(sum(vals) / len(vals), 3),
            "wall_s": round(quad_overlap_s, 3),
            "throughput_amortized_s": round(quad_overlap_s / n_quad, 3),
        }
        if single_s is not None:
            out["per_request_latency_summary"]["single_request_s"] = round(single_s, 3)
            out["per_request_latency_summary"]["latency_vs_single_ratio"] = round(
                max(vals) / single_s, 3,
            )

    if pp_rank == 0:
        m = out["modes"]
        q = m["quad_pp_sp_overlap"]
        ratio_key = f"ratio_vs_{n_gpu}x_single_gpu_ideal"
        print(
            f"quad pp+sp ({n_quad} req): {quad_overlap_s:.1f}s "
            f"rps={q['cluster_throughput_rps']:.4f} "
            f"({100 * q[ratio_key]:.1f}% ideal)"
        )
        if per_req_lat:
            s = out["per_request_latency_summary"]
            print(
                f"per-request latency: min={s['min_s']:.1f}s max={s['max_s']:.1f}s "
                f"(amortized {s['throughput_amortized_s']:.1f}s ≠ latency)"
            )
        if single_s is not None:
            print(f"single: {single_s:.1f}s")
            print(f"pp m=2 (2 req): {pp_m2_s:.1f}s rps={m['pp_gpipe_m2_2req']['cluster_throughput_rps']:.4f}")
            print(
                f"quad pp-only ({n_quad} req): {quad_serial_s:.1f}s "
                f"rps={m['quad_pp_only']['cluster_throughput_rps']:.4f} "
                f"({100 * m['quad_pp_only'][ratio_key]:.1f}% ideal)"
            )

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
