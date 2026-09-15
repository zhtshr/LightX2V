#!/usr/bin/env python3
"""Sweep pp_layers_per_stage: comm vs compute overlap + dual-tenant throughput."""

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
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.networks.wan.pp_utils import pp_num_stages
from scripts.disagg.pp_interleaved_pipeline import (
    PpTenantCtx,
    gpipe_dual_pipeline_step,
    run_gpipe_dual_pipeline,
    theoretical_utilization,
)
from scripts.disagg.run_phase1_transformer_bench import (
    _bench_transformer_once,
    _init_distributed,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)


def _load_config(args: argparse.Namespace, layers_per_stage: int | None) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        config_path=args.config_json,
    )
    config["parallel"] = {"pipe_p_size": 2}
    if layers_per_stage is not None:
        config["pp_layers_per_stage"] = layers_per_stage
    return config


def _profile_comm_compute(
    runner: MultiModelStruct,
    tenants: list[PpTenantCtx],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int | None,
) -> dict[str, float]:
    for tenant in tenants:
        tenant.pre = None
    profile: dict[str, float] = {}
    if pp_rank == 0:
        for tenant in tenants:
            tenant.scheduler.step_pre(0)
    else:
        for tenant in tenants:
            tenant.scheduler.step_index = 0
    if dist.is_initialized():
        dist.barrier(group=pp_group)
    gpipe_dual_pipeline_step(
        runner,
        tenants,
        pp_group,
        pp_rank,
        pp_size,
        num_layers,
        layers_per_stage,
        profile=profile,
    )
    if pp_rank == 0:
        for tenant in tenants:
            tenant.scheduler.step_post()
    local = dict(profile)
    if pp_rank == 0:
        gathered: list[dict] = [None, None]
        dist.gather_object(local, gathered, dst=0)
        merged = {"compute_ms": 0.0, "comm_ms": 0.0}
        for g in gathered:
            if not g:
                continue
            merged["compute_ms"] += g.get("compute_ms", 0.0)
            merged["comm_ms"] += g.get("comm_ms", 0.0)
        return merged
    dist.gather_object(local, None, dst=0)
    return {}


def _bench_dual_gpipe(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int | None,
) -> float:
    scheds = [WanScheduler(runner.config) for _ in payloads]
    tenants = [PpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads)]
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
    )
    _sync_device()
    return time.perf_counter() - t0


def _run_single_pp2(
    runner: MultiModelStruct,
    scheduler: WanScheduler,
    payload: dict,
) -> float:
    _sync_device()
    t0 = time.perf_counter()
    _bench_transformer_once(scheduler, runner, payload)
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--layers_per_stage_list", default="20,10,5,4,2,1")
    parser.add_argument("--single_gpu_baseline_s", type=float, default=72.67)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed_b", type=int, default=43)
    args = parser.parse_args()

    lps_list = [int(x) for x in args.layers_per_stage_list.split(",") if x.strip()]
    results: list[dict[str, Any]] = []

    for lps in lps_list:
        config = _load_config(args, lps)
        _init_distributed(config)
        pp_group = config["device_mesh"].get_group(mesh_dim="pipe_p")
        pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        num_layers = int(config["num_layers"])
        num_stages = pp_num_stages(num_layers, pp_size, lps)

        payload_a = _prepare_payload_on_device(
            _prepare_inputs_cache(
                config=config,
                cache_path=Path(args.inputs_cache),
                prompt="bench",
                image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
                seed=42,
            )
        )
        payload_b = copy.deepcopy(payload_a)
        payload_b["seed"] = args.seed_b

        runner = load_wan_transformer(config)
        assert isinstance(runner, MultiModelStruct)
        scheduler = WanScheduler(config)
        runner.set_scheduler(scheduler)

        if pp_rank == 0:
            print(f"\n=== pp_layers_per_stage={lps} num_stages={num_stages} ===")

        single_s = _run_single_pp2(runner, scheduler, payload_a)

        scheds = [WanScheduler(config), WanScheduler(config)]
        tenants = [
            PpTenantCtx(scheduler=scheds[0], inputs=payload_a["inputs"]),
            PpTenantCtx(scheduler=scheds[1], inputs=payload_b["inputs"]),
        ]
        scheds[0].prepare(
            seed=payload_a["seed"],
            latent_shape=payload_a["latent_shape"],
            image_encoder_output=payload_a["image_encoder_output"],
        )
        scheds[1].prepare(
            seed=payload_b["seed"],
            latent_shape=payload_b["latent_shape"],
            image_encoder_output=payload_b["image_encoder_output"],
        )

        prof = _profile_comm_compute(
            runner, tenants, pp_group, pp_rank, pp_size, num_layers, lps
        )

        dual_s = _bench_dual_gpipe(
            runner,
            [payload_a, payload_b],
            pp_group,
            pp_rank,
            pp_size,
            num_layers,
            lps,
        )

        m = 2
        theo_util = theoretical_utilization(m, num_stages, pp_size)
        single_per_gpu = 1.0 / single_s / pp_size
        dual_per_gpu = m / dual_s / pp_size
        baseline_per_gpu = 1.0 / args.single_gpu_baseline_s
        ratio_vs_single_gpu = dual_per_gpu / baseline_per_gpu
        ratio_vs_single_pp = dual_per_gpu / single_per_gpu

        compute_ms = prof.get("compute_ms", 0.0)
        comm_ms = prof.get("comm_ms", 0.0)
        comm_overlap_ok = compute_ms > comm_ms if comm_ms > 0 else True

        row = {
            "pp_layers_per_stage": lps,
            "num_stages": num_stages,
            "single_transformer_s": single_s,
            "dual_pipeline_2req_s": dual_s,
            "dual_per_gpu_req_per_s": dual_per_gpu,
            "single_pp_per_gpu_req_per_s": single_per_gpu,
            "ratio_vs_single_gpu_baseline": ratio_vs_single_gpu,
            "ratio_vs_single_pp_per_gpu": ratio_vs_single_pp,
            "theoretical_utilization_2gpu": theo_util,
            "theoretical_ratio_vs_single_gpu_m2": theo_util,
            "profile_one_step_compute_ms_rank0": compute_ms,
            "profile_one_step_comm_send_ms_rank0": comm_ms,
            "compute_gt_comm": comm_overlap_ok,
            "p2p_transfers_per_forward": num_stages - 1,
        }
        if pp_rank == 0:
            results.append(row)
            print(
                f"lps={lps} p={num_stages} single={single_s:.1f}s dual2={dual_s:.1f}s "
                f"per_gpu={dual_per_gpu:.4f} (theo={theo_util:.3f}) "
                f"vs1GPU={ratio_vs_single_gpu:.3f} compute={compute_ms:.0f}ms comm={comm_ms:.1f}ms"
            )

        if dist.is_initialized():
            dist.destroy_process_group()

    if results:
        out = {
            "description": "pp_layers_per_stage sweep: comm overlap + dual-tenant GPipe throughput",
            "single_gpu_baseline_s": args.single_gpu_baseline_s,
            "num_microbatches": 2,
            "pp_size": 2,
            "results": results,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
