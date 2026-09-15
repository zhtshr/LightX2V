#!/usr/bin/env python3
"""Corrected overhead profile for PP×SP quad overlap (lps=2).

Uses per-slot critical-path timing (MAX across pipe_p per slot, summed over slots)
instead of cumulative per-rank timers that inflated barrier/P2P percentages.
"""

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
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

from scripts.disagg.pp_interleaved_pipeline import (
    PpTenantCtx,
    run_gpipe_dual_pipeline,
    theoretical_utilization,
)
from scripts.disagg.pp_sp_quad_overlap import QuadPpTenantCtx, run_gpipe_quad_sp_pipeline
from scripts.disagg.run_phase1_transformer_bench import (
    _bench_transformer_once,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)

_SLOT_KEYS = (
    "slot_critical_ms",
    "slot_compute_ms",
    "slot_meta_ms",
    "slot_pre_infer_ms",
    "slot_barrier_ms",
    "slot_noise_p2p_ms",
    "slot_activation_p2p_ms",
    "slot_orch_sync_ms",
)
_SUM_KEYS = ("n_busy_slots", "n_idle_slots", "stage_compute_ms", "post_infer_ms", "step_scheduler_ms")
_MAX_WORLD_KEYS = ("step_wall_ms",)
_MAX_PP_KEYS = _SLOT_KEYS


def _init_distributed(config: dict[str, Any]) -> None:
    import os

    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _reduce_profile(local: dict[str, float], pp_group) -> dict[str, float]:
    if not dist.is_initialized():
        return local
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    out: dict[str, float] = {}
    for key in _SUM_KEYS:
        t = torch.tensor([local.get(key, 0.0)], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[key] = float(t.item())
    for key in _MAX_WORLD_KEYS:
        t = torch.tensor([local.get(key, 0.0)], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        out[key] = float(t.item())
    for key in _MAX_PP_KEYS:
        t = torch.tensor([local.get(key, 0.0)], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pp_group)
        out[key] = float(t.item())
    return out


def _run_quad(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    lps: int,
    *,
    sp_overlap: bool,
    seq_parallel: bool,
) -> tuple[float, dict[str, float]]:
    scheds = [WanScheduler(runner.config) for _ in payloads[:4]]
    tenants = [QuadPpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads[:4])]
    profile: dict[str, float] = {}
    n_steps = int(runner.config["infer_steps"])

    _sync_device()
    wall_t0 = time.perf_counter()
    from scripts.disagg import pp_sp_quad_overlap as quad_mod

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    orch = None
    stats = None
    if sp_overlap and seq_parallel:
        orch = quad_mod.A2AOrchestrator(device)
        orch.install()
    try:
        for tenant, payload in zip(tenants, payloads):
            tenant.scheduler.prepare(
                seed=int(payload["seed"]),
                latent_shape=payload["latent_shape"],
                image_encoder_output=payload["image_encoder_output"],
            )
        for step_index in range(n_steps):
            t_sched = time.perf_counter()
            for tenant in tenants:
                tenant.scheduler.step_pre(step_index=step_index)
            quad_mod._profile_add(profile, "step_scheduler_ms", (time.perf_counter() - t_sched) * 1000.0)
            t_step = time.perf_counter()
            quad_mod.gpipe_quad_sp_pipeline_step(
                runner, tenants, pp_group, pp_rank, pp_size, num_layers, lps,
                orch, seq_parallel=seq_parallel, sp_overlap=sp_overlap, profile=profile,
            )
            if pp_rank == 0:
                t_post = time.perf_counter()
                for tenant in tenants:
                    tenant.scheduler.step_post()
                quad_mod._profile_add(profile, "step_scheduler_ms", (time.perf_counter() - t_post) * 1000.0)
            step_ms = (time.perf_counter() - t_step) * 1000.0
            st = torch.tensor([step_ms], device=device, dtype=torch.float64)
            dist.all_reduce(st, op=dist.ReduceOp.MAX)
            quad_mod._profile_add(profile, "step_wall_ms", float(st.item()))
        if orch is not None:
            stats = orch.stats_snapshot()
    finally:
        if orch is not None:
            orch.restore()
    _sync_device()
    wall = time.perf_counter() - wall_t0
    if stats and pp_rank == 0:
        for k, v in stats.items():
            profile[f"orch_{k}"] = float(v)
    return wall, profile


def _analyze(
    merged: dict[str, float],
    *,
    wall_s: float,
    single_s: float,
    num_stages: int,
    n_steps: int,
    n_infer_steps: int,
    theory_pp: float,
    ideal_4gpu_rps: float,
    mode: str,
) -> dict[str, Any]:
    n_gpus = dist.get_world_size() if dist.is_initialized() else 4
    slots_per_step = 2 + num_stages - 1  # m=2
    busy = merged.get("n_busy_slots", 0.0) / n_gpus
    idle = merged.get("n_idle_slots", 0.0) / n_gpus
    slot_total = busy + idle
    measured_slot_util = busy / slot_total if slot_total > 0 else 0.0

    step_wall_ms = merged.get("step_wall_ms", 0.0) / max(n_infer_steps, 1)
    slot_crit = merged.get("slot_critical_ms", 0.0) / max(n_infer_steps, 1)

    compute_ms = merged.get("slot_compute_ms", 0.0) / max(n_infer_steps, 1)
    orch_sync_ms = merged.get("slot_orch_sync_ms", 0.0) / max(n_infer_steps, 1)
    barrier_ms = merged.get("slot_barrier_ms", 0.0) / max(n_infer_steps, 1)
    parts = {
        "stage_compute_incl_sync": compute_ms,
        "orch_stream_sync_inside_compute": orch_sync_ms,
        "gpipe_barrier": barrier_ms,
        "metadata_path": merged.get("slot_meta_ms", 0.0) / max(n_infer_steps, 1),
        "pre_infer_in_meta": merged.get("slot_pre_infer_ms", 0.0) / max(n_infer_steps, 1),
        "activation_p2p": merged.get("slot_activation_p2p_ms", 0.0) / max(n_infer_steps, 1),
        "noise_p2p": merged.get("slot_noise_p2p_ms", 0.0) / max(n_infer_steps, 1),
        "scheduler_pre_post": merged.get("step_scheduler_ms", 0.0) / max(n_infer_steps, 1),
    }
    # Exclusive-ish: sync is inside compute cuda window
    parts["kernel_work_estimate"] = max(0.0, compute_ms - orch_sync_ms)

    def pct(x: float) -> float:
        return round(100.0 * x / slot_crit, 1) if slot_crit > 0 else 0.0

    bubble_slots_per_step = idle / max(n_infer_steps, 1)
    bubble_pct = 100.0 * bubble_slots_per_step / slots_per_step if slots_per_step else 0.0

    cluster_rps = 4.0 / wall_s if wall_s > 0 and mode.startswith("quad") else (
        2.0 / wall_s if mode == "pp_m2" else 1.0 / wall_s
    )

    return {
        "mode": mode,
        "wall_s": round(wall_s, 3),
        "cluster_throughput_rps": round(cluster_rps, 5),
        "throughput_vs_4x_ideal": round(cluster_rps / ideal_4gpu_rps, 3) if mode.startswith("quad") else None,
        "single_transformer_s": round(single_s, 3),
        "per_gpu_slot_util": round(measured_slot_util, 3),
        "theory_pp_slot_util": round(theory_pp, 3),
        "slot_util_gap_vs_theory": round(measured_slot_util - theory_pp, 3),
        "gpipe_bubble_slots_per_gpu_per_step": round(bubble_slots_per_step, 2),
        "gpipe_bubble_pct": round(bubble_pct, 1),
        "per_infer_step_ms": {
            "step_wall_max_gpu": round(step_wall_ms, 1),
            "slot_critical_path_sum": round(slot_crit, 1),
            "accounting_gap": round(step_wall_ms - slot_crit, 1),
        },
        "critical_path_breakdown_ms_per_step": {k: round(v, 1) for k, v in parts.items()},
        "critical_path_breakdown_pct": {
            "stage_compute_incl_sync": pct(parts["stage_compute_incl_sync"]),
            "orch_stream_sync_inside_compute": pct(parts["orch_stream_sync_inside_compute"]),
            "kernel_work_estimate": pct(parts["kernel_work_estimate"]),
            "gpipe_barrier": pct(parts["gpipe_barrier"]),
            "metadata_path": pct(parts["metadata_path"]),
            "pre_infer_in_meta": pct(parts["pre_infer_in_meta"]),
            "activation_p2p": pct(parts["activation_p2p"]),
            "noise_p2p": pct(parts["noise_p2p"]),
            "scheduler_pre_post": pct(parts["scheduler_pre_post"]),
        },
        "scaling_vs_single": {
            "naive_4x_serial_s": round(4 * single_s, 1),
            "speedup_vs_naive_4x": round(4 * single_s / wall_s, 3) if wall_s > 0 and mode.startswith("quad") else None,
            "per_request_effective_s": round(wall_s / 4, 2) if mode.startswith("quad") else None,
            "per_request_vs_single": round(single_s / (wall_s / 4), 3) if wall_s > 0 and mode.startswith("quad") else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--layers_per_stage", type=int, default=2)
    parser.add_argument("--single_gpu_baseline_s", type=float, default=72.67)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--base_seed", type=int, default=42)
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
    num_stages = pp_num_stages(num_layers, pp_size, lps)
    n_infer_steps = int(config["infer_steps"])
    theory_pp = theoretical_utilization(2, num_stages, pp_size)
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
    scheduler = WanScheduler(config)
    runner.set_scheduler(scheduler)

    if pp_rank == 0:
        print(f"Overhead profile lps={lps} stages={num_stages} theory_pp={theory_pp:.3f}")

    _sync_device()
    t0 = time.perf_counter()
    single_s = _bench_transformer_once(scheduler, runner, payload0)
    _sync_device()
    if pp_rank == 0:
        print(f"single: {single_s:.1f}s")

    results: dict[str, Any] = {}
    for mode, sp_overlap in (("quad_pp_sp", True), ("quad_pp_only", False)):
        wall, prof = _run_quad(
            runner, payloads, pp_group, pp_rank, pp_size, num_layers, lps,
            sp_overlap=sp_overlap, seq_parallel=seq_parallel,
        )
        merged = _reduce_profile(prof, pp_group)
        results[mode] = _analyze(
            merged, wall_s=wall, single_s=single_s, num_stages=num_stages,
            n_steps=4, n_infer_steps=n_infer_steps, theory_pp=theory_pp,
            ideal_4gpu_rps=ideal_4gpu, mode=mode,
        )
        if pp_rank == 0:
            b = results[mode]["critical_path_breakdown_pct"]
            print(f"\n{mode}: wall={wall:.1f}s slot_util={results[mode]['per_gpu_slot_util']:.3f}")
            print(f"  per-step critical path {results[mode]['per_infer_step_ms']['slot_critical_path_sum']:.0f}ms")
            print(f"  compute={b['stage_compute_incl_sync']:.1f}% sync_in_compute={b['orch_stream_sync_inside_compute']:.1f}% "
                  f"barrier={b['gpipe_barrier']:.1f}% act_p2p={b['activation_p2p']:.1f}%")

    # PP m=2 2-req reference (no slot instrumentation)
    scheds2 = [WanScheduler(runner.config) for _ in payloads[:2]]
    tenants2 = [PpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds2, payloads[:2])]
    _sync_device()
    t0 = time.perf_counter()
    run_gpipe_dual_pipeline(
        runner, tenants2, payloads[:2], pp_group, pp_rank, pp_size, num_layers, lps,
        seq_parallel=seq_parallel,
    )
    _sync_device()
    pp_m2_s = time.perf_counter() - t0
    results["pp_m2_2req"] = {
        "mode": "pp_m2_2req",
        "wall_s": round(pp_m2_s, 3),
        "cluster_throughput_rps": round(2.0 / pp_m2_s, 5),
        "per_column_2req_s": round(pp_m2_s, 3),
        "note": "one PP column, 2 req GPipe m=2, no SP cross-column overlap",
    }

    out = {
        "description": "Corrected quad overhead profile (lps=2, slot critical-path accounting)",
        "pp_layers_per_stage": lps,
        "num_stages": num_stages,
        "infer_steps": n_infer_steps,
        "theoretical_pp_slot_util_m2": round(theory_pp, 4),
        "single_transformer_s": round(single_s, 3),
        "results": results,
        "interpretation_notes": [
            "slot_critical_path_sum = sum over GPipe slots of MAX(rank times) per slot per infer step",
            "Percentages are vs slot critical path, not wall (avoids >100% cumulative bug)",
            "step_scheduler_ms is outside slot loop; gpipe_barrier is per-slot pipe_p barrier",
        ],
    }

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
