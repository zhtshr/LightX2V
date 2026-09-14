#!/usr/bin/env python3
"""Profile PP×SP quad overlap: slot utilization, compute/comm/barrier breakdown, GPU util."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
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

from scripts.disagg.pp_interleaved_pipeline import theoretical_utilization
from scripts.disagg.pp_sp_quad_overlap import QuadPpTenantCtx, run_gpipe_quad_sp_pipeline
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


_PROFILE_KEYS = (
    "busy_slots",
    "idle_slots",
    "stage_compute_ms",
    "metadata_ms",
    "activation_p2p_ms",
    "noise_p2p_ms",
    "barrier_ms",
    "step_barrier_ms",
    "post_infer_ms",
    "orch_a2a_calls",
    "orch_a2a_overlap_windows",
    "orch_overlap_windows",
    "orch_all_gather_calls",
)


_SYNC_KEYS = frozenset({"barrier_ms", "step_barrier_ms", "metadata_ms", "activation_p2p_ms", "noise_p2p_ms"})


def _reduce_profiles(local: dict[str, float]) -> dict[str, float]:
    if not dist.is_initialized():
        return local
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    merged: dict[str, float] = {}
    for key in _PROFILE_KEYS:
        val = torch.tensor([local.get(key, 0.0)], device=device, dtype=torch.float64)
        if key in _SYNC_KEYS:
            dist.all_reduce(val, op=dist.ReduceOp.MAX)
        else:
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
        merged[key] = float(val.item())
    return merged

def _profile_full_run(
    runner: MultiModelStruct,
    payloads: list[dict],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    lps: int,
    *,
    seq_parallel: bool,
    sp_overlap: bool,
) -> tuple[float, dict[str, float]]:
    scheds = [WanScheduler(runner.config) for _ in payloads[:4]]
    tenants = [QuadPpTenantCtx(scheduler=s, inputs=p["inputs"]) for s, p in zip(scheds, payloads[:4])]
    profile: dict[str, float] = {}
    _sync_device()
    t0 = time.perf_counter()
    stats = run_gpipe_quad_sp_pipeline(
        runner,
        tenants,
        payloads[:4],
        pp_group,
        pp_rank,
        pp_size,
        num_layers,
        lps,
        seq_parallel=seq_parallel,
        sp_overlap=sp_overlap,
        profile=profile,
    )
    _sync_device()
    wall = time.perf_counter() - t0
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
    ideal_4gpu_rps: float,
    theory_pp_util: float,
) -> dict[str, Any]:
    busy = merged.get("busy_slots", 0.0)
    idle = merged.get("idle_slots", 0.0)
    slot_total = busy + idle
    measured_pp_slot_util = busy / slot_total if slot_total > 0 else 0.0

    stage_compute = merged.get("stage_compute_ms", 0.0)
    metadata = merged.get("metadata_ms", 0.0)
    post_infer = merged.get("post_infer_ms", 0.0)
    act_p2p = merged.get("activation_p2p_ms", 0.0)
    noise_p2p = merged.get("noise_p2p_ms", 0.0)
    barrier = merged.get("barrier_ms", 0.0) + merged.get("step_barrier_ms", 0.0)

    n_gpus = dist.get_world_size() if dist.is_initialized() else 4
    wall_ms = wall_s * 1000.0
    # stage_compute is summed GPU-ms; divide by n_gpus*wall for avg device busy fraction
    avg_gpu_compute_util = stage_compute / (n_gpus * wall_ms) if wall_ms > 0 else 0.0
    pp_p2p_ms = metadata + act_p2p + noise_p2p

    cluster_rps = 4.0 / wall_s if wall_s > 0 else 0.0
    throughput_vs_ideal = cluster_rps / ideal_4gpu_rps if ideal_4gpu_rps > 0 else 0.0

    naive_4_serial_s = 4 * single_s
    # PP-only floor: if PP slots were perfect (theory util), wall scales by measured/theory
    pp_bubble_inflation = measured_pp_slot_util / theory_pp_util if theory_pp_util > 0 else 1.0
    est_pp_only_floor_s = wall_s / pp_bubble_inflation if pp_bubble_inflation > 0 else wall_s

    orch_a2a = merged.get("orch_a2a_calls", 0.0)
    orch_overlap = merged.get("orch_a2a_overlap_windows", 0.0)
    sp_overlap_frac = orch_overlap / orch_a2a if orch_a2a > 0 else 0.0

    return {
        "wall_s": round(wall_s, 3),
        "cluster_throughput_rps": round(cluster_rps, 5),
        "throughput_vs_4x_p1_ideal": round(throughput_vs_ideal, 3),
        "theoretical_pp_slot_util_m2": round(theory_pp_util, 3),
        "measured_pp_slot_util_cluster": round(measured_pp_slot_util, 3),
        "pp_slot_util_gap_vs_theory": round(measured_pp_slot_util - theory_pp_util, 3),
        "avg_gpu_compute_util_from_timers": round(avg_gpu_compute_util, 3),
        "time_breakdown_ms": {
            "stage_compute_sum_gpu_ms": round(stage_compute, 1),
            "metadata_p2p_max_ms": round(metadata, 1),
            "activation_p2p_max_ms": round(act_p2p, 1),
            "noise_p2p_max_ms": round(noise_p2p, 1),
            "barrier_sync_max_ms": round(barrier, 1),
            "post_infer_sum_gpu_ms": round(post_infer, 1),
            "wall_total": round(wall_ms, 1),
        },
        "time_breakdown_pct_of_wall": {
            "avg_gpu_compute": round(100.0 * avg_gpu_compute_util, 1),
            "pp_p2p_blocking": round(100.0 * pp_p2p_ms / wall_ms, 1) if wall_ms > 0 else 0.0,
            "barrier_sync": round(100.0 * barrier / wall_ms, 1) if wall_ms > 0 else 0.0,
            "post_infer_sum_over_wall": round(100.0 * post_infer / (n_gpus * wall_ms), 1) if wall_ms > 0 else 0.0,
        },
        "gpipe_slots_cluster": {
            "busy": int(busy),
            "idle": int(idle),
            "per_rank_busy_avg": round(busy / n_gpus, 1) if n_gpus else 0,
            "per_rank_idle_avg": round(idle / n_gpus, 1) if n_gpus else 0,
        },
        "sp_a2a_overlap_fraction": round(sp_overlap_frac, 3),
        "ceilings": {
            "naive_4x_serial_s": round(naive_4_serial_s, 1),
            "throughput_at_theory_pp_slots_only_s": round(wall_s * theory_pp_util / max(measured_pp_slot_util, 1e-6), 1),
            "ideal_4gpu_rps": round(ideal_4gpu_rps, 5),
            "gap_throughput_vs_theory_pp_slots_pct": round(
                100.0 * (throughput_vs_ideal - theory_pp_util) / theory_pp_util, 1
            )
            if theory_pp_util > 0
            else None,
        },
        "orch_stats": {k: int(v) for k, v in merged.items() if k.startswith("orch_")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_sp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--layers_per_stage", type=int, default=2)
    parser.add_argument("--single_gpu_baseline_s", type=float, default=72.67)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpu_util_log", default="")
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
    theory_pp = theoretical_utilization(2, num_stages, pp_size)
    ideal_4gpu_rps = 4.0 / args.single_gpu_baseline_s
    n_steps = 4

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

    gpu_proc = None
    gpu_util_log = args.gpu_util_log
    if is_main_process() and gpu_util_log:
        with open(gpu_util_log, "w") as f:
            f.write("# nvidia-smi dmon during quad overlap full run\n")
        gpu_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
            stdout=open(gpu_util_log, "a"),
            stderr=subprocess.DEVNULL,
        )

    if pp_rank == 0:
        print(f"Profiling quad overlap lps={lps} theory_pp_util={theory_pp:.3f}")

    wall_s, full_prof = _profile_full_run(
        runner, payloads, pp_group, pp_rank, pp_size, num_layers, lps,
        seq_parallel=seq_parallel, sp_overlap=True,
    )
    full_merged = _reduce_profiles(full_prof)

    if gpu_proc is not None:
        gpu_proc.terminate()
        gpu_proc.wait(timeout=5)

    single_s = 42.8  # from prior bench; profile run focuses on breakdown
    analysis = _analyze(
        full_merged,
        wall_s=wall_s,
        single_s=single_s,
        num_stages=num_stages,
        n_steps=n_steps,
        ideal_4gpu_rps=ideal_4gpu_rps,
        theory_pp_util=theory_pp,
    )
    per_step = {k: round(v / n_steps, 1) for k, v in analysis["time_breakdown_ms"].items() if k != "wall_total"}

    gpu_util_summary = {}
    if is_main_process() and gpu_util_log and Path(gpu_util_log).exists():
        utils: list[float] = []
        for line in Path(gpu_util_log).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("gpu") or line.startswith("Idx"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                gpu_id = int(parts[0])
                if gpu_id >= 4:
                    continue
                try:
                    utils.append(float(parts[1]))
                except ValueError:
                    pass
        if utils:
            gpu_util_summary = {
                "sm_util_avg_pct": round(sum(utils) / len(utils), 1),
                "sm_util_max_pct": round(max(utils), 1),
                "sm_util_min_pct": round(min(utils), 1),
                "samples": len(utils),
            }

    out = {
        "description": "PP×SP quad overlap profile (lps=2, 4 req, PP GPipe m=2 + SP a2a)",
        "pp_layers_per_stage": lps,
        "num_stages": num_stages,
        "theoretical_pp_slot_util": round(theory_pp, 4),
        "full_run": analysis,
        "per_step_avg_ms": per_step,
        "gpu_util_dmon": gpu_util_summary,
        "interpretation": {
            "theory_94pct_refers_to": "PP GPipe stage-slot utilization: m*S / (pp*(m+S-1)) with m=2,S=20",
            "throughput_ideal_4gpu": round(ideal_4gpu_rps, 5),
            "measured_vs_theory_pp_slots": (
                f"measured {analysis['measured_pp_slot_util_cluster']:.1%} vs theory {theory_pp:.1%}"
            ),
        },
    }

    if pp_rank == 0:
        tb = analysis["time_breakdown_pct_of_wall"]
        print(f"wall={wall_s:.1f}s rps={analysis['cluster_throughput_rps']:.4f} "
              f"({100*analysis['throughput_vs_4x_p1_ideal']:.1f}% of 4×P1 ideal)")
        print(f"PP slot util: measured={analysis['measured_pp_slot_util_cluster']:.3f} theory={theory_pp:.3f}")
        print(f"time%: avg_gpu_compute={tb['avg_gpu_compute']:.1f} pp_p2p={tb['pp_p2p_blocking']:.1f} "
              f"barrier={tb['barrier_sync']:.1f}")
        if gpu_util_summary:
            print(f"GPU SM util avg={gpu_util_summary['sm_util_avg_pct']:.1f}% "
                  f"max={gpu_util_summary['sm_util_max_pct']:.1f}%")

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
