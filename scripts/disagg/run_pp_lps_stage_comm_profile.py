#!/usr/bin/env python3
"""Per-stage compute vs inter-stage P2P comm for PP=2 GPipe (lps sweep).

Profiles one dual-tenant GPipe denoise step (m=2) and aggregates:
  - per-stage compute_ms (by stage index, averaged over microbatches)
  - per-transition activation P2P ms (stage -> stage+1)
  - per-slot critical path (max rank compute + barrier + P2P in that slot)

Overlap verdict uses slot-level accounting: comm is hideable when
median stage_compute_ms > p95 activation_p2p_ms for that lps.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.networks.wan.infer.pipeline_parallel import (
    recv_activation,
    recv_noise_pred,
    recv_pre_metadata,
    send_activation,
    send_noise_pred,
    send_pre_metadata,
)
from lightx2v.models.networks.wan.pp_utils import pp_last_stage_owner, pp_num_stages, pp_stage_owner
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from scripts.disagg.pp_interleaved_pipeline import (
    PpTenantCtx,
    _active_wan,
    _ensure_tenant_pre,
    _run_stage_compute,
    gpipe_slot,
    run_gpipe_dual_pipeline,
    theoretical_utilization,
)
from scripts.disagg.run_phase1_transformer_bench import (
    _init_distributed,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)


def _cuda_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


@dataclass
class SlotRecord:
    slot_index: int
    microbatch: int | None = None
    stage: int | None = None
    pp_rank: int | None = None
    rank0_compute_ms: float = 0.0
    rank1_compute_ms: float = 0.0
    barrier_ms: float = 0.0
    noise_p2p_ms: float = 0.0
    activation_p2p_ms: float = 0.0
    activation_transitions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def rank_compute_ms(self) -> tuple[float, float]:
        return self.rank0_compute_ms, self.rank1_compute_ms

    @property
    def slot_critical_ms(self) -> float:
        return max(self.rank0_compute_ms, self.rank1_compute_ms) + self.barrier_ms + self.noise_p2p_ms + self.activation_p2p_ms


def _profile_gpipe_step(
    runner: MultiModelStruct,
    tenants: list[PpTenantCtx],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int,
) -> list[SlotRecord]:
    """Instrumented copy of gpipe_dual_pipeline_step with per-slot timing."""
    num_microbatches = len(tenants)
    num_stages = pp_num_stages(num_layers, pp_size, layers_per_stage)
    last_rank = pp_last_stage_owner(num_layers, pp_size, layers_per_stage)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    for tenant in tenants:
        tenant.pre = None

    pending_x: dict[int, torch.Tensor] = {}
    local_slots: list[SlotRecord] = []

    total_steps = num_microbatches + num_stages - 1
    for k in range(total_steps):
        slot0 = gpipe_slot(k, 0, pp_size, num_stages, num_microbatches)
        slot1 = gpipe_slot(k, 1, pp_size, num_stages, num_microbatches)
        my_slot = slot0 if pp_rank == 0 else slot1

        rec = SlotRecord(slot_index=k)
        if my_slot is not None:
            rec.microbatch, rec.stage = my_slot
            rec.pp_rank = pp_rank

        if slot0 is not None and slot0[1] == 0:
            mb = slot0[0]
            if pp_rank == 0:
                wan = _active_wan(runner, tenants[mb].scheduler)
                _ensure_tenant_pre(wan, tenants[mb])
                send_pre_metadata(tenants[mb].pre, last_rank, pp_group, 5000 + mb)
            if pp_rank == last_rank:
                tenants[mb].pre = recv_pre_metadata(0, pp_group, device, 5000 + mb)

        outputs: dict[tuple[int, int], torch.Tensor] = {}
        pending_noise: dict[int, torch.Tensor] = {}
        compute_ms = 0.0
        if my_slot is not None:
            mb, stage = my_slot
            tenant = tenants[mb]
            wan = _active_wan(runner, tenant.scheduler)
            x_in = pending_x.pop(mb, None)
            c0 = torch.cuda.Event(enable_timing=True)
            c1 = torch.cuda.Event(enable_timing=True)
            c0.record()
            x_out = _run_stage_compute(wan, tenant, stage, num_stages, layers_per_stage, x_in)
            c1.record()
            compute_ms = _cuda_ms(c0, c1)
            if stage == num_stages - 1 and pp_rank == last_rank:
                pending_noise[mb] = wan.post_infer.infer(x_out, tenant.pre)[0]
            elif stage < num_stages - 1:
                outputs[(mb, stage)] = x_out

        if pp_rank == 0:
            rec.rank0_compute_ms = compute_ms
        else:
            rec.rank1_compute_ms = compute_ms

        if dist.is_initialized():
            b0 = torch.cuda.Event(enable_timing=True)
            b1 = torch.cuda.Event(enable_timing=True)
            b0.record()
            dist.barrier(group=pp_group)
            b1.record()
            barrier_ms = _cuda_ms(b0, b1)
        else:
            barrier_ms = 0.0
        rec.barrier_ms = barrier_ms

        noise_mbs: list[int] = []
        for slot in (slot0, slot1):
            if slot is not None and slot[1] == num_stages - 1:
                noise_mbs.append(slot[0])
        noise_ms = 0.0
        for mb in sorted(set(noise_mbs)):
            if pp_rank == last_rank:
                n0 = torch.cuda.Event(enable_timing=True)
                n1 = torch.cuda.Event(enable_timing=True)
                n0.record()
                send_noise_pred(pending_noise[mb], 0, pp_group, 6000 + mb)
                n1.record()
                noise_ms += _cuda_ms(n0, n1)
            if pp_rank == 0:
                n0 = torch.cuda.Event(enable_timing=True)
                n1 = torch.cuda.Event(enable_timing=True)
                n0.record()
                recv_noise_pred(last_rank, pp_group, device, 6000 + mb)
                n1.record()
                noise_ms += _cuda_ms(n0, n1)
        rec.noise_p2p_ms = noise_ms

        send_plans: list[tuple[int, int]] = []
        for slot in (slot0, slot1):
            if slot is not None and slot[1] < num_stages - 1:
                send_plans.append((slot[0], slot[1]))
        send_plans.sort()

        act_ms = 0.0
        for mb, stage in send_plans:
            sender = pp_stage_owner(stage, pp_size)
            receiver = pp_stage_owner(stage + 1, pp_size)
            rec.activation_transitions.append((stage, stage + 1))
            if pp_rank == sender:
                tensor = outputs.get((mb, stage))
                assert tensor is not None
                a0 = torch.cuda.Event(enable_timing=True)
                a1 = torch.cuda.Event(enable_timing=True)
                a0.record()
                send_activation(tensor, receiver, pp_group, 1000 + mb * 100 + stage)
                a1.record()
                act_ms += _cuda_ms(a0, a1)
            if pp_rank == receiver:
                a0 = torch.cuda.Event(enable_timing=True)
                a1 = torch.cuda.Event(enable_timing=True)
                a0.record()
                x_next = recv_activation(sender, pp_group, device, 1000 + mb * 100 + stage)
                a1.record()
                act_ms += _cuda_ms(a0, a1)
                pending_x[mb] = x_next
        rec.activation_p2p_ms = act_ms
        local_slots.append(rec)

    if pp_rank == 0:
        gathered: list[list[SlotRecord]] = [None, None]  # type: ignore[list-item]
        dist.gather_object(local_slots, gathered, dst=0)
        merged: list[SlotRecord] = []
        for k in range(total_steps):
            s0 = gathered[0][k]
            s1 = gathered[1][k]
            merged.append(
                SlotRecord(
                    slot_index=k,
                    microbatch=s0.microbatch if s0.microbatch is not None else s1.microbatch,
                    stage=s0.stage if s0.stage is not None else s1.stage,
                    rank0_compute_ms=s0.rank0_compute_ms,
                    rank1_compute_ms=s1.rank1_compute_ms,
                    barrier_ms=max(s0.barrier_ms, s1.barrier_ms),
                    noise_p2p_ms=s0.noise_p2p_ms + s1.noise_p2p_ms,
                    activation_p2p_ms=s0.activation_p2p_ms + s1.activation_p2p_ms,
                    activation_transitions=s0.activation_transitions or s1.activation_transitions,
                )
            )
        return merged

    dist.gather_object(local_slots, None, dst=0)
    return []


def _aggregate_stage_compute(slots: list[SlotRecord], num_stages: int) -> dict[int, dict[str, float]]:
    by_stage: dict[int, list[float]] = {s: [] for s in range(num_stages)}
    for rec in slots:
        if rec.stage is None:
            continue
        ms = rec.rank0_compute_ms if rec.rank0_compute_ms > 0 else rec.rank1_compute_ms
        if ms > 0:
            by_stage[rec.stage].append(ms)
    out: dict[int, dict[str, float]] = {}
    for stage, vals in by_stage.items():
        if not vals:
            continue
        out[stage] = {
            "mean_ms": statistics.mean(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
            "count": len(vals),
        }
    return out


def _aggregate_p2p_by_transition(slots: list[SlotRecord]) -> dict[str, dict[str, float]]:
    by_trans: dict[str, list[float]] = {}
    for rec in slots:
        if rec.activation_p2p_ms <= 0 or not rec.activation_transitions:
            continue
        per = rec.activation_p2p_ms / len(rec.activation_transitions)
        for s0, s1 in rec.activation_transitions:
            key = f"stage{s0}_to_{s1}"
            by_trans.setdefault(key, []).append(per)
    return {
        k: {
            "mean_ms": statistics.mean(v),
            "min_ms": min(v),
            "max_ms": max(v),
            "count": len(v),
        }
        for k, v in sorted(by_trans.items(), key=lambda x: int(x[0].split("_")[0].replace("stage", "")))
    }


def _overlap_analysis(
    stage_compute: dict[int, dict[str, float]],
    p2p_by_trans: dict[str, dict[str, float]],
) -> dict[str, Any]:
    stage_means = [v["mean_ms"] for v in stage_compute.values()]
    p2p_means = [v["mean_ms"] for v in p2p_by_trans.values()]
    if not stage_means or not p2p_means:
        return {"verdict": "insufficient_data"}

    min_stage = min(stage_means)
    med_stage = statistics.median(stage_means)
    mean_stage = statistics.mean(stage_means)
    max_p2p = max(p2p_means)
    mean_p2p = statistics.mean(p2p_means)
    total_p2p = sum(p2p_means) * max(1, len(p2p_by_trans)) / max(len(p2p_by_trans), 1)

    # ratio: smallest stage compute vs largest single P2P hop
    hideable_per_hop = min_stage > max_p2p
    hideable_median = med_stage > max_p2p
    comm_to_compute = mean_p2p / mean_stage if mean_stage > 0 else float("inf")

    return {
        "stage_compute_mean_ms": round(mean_stage, 3),
        "stage_compute_median_ms": round(med_stage, 3),
        "stage_compute_min_ms": round(min_stage, 3),
        "p2p_per_hop_mean_ms": round(mean_p2p, 3),
        "p2p_per_hop_max_ms": round(max_p2p, 3),
        "comm_to_compute_ratio": round(comm_to_compute, 4),
        "min_stage_gt_max_p2p": hideable_per_hop,
        "median_stage_gt_max_p2p": hideable_median,
        "verdict": (
            "comm_hideable_behind_stage_compute"
            if hideable_median
            else "comm_may_expose_on_critical_path"
        ),
    }


def _slot_summary(slots: list[SlotRecord]) -> dict[str, Any]:
    crit = [s.slot_critical_ms for s in slots]
    compute_only = [max(s.rank0_compute_ms, s.rank1_compute_ms) for s in slots]
    comm_only = [s.barrier_ms + s.noise_p2p_ms + s.activation_p2p_ms for s in slots]
    idle_slots = sum(1 for s in slots if max(s.rank0_compute_ms, s.rank1_compute_ms) < 1.0)
    dual_busy = sum(
        1 for s in slots if s.rank0_compute_ms > 1.0 and s.rank1_compute_ms > 1.0
    )
    return {
        "num_slots": len(slots),
        "idle_slots": idle_slots,
        "dual_busy_slots": dual_busy,
        "slot_critical_sum_ms": round(sum(crit), 3),
        "slot_critical_mean_ms": round(statistics.mean(crit), 3),
        "compute_max_per_slot_mean_ms": round(statistics.mean(compute_only), 3),
        "comm_per_slot_mean_ms": round(statistics.mean(comm_only), 3),
        "comm_fraction_of_slot": round(
            sum(comm_only) / max(sum(crit), 1e-6), 4
        ),
    }


def _profile_lps(
    args: argparse.Namespace,
    lps: int,
) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        config_path=args.config_json,
    )
    config["parallel"] = {"pipe_p_size": 2}
    config["pp_layers_per_stage"] = lps

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
    payload_b["seed"] = 43

    runner = load_wan_transformer(config)
    assert isinstance(runner, MultiModelStruct)
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
    scheds[0].step_pre(0)
    scheds[1].step_pre(0)

    if pp_rank == 0:
        print(f"\n=== profile lps={lps} stages={num_stages} ===")

    slots = _profile_gpipe_step(
        runner, tenants, pp_group, pp_rank, pp_size, num_layers, lps
    )

    stage_compute = _aggregate_stage_compute(slots, num_stages) if pp_rank == 0 else {}
    p2p_by_trans = _aggregate_p2p_by_transition(slots) if pp_rank == 0 else {}
    overlap = _overlap_analysis(stage_compute, p2p_by_trans) if pp_rank == 0 else {}
    slot_sum = _slot_summary(slots) if pp_rank == 0 else {}

    dual_wall = 0.0
    if args.run_dual_e2e:
        for tenant in tenants:
            tenant.pre = None
        scheds[0].step_pre(0)
        scheds[1].step_pre(0)
        _sync_device()
        t0 = time.perf_counter()
        run_gpipe_dual_pipeline(
            runner,
            tenants,
            [payload_a, payload_b],
            pp_group,
            pp_rank,
            pp_size,
            num_layers,
            lps,
        )
        _sync_device()
        dual_wall = time.perf_counter() - t0

    result: dict[str, Any] = {}
    if pp_rank == 0:
        result = {
            "pp_layers_per_stage": lps,
            "num_stages": num_stages,
            "num_microbatches": 2,
            "theoretical_utilization_m2": round(theoretical_utilization(2, num_stages, pp_size), 4),
            "per_stage_compute_ms": {str(k): v for k, v in stage_compute.items()},
            "per_transition_p2p_ms": p2p_by_trans,
            "overlap_analysis": overlap,
            "slot_summary": slot_sum,
            "per_slot_timeline": [
                {
                    "slot": s.slot_index,
                    "stage": s.stage,
                    "microbatch": s.microbatch,
                    "rank0_compute_ms": round(s.rank0_compute_ms, 3),
                    "rank1_compute_ms": round(s.rank1_compute_ms, 3),
                    "barrier_ms": round(s.barrier_ms, 3),
                    "noise_p2p_ms": round(s.noise_p2p_ms, 3),
                    "activation_p2p_ms": round(s.activation_p2p_ms, 3),
                    "slot_critical_ms": round(s.slot_critical_ms, 3),
                    "p2p_transitions": s.activation_transitions,
                }
                for s in slots
            ],
        }
        if args.run_dual_e2e:
            result["dual_pipeline_2req_s"] = round(dual_wall, 3)
            result["dual_throughput_rps"] = round(2.0 / dual_wall, 5) if dual_wall > 0 else 0.0
        print(
            f"lps={lps}: stage_compute_mean={overlap.get('stage_compute_mean_ms')}ms "
            f"p2p_max={overlap.get('p2p_per_hop_max_ms')}ms "
            f"verdict={overlap.get('verdict')} "
            f"slot_comm_frac={slot_sum.get('comm_fraction_of_slot')}"
        )

    if dist.is_initialized():
        dist.destroy_process_group()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--layers_per_stage_list", default="1,2")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--run_dual_e2e", action="store_true")
    parser.add_argument(
        "--single_lps",
        type=int,
        default=None,
        help="If set, profile only this lps (use separate torchrun per value to avoid NCCL re-init issues).",
    )
    args = parser.parse_args()

    if args.single_lps is not None:
        lps_list = [int(args.single_lps)]
    else:
        lps_list = [int(x) for x in args.layers_per_stage_list.split(",") if x.strip()]
        if len(lps_list) > 1:
            raise SystemExit(
                "Profile one lps per torchrun (NCCL re-init is unreliable). "
                "Pass --single_lps N or run the helper shell script."
            )

    all_results: list[dict[str, Any]] = []
    for lps in lps_list:
        all_results.append(_profile_lps(args, lps))

    out = {
        "description": "PP=2 per-stage compute vs inter-stage P2P (one GPipe step, m=2)",
        "metric": "cuda events; one denoise step with dual microbatches",
        "note": (
            "GPipe schedules compute on both ranks within a slot, then barrier + P2P serially. "
            "comm_hideable when median stage compute > max single-hop P2P. "
            "Higher comm_fraction_of_slot at same utilization implies comm on critical path."
        ),
        "results": all_results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
