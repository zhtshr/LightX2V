"""Interleaved PP (GPipe) dual-tenant pipeline on 2 GPUs — matched P2P ordering."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.pipeline_parallel import (
    recv_activation,
    recv_noise_pred,
    recv_pre_metadata,
    send_activation,
    send_noise_pred,
    send_pre_metadata,
)
from lightx2v.models.networks.wan.pp_utils import (
    pp_last_stage_owner,
    pp_num_stages,
    pp_stage_layer_range,
    pp_stage_owner,
)
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler


@dataclass
class PpTenantCtx:
    scheduler: WanScheduler
    inputs: dict[str, Any]
    pre: Any | None = None


def _tag_meta(microbatch: int) -> int:
    return 5000 + microbatch


def _tag_activation(microbatch: int, stage: int) -> int:
    return 1000 + microbatch * 100 + stage


def _tag_noise(microbatch: int) -> int:
    return 6000 + microbatch


def _active_wan(runner: MultiModelStruct, sched: WanScheduler):
    runner.scheduler = sched
    runner.get_current_model_index()
    wan = runner.model[runner.cur_model_index]
    wan.set_scheduler(sched)
    sched.infer_condition = True
    return wan


def _blocks_for_stage(blocks, stage: int, layers_per_stage: int) -> list:
    lo, hi = pp_stage_layer_range(stage, layers_per_stage)
    return [b for b in blocks if lo <= b.block_index < hi]


def _ensure_tenant_pre(wan: Any, tenant: PpTenantCtx) -> None:
    """Run pre_infer once per tenant per denoise step; apply SP shard when enabled."""
    if tenant.pre is not None:
        return
    pre = wan.pre_infer.infer(wan.pre_weight, tenant.inputs)
    if wan.config.get("seq_parallel"):
        pre = wan._seq_parallel_pre_process(pre)
    tenant.pre = pre


def _sync_step_group(pp_group, seq_parallel: bool) -> None:
    if not dist.is_initialized():
        return
    # PP×SP: seq_p peers sync inside attention; only pipe_p partners must align
    # on GPipe slot boundaries. World barrier forces idle SP columns to wait.
    del seq_parallel
    dist.barrier(group=pp_group)


def _run_stage_compute(
    wan: Any,
    tenant: PpTenantCtx,
    stage: int,
    num_stages: int,
    layers_per_stage: int,
    x_in: torch.Tensor | None,
) -> torch.Tensor:
    blocks = _blocks_for_stage(wan.transformer_weights.blocks, stage, layers_per_stage)
    if stage == 0:
        _ensure_tenant_pre(wan, tenant)
        pre = tenant.pre
        x = pre.x if x_in is None else x_in
    else:
        pre = tenant.pre
        assert pre is not None and x_in is not None
        x = x_in
    pre.x = x
    wan.transformer_infer.cos_sin = pre.cos_sin
    wan.transformer_infer.reset_infer_states()
    x = wan.transformer_infer.infer_main_blocks(blocks, pre)
    if stage == num_stages - 1:
        x = wan.transformer_infer.infer_non_blocks(wan.transformer_weights, x, pre.embed)
        if wan.config.get("seq_parallel"):
            x = wan._seq_parallel_post_process(x)
    return x


def gpipe_slot(timestep: int, gpu: int, pp_size: int, num_stages: int, num_microbatches: int):
    for stage in range(num_stages):
        if stage % pp_size != gpu:
            continue
        microbatch = timestep - stage
        if 0 <= microbatch < num_microbatches:
            return microbatch, stage
    return None


def _paired_activation_p2p(
    pp_rank: int,
    pp_group,
    device: torch.device,
    sender: int,
    receiver: int,
    microbatch: int,
    stage: int,
    tensor: torch.Tensor | None,
    *,
    profile: dict[str, float] | None = None,
) -> torch.Tensor | None:
    if pp_rank == sender:
        assert tensor is not None
        c0 = None
        if profile is not None:
            c0 = torch.cuda.Event(enable_timing=True)
            c0.record()
        send_activation(tensor, receiver, pp_group, _tag_activation(microbatch, stage))
        if profile is not None and c0 is not None:
            c1 = torch.cuda.Event(enable_timing=True)
            c1.record()
            torch.cuda.synchronize()
            profile["comm_ms"] = profile.get("comm_ms", 0.0) + float(c0.elapsed_time(c1))
        return None
    if pp_rank == receiver:
        return recv_activation(sender, pp_group, device, _tag_activation(microbatch, stage))
    return None


def gpipe_dual_pipeline_step(
    runner: MultiModelStruct,
    tenants: list[PpTenantCtx],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int,
    *,
    seq_parallel: bool = False,
    profile: dict[str, float] | None = None,
) -> None:
    num_microbatches = len(tenants)
    num_stages = pp_num_stages(num_layers, pp_size, layers_per_stage)
    last_rank = pp_last_stage_owner(num_layers, pp_size, layers_per_stage)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    for tenant in tenants:
        tenant.pre = None

    pending_x: dict[int, torch.Tensor] = {}

    total_steps = num_microbatches + num_stages - 1
    for k in range(total_steps):
        slot0 = gpipe_slot(k, 0, pp_size, num_stages, num_microbatches)
        slot1 = gpipe_slot(k, 1, pp_size, num_stages, num_microbatches)
        my_slot = slot0 if pp_rank == 0 else slot1

        # Metadata when a microbatch starts stage 0
        if slot0 is not None and slot0[1] == 0:
            mb = slot0[0]
            if pp_rank == 0:
                wan = _active_wan(runner, tenants[mb].scheduler)
                _ensure_tenant_pre(wan, tenants[mb])
                send_pre_metadata(tenants[mb].pre, last_rank, pp_group, _tag_meta(mb))
            if pp_rank == last_rank:
                tenants[mb].pre = recv_pre_metadata(0, pp_group, device, _tag_meta(mb))

        outputs: dict[tuple[int, int], torch.Tensor] = {}
        pending_noise: dict[int, torch.Tensor] = {}
        if my_slot is not None:
            mb, stage = my_slot
            tenant = tenants[mb]
            wan = _active_wan(runner, tenant.scheduler)
            x_in = pending_x.pop(mb, None)

            t0 = None
            if profile is not None:
                t0 = torch.cuda.Event(enable_timing=True)
                t0.record()

            x_out = _run_stage_compute(wan, tenant, stage, num_stages, layers_per_stage, x_in)

            if profile is not None and t0 is not None:
                t1 = torch.cuda.Event(enable_timing=True)
                t1.record()
                torch.cuda.synchronize()
                profile["compute_ms"] = profile.get("compute_ms", 0.0) + float(t0.elapsed_time(t1))

            if stage == num_stages - 1 and pp_rank == last_rank:
                pending_noise[mb] = wan.post_infer.infer(x_out, tenant.pre)[0]
            elif stage < num_stages - 1:
                outputs[(mb, stage)] = x_out

        _sync_step_group(pp_group, seq_parallel)

        noise_mbs: list[int] = []
        for slot in (slot0, slot1):
            if slot is not None and slot[1] == num_stages - 1:
                noise_mbs.append(slot[0])
        for mb in sorted(set(noise_mbs)):
            if pp_rank == last_rank:
                send_noise_pred(pending_noise[mb], 0, pp_group, _tag_noise(mb))
            if pp_rank == 0:
                noise = recv_noise_pred(last_rank, pp_group, device, _tag_noise(mb))
                tenant = tenants[mb]
                tenant.scheduler.noise_pred_cond = noise
                tenant.scheduler.noise_pred_uncond = None
                tenant.scheduler.noise_pred_guided = noise
                tenant.scheduler.noise_pred = noise

        # P2P activations: both ranks execute the same send plans (in order)
        send_plans: list[tuple[int, int]] = []
        for slot in (slot0, slot1):
            if slot is not None and slot[1] < num_stages - 1:
                send_plans.append((slot[0], slot[1]))
        send_plans.sort()
        for mb, stage in send_plans:
            sender = pp_stage_owner(stage, pp_size)
            receiver = pp_stage_owner(stage + 1, pp_size)
            x_next = _paired_activation_p2p(
                pp_rank,
                pp_group,
                device,
                sender,
                receiver,
                mb,
                stage,
                outputs.get((mb, stage)) if pp_rank == sender else None,
                profile=profile,
            )
            if pp_rank == receiver:
                assert x_next is not None
                pending_x[mb] = x_next


def run_gpipe_dual_pipeline(
    runner: MultiModelStruct,
    tenants: list[PpTenantCtx],
    payloads: list[dict[str, Any]],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int,
    *,
    seq_parallel: bool = False,
    per_request_latency_s: dict[int, float] | None = None,
) -> None:
    is_pp_leader = pp_rank == 0
    wall_t0 = time.perf_counter()
    for tenant, payload in zip(tenants, payloads):
        tenant.scheduler.prepare(
            seed=int(payload["seed"]),
            latent_shape=payload["latent_shape"],
            image_encoder_output=payload["image_encoder_output"],
        )
    n_steps = tenants[0].scheduler.infer_steps
    for step_index in range(n_steps):
        if is_pp_leader:
            for tenant in tenants:
                tenant.scheduler.step_pre(step_index=step_index)
        else:
            for tenant in tenants:
                tenant.scheduler.step_index = step_index
        _sync_step_group(pp_group, seq_parallel)
        gpipe_dual_pipeline_step(
            runner,
            tenants,
            pp_group,
            pp_rank,
            pp_size,
            num_layers,
            layers_per_stage,
            seq_parallel=seq_parallel,
        )
        if is_pp_leader:
            for tenant_id, tenant in enumerate(tenants):
                tenant.scheduler.step_post()
                if per_request_latency_s is not None and step_index == n_steps - 1:
                    per_request_latency_s[tenant_id] = time.perf_counter() - wall_t0
        _sync_step_group(pp_group, seq_parallel)


def theoretical_utilization(num_microbatches: int, num_stages: int, pp_size: int = 2) -> float:
    """Fraction of PP GPU-slots used vs naive m×S full compute (≤1 when schedule is feasible)."""
    total_slots = pp_size * (num_microbatches + num_stages - 1)
    useful = num_microbatches * num_stages
    if total_slots <= 0:
        return 0.0
    return useful / total_slots


def max_feasible_microbatches(num_stages: int, pp_size: int = 2) -> int:
    """Largest m with m*S <= pp_size*(m+S-1) for GPipe slot schedule."""
    if num_stages <= 2:
        return max(1, num_stages)
    # m <= pp_size*(S-1) / (S-pp_size); for pp_size=2: m <= 2*(S-1)/(S-2)
    return max(1, (pp_size * (num_stages - 1)) // (num_stages - pp_size))
