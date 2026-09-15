"""PP=2 dual-tenant layer pipeline: overlap stage0/stage1 across two requests."""

from __future__ import annotations

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
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler


@dataclass
class PpTenantCtx:
    scheduler: WanScheduler
    inputs: dict[str, Any]


def _active_wan(runner: MultiModelStruct, sched: WanScheduler):
    runner.scheduler = sched
    runner.get_current_model_index()
    wan = runner.model[runner.cur_model_index]
    wan.set_scheduler(sched)
    sched.infer_condition = True
    return wan


def _stage0(wan: Any, inputs: dict[str, Any]) -> tuple[Any, torch.Tensor]:
    pre = wan.pre_infer.infer(wan.pre_weight, inputs)
    wan.transformer_infer.cos_sin = pre.cos_sin
    wan.transformer_infer.reset_infer_states()
    x = wan.transformer_infer.infer_main_blocks(wan.transformer_weights.blocks, pre)
    return pre, x


def _stage1(wan: Any, pre: Any, x: torch.Tensor) -> torch.Tensor:
    pre.x = x
    wan.transformer_infer.cos_sin = pre.cos_sin
    wan.transformer_infer.reset_infer_states()
    x = wan.transformer_infer.infer_main_blocks(wan.transformer_weights.blocks, pre)
    x = wan.transformer_infer.infer_non_blocks(wan.transformer_weights, x, pre.embed)
    return wan.post_infer.infer(x, pre)[0]


def _assign_noise(sched: WanScheduler, noise_pred: torch.Tensor) -> None:
    sched.noise_pred_cond = noise_pred
    sched.noise_pred_uncond = None
    sched.noise_pred_guided = noise_pred
    sched.noise_pred = noise_pred


def pp2_dual_pipeline_step(
    runner: MultiModelStruct,
    tenant_a: PpTenantCtx,
    tenant_b: PpTenantCtx,
    pp_group,
    pp_rank: int,
    pp_size: int,
    *,
    overlap: bool,
) -> None:
    """One denoise step: two in-flight microbatches on PP=2."""
    last_rank = pp_size - 1
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if overlap:
        if pp_rank == 0:
            wan_a = _active_wan(runner, tenant_a.scheduler)
            pre_a, x_a = _stage0(wan_a, tenant_a.inputs)
            send_pre_metadata(pre_a, last_rank, pp_group)
            send_activation(x_a, last_rank, pp_group)

            # B stage0 overlaps rank1 stage1 for A (send returns after rank1 recv, not after stage1).
            wan_b = _active_wan(runner, tenant_b.scheduler)
            pre_b, x_b = _stage0(wan_b, tenant_b.inputs)
            send_pre_metadata(pre_b, last_rank, pp_group)
            send_activation(x_b, last_rank, pp_group)

            noise_a = recv_noise_pred(last_rank, pp_group, device)
            noise_b = recv_noise_pred(last_rank, pp_group, device)
            _assign_noise(tenant_a.scheduler, noise_a)
            _assign_noise(tenant_b.scheduler, noise_b)
            return

        if pp_rank == last_rank:
            pre_a = recv_pre_metadata(0, pp_group, device)
            x_a = recv_activation(0, pp_group, device)
            wan_a = _active_wan(runner, tenant_a.scheduler)
            noise_a = _stage1(wan_a, pre_a, x_a)

            pre_b = recv_pre_metadata(0, pp_group, device)
            x_b = recv_activation(0, pp_group, device)
            send_noise_pred(noise_a, 0, pp_group)

            wan_b = _active_wan(runner, tenant_b.scheduler)
            noise_b = _stage1(wan_b, pre_b, x_b)
            send_noise_pred(noise_b, 0, pp_group)
            return

        raise RuntimeError(f"unsupported pp_rank={pp_rank}")

    # Serial within step: A then B on both ranks (sanity baseline, no stage overlap).
    if pp_rank == 0:
        for tenant in (tenant_a, tenant_b):
            wan = _active_wan(runner, tenant.scheduler)
            pre, x = _stage0(wan, tenant.inputs)
            send_pre_metadata(pre, last_rank, pp_group)
            send_activation(x, last_rank, pp_group)
            noise = recv_noise_pred(last_rank, pp_group, device)
            _assign_noise(tenant.scheduler, noise)
        return

    if pp_rank == last_rank:
        for tenant in (tenant_a, tenant_b):
            pre = recv_pre_metadata(0, pp_group, device)
            x = recv_activation(0, pp_group, device)
            wan = _active_wan(runner, tenant.scheduler)
            noise = _stage1(wan, pre, x)
            send_noise_pred(noise, 0, pp_group)
        return

    raise RuntimeError(f"unsupported pp_rank={pp_rank}")


def run_pp2_dual_pipeline(
    runner: MultiModelStruct,
    tenant_a: PpTenantCtx,
    tenant_b: PpTenantCtx,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    pp_group,
    pp_rank: int,
    pp_size: int,
    *,
    overlap: bool,
) -> None:
    tenant_a.scheduler.prepare(
        seed=int(payload_a["seed"]),
        latent_shape=payload_a["latent_shape"],
        image_encoder_output=payload_a["image_encoder_output"],
    )
    tenant_b.scheduler.prepare(
        seed=int(payload_b["seed"]),
        latent_shape=payload_b["latent_shape"],
        image_encoder_output=payload_b["image_encoder_output"],
    )
    n_steps = tenant_a.scheduler.infer_steps
    for step_index in range(n_steps):
        if pp_rank == 0:
            tenant_a.scheduler.step_pre(step_index=step_index)
            tenant_b.scheduler.step_pre(step_index=step_index)
        else:
            tenant_a.scheduler.step_index = step_index
            tenant_b.scheduler.step_index = step_index
        if dist.is_initialized():
            dist.barrier(group=pp_group)
        pp2_dual_pipeline_step(
            runner,
            tenant_a,
            tenant_b,
            pp_group,
            pp_rank,
            pp_size,
            overlap=overlap,
        )
        if pp_rank == 0:
            tenant_a.scheduler.step_post()
            tenant_b.scheduler.step_post()
        if dist.is_initialized():
            dist.barrier(group=pp_group)
