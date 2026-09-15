"""PP×SP quad pipeline: GPipe m=2 (PP overlap) + N requests with optional SP a2a overlap.

Layouts: ``oct`` = 2×seq_p tenants (e.g. 8 req @ seq_p=4); ``dual`` = 2×m = 4 tenants
(one SP overlap pair per PP microbatch; default for seq_p≥4 in bench).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist

from scripts.disagg.pp_interleaved_pipeline import (
    PpTenantCtx,
    _active_wan,
    _blocks_for_stage,
    _ensure_tenant_pre,
    _paired_activation_p2p,
    _run_stage_compute,
    _sync_step_group,
    _tag_activation,
    _tag_meta,
    _tag_noise,
    gpipe_slot,
)
from scripts.disagg.run_phase3_dual_overlap_bench import (
    A2AOrchestrator,
    BlockMid,
    TenantCtx,
    _OnceCompute,
    _bind_tenant,
)
from lightx2v.models.networks.wan.infer.pipeline_parallel import (
    P2pAsyncSend,
    recv_activation,
    recv_noise_pred,
    recv_pre_metadata,
    send_noise_pred,
    send_pre_metadata,
    _encode_dtype,
    _global_rank,
)
from lightx2v.models.networks.wan.pp_utils import pp_last_stage_owner, pp_num_stages, pp_stage_owner
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct

def _cuda_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _profile_add(profile: dict[str, float] | None, key: str, ms: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + ms


def _profile_max_pp(profile: dict[str, float] | None, pp_group, key: str, local_ms: float) -> None:
    """Accumulate per-slot critical-path time (MAX across pipe_p ranks)."""
    if profile is None:
        return
    ms = local_ms
    if dist.is_initialized():
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = torch.tensor([ms], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pp_group)
        ms = float(t.item())
    _profile_add(profile, key, ms)


PP_GPIPE_MICROBATCHES = 2

SlotBarrierMode = Literal["always", "bubble_only", "none"]
QuadTenantLayout = Literal["oct", "dual"]


def num_quad_tenants(seq_p_size: int, layout: QuadTenantLayout = "oct") -> int:
    """Concurrent requests: oct = 2×seq_p (full SP pairing); dual = 2×m (one SP pair per mb)."""
    if layout == "dual":
        return PP_GPIPE_MICROBATCHES * 2
    return PP_GPIPE_MICROBATCHES * seq_p_size


def sp_pair_offset(seq_p_size: int) -> int:
    """Tenant-index gap between SP overlap partners (same PP microbatch)."""
    return PP_GPIPE_MICROBATCHES * (seq_p_size // 2)


def tenants_for_mb(mb: int, seq_p_size: int, layout: QuadTenantLayout = "oct") -> list[int]:
    if layout == "dual":
        return [mb * 2, mb * 2 + 1]
    return [mb + sp_rank * PP_GPIPE_MICROBATCHES for sp_rank in range(seq_p_size)]


def sp_overlap_pairs(mb: int, seq_p_size: int, layout: QuadTenantLayout = "oct") -> list[tuple[int, int]]:
    if layout == "dual":
        return [(mb * 2, mb * 2 + 1)]
    offset = sp_pair_offset(seq_p_size)
    return [
        (mb + i * PP_GPIPE_MICROBATCHES, mb + i * PP_GPIPE_MICROBATCHES + offset)
        for i in range(seq_p_size // 2)
    ]


@dataclass
class QuadPipelineOpts:
    """Runtime toggles for quad overlap optimizations."""

    defer_orch_stream_sync: bool = False
    orch_sync_per_step: bool = False
    slot_barrier_mode: SlotBarrierMode = "always"
    barrier_after_p2p: bool = False
    async_activation_p2p: bool = False


def _finish_orch_streams(orch: A2AOrchestrator, *, defer: bool) -> float:
    t0 = time.perf_counter()
    if defer:
        default = torch.cuda.current_stream(device=orch.device)
        default.wait_stream(orch.compute_stream)
        default.wait_stream(orch.comm_stream)
    else:
        orch.compute_stream.synchronize()
        orch.comm_stream.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _slot_needs_barrier(
    slot0: tuple[int, int] | None,
    slot1: tuple[int, int] | None,
    mode: SlotBarrierMode,
) -> bool:
    if mode == "always":
        return True
    if mode == "none":
        return False
    # bubble_only: sync when either PP rank has no compute this slot
    return slot0 is None or slot1 is None


def _async_send_activation(
    tensor: torch.Tensor,
    dst: int,
    pp_group,
    tag: int,
    batch: P2pAsyncSend,
) -> None:
    shape = torch.zeros(8, dtype=torch.int64, device=tensor.device)
    shape[0] = tensor.ndim
    for i, s in enumerate(tensor.shape):
        shape[i + 1] = s
    meta = _encode_dtype(tensor.dtype, tensor.device)
    batch._keepalive.extend([shape, meta])
    t = tensor.contiguous()
    batch._keepalive.append(t)
    dst_rank = _global_rank(pp_group, dst)
    batch._works.append(dist.isend(shape, dst=dst_rank, group=pp_group, tag=tag))
    batch._works.append(dist.isend(meta, dst=dst_rank, group=pp_group, tag=tag))
    batch._works.append(dist.isend(t, dst=dst_rank, group=pp_group, tag=tag))


@dataclass
class QuadPpTenantCtx(PpTenantCtx):
    x: torch.Tensor | None = None
    ti_snap: Any | None = None
    block_cache: dict[int, Any] = field(default_factory=dict)


def _as_overlap_tenant(t: QuadPpTenantCtx, name: str) -> TenantCtx:
    return TenantCtx(
        name=name,
        scheduler=t.scheduler,
        inputs=t.inputs,
        pre_infer_out=t.pre,
        x=t.x,
        ti_snap=t.ti_snap,
        block_cache=t.block_cache,
    )


def _sync_overlap_tenant(t: QuadPpTenantCtx, ot: TenantCtx) -> None:
    t.x = ot.x
    t.ti_snap = ot.ti_snap
    t.block_cache = ot.block_cache


def _ensure_block_pp(tenant: TenantCtx | QuadPpTenantCtx, wan: Any, ti: Any, block_idx: int) -> Any:
    if block_idx in tenant.block_cache:
        return tenant.block_cache[block_idx]
    for block in wan.transformer_weights.blocks:
        if block.block_index == block_idx:
            tenant.block_cache[block_idx] = block
            return block
    raise IndexError(f"block_index {block_idx} not loaded on pp_rank")


def _run_self_attn_block_pp(wan: Any, ti: Any, tenant: TenantCtx, block_idx: int) -> BlockMid:
    block = _ensure_block_pp(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    x = tenant.x
    pre = tenant.pre_infer_out
    if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
        x = block.compute_phases[0].before_proj.apply(x) + pre.x
    mods = ti.pre_process(block.compute_phases[0].modulation, pre.embed0)
    y_out = ti.infer_self_attn(block.compute_phases[0], x, mods[0], mods[1])
    tenant.x = x
    from scripts.disagg.run_phase3_dual_overlap_bench import _capture_ti_snap

    tenant.ti_snap = _capture_ti_snap(ti)
    return BlockMid(mods=mods, y_out=y_out)


def _run_cross_ffn_block_pp(wan: Any, ti: Any, tenant: TenantCtx, block_idx: int, mid: BlockMid) -> None:
    block = _ensure_block_pp(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mid.mods
    x, attn_out = ti.infer_cross_attn(
        block.compute_phases[1], tenant.x, tenant.pre_infer_out.context, mid.y_out, gate_msa,
    )
    y = ti.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
    tenant.x = ti.post_process(x, y, c_gate_msa, tenant.pre_infer_out)
    from scripts.disagg.run_phase3_dual_overlap_bench import _capture_ti_snap

    tenant.ti_snap = _capture_ti_snap(ti)


def _a2a_overlap_block_range(
    runner: MultiModelStruct,
    tenant_a: QuadPpTenantCtx,
    tenant_b: QuadPpTenantCtx,
    blocks: list,
    orch: A2AOrchestrator,
    *,
    overlap: bool,
    defer_stream_sync: bool = False,
    skip_stage_sync: bool = False,
) -> float:
    """Layer-wise a2a overlap on a PP stage's block list for two SP tenants."""
    if not blocks:
        raise ValueError("empty block list for SP overlap")

    oa = _as_overlap_tenant(tenant_a, "a")
    ob = _as_overlap_tenant(tenant_b, "b")
    wan_a, ti_a = _bind_tenant(runner, oa)
    wan_b, ti_b = _bind_tenant(runner, ob)
    for block in blocks:
        idx = block.block_index
        _ensure_block_pp(oa, wan_a, ti_a, idx)
        _ensure_block_pp(ob, wan_b, ti_b, idx)

    lo = blocks[0].block_index
    hi = blocks[-1].block_index + 1

    orch.enabled = True
    orch.overlap_enabled = False
    orch.other_compute_cb = None

    mid_a = _run_self_attn_block_pp(wan_a, ti_a, oa, lo)

    for k in range(lo, hi):
        if overlap:
            orch.overlap_enabled = True
            orch.other_compute_cb = _OnceCompute(
                lambda k=k, m=mid_a: _run_cross_ffn_block_pp(
                    *_bind_tenant(runner, oa), oa, k, m,
                ),
            )
        else:
            orch.overlap_enabled = False
            orch.other_compute_cb = None

        wan_b, ti_b = _bind_tenant(runner, ob)
        if overlap:
            orch.ti_ref = ti_b
        try:
            mid_b = _run_self_attn_block_pp(wan_b, ti_b, ob, k)
        finally:
            if overlap:
                orch.ti_ref = None

        if not overlap:
            wan_a, ti_a = _bind_tenant(runner, oa)
            _run_cross_ffn_block_pp(wan_a, ti_a, oa, k, mid_a)

        if k + 1 < hi:
            if overlap:
                orch.overlap_enabled = True
                orch.other_compute_cb = _OnceCompute(
                    lambda k=k, mb=mid_b: _run_cross_ffn_block_pp(
                        *_bind_tenant(runner, ob), ob, k, mb,
                    ),
                )
            else:
                orch.overlap_enabled = False
                orch.other_compute_cb = None

            wan_a, ti_a = _bind_tenant(runner, oa)
            if overlap:
                orch.ti_ref = ti_a
            try:
                mid_a = _run_self_attn_block_pp(wan_a, ti_a, oa, k + 1)
            finally:
                if overlap:
                    orch.ti_ref = None

            if not overlap:
                wan_b, ti_b = _bind_tenant(runner, ob)
                _run_cross_ffn_block_pp(wan_b, ti_b, ob, k, mid_b)
        else:
            orch.overlap_enabled = False
            orch.other_compute_cb = None
            wan_b, ti_b = _bind_tenant(runner, ob)
            _run_cross_ffn_block_pp(wan_b, ti_b, ob, k, mid_b)

    orch.overlap_enabled = False
    orch.other_compute_cb = None
    orch.enabled = False
    if skip_stage_sync:
        sync_ms = 0.0
    else:
        sync_ms = _finish_orch_streams(orch, defer=defer_stream_sync)

    _sync_overlap_tenant(tenant_a, oa)
    _sync_overlap_tenant(tenant_b, ob)
    return sync_ms


def _run_stage_sp_dual(
    runner: MultiModelStruct,
    tenant_a: QuadPpTenantCtx,
    tenant_b: QuadPpTenantCtx,
    stage: int,
    num_stages: int,
    layers_per_stage: int,
    x_in_a: torch.Tensor | None,
    x_in_b: torch.Tensor | None,
    orch: A2AOrchestrator,
    *,
    overlap: bool,
    defer_stream_sync: bool = False,
    skip_stage_sync: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    wan_a = _active_wan(runner, tenant_a.scheduler)
    wan_b = _active_wan(runner, tenant_b.scheduler)
    blocks = _blocks_for_stage(wan_a.transformer_weights.blocks, stage, layers_per_stage)
    if not blocks:
        raise RuntimeError(f"empty stage {stage}")

    for tenant, wan, x_in in (
        (tenant_a, wan_a, x_in_a),
        (tenant_b, wan_b, x_in_b),
    ):
        if stage == 0:
            _ensure_tenant_pre(wan, tenant)
            tenant.x = tenant.pre.x if x_in is None else x_in
        else:
            assert tenant.pre is not None and x_in is not None
            tenant.x = x_in
        wan.transformer_infer.cos_sin = tenant.pre.cos_sin
        wan.transformer_infer.reset_infer_states()

    sync_ms = _a2a_overlap_block_range(
        runner, tenant_a, tenant_b, blocks, orch, overlap=overlap,
        defer_stream_sync=defer_stream_sync,
        skip_stage_sync=skip_stage_sync,
    )

    wan_a = _active_wan(runner, tenant_a.scheduler)
    wan_b = _active_wan(runner, tenant_b.scheduler)
    x_a = tenant_a.x
    x_b = tenant_b.x
    if stage == num_stages - 1:
        x_a = wan_a.transformer_infer.infer_non_blocks(wan_a.transformer_weights, x_a, tenant_a.pre.embed)
        x_b = wan_b.transformer_infer.infer_non_blocks(wan_b.transformer_weights, x_b, tenant_b.pre.embed)
        if wan_a.config.get("seq_parallel"):
            x_a = wan_a._seq_parallel_post_process(x_a)
            x_b = wan_b._seq_parallel_post_process(x_b)
    return x_a, x_b, sync_ms


def gpipe_quad_sp_pipeline_step(
    runner: MultiModelStruct,
    tenants: list[QuadPpTenantCtx],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int,
    orch: A2AOrchestrator | None,
    *,
    seq_p_size: int = 2,
    layout: QuadTenantLayout = "oct",
    seq_parallel: bool,
    sp_overlap: bool,
    profile: dict[str, float] | None = None,
    pipeline_opts: QuadPipelineOpts | None = None,
) -> None:
    opts = pipeline_opts or QuadPipelineOpts()
    expected = num_quad_tenants(seq_p_size, layout)
    assert len(tenants) == expected, f"expected {expected} tenants for seq_p={seq_p_size} layout={layout}"
    num_microbatches = PP_GPIPE_MICROBATCHES
    num_stages = pp_num_stages(num_layers, pp_size, layers_per_stage)
    last_rank = pp_last_stage_owner(num_layers, pp_size, layers_per_stage)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    for tenant in tenants:
        tenant.pre = None

    pending_x: dict[int, torch.Tensor] = {}
    total_steps = num_microbatches + num_stages - 1

    use_sp_dual = sp_overlap and seq_parallel and orch is not None

    for k in range(total_steps):
        t_slot = time.perf_counter()
        slot_compute_ms = 0.0
        slot_meta_ms = 0.0
        slot_noise_ms = 0.0
        slot_act_ms = 0.0
        slot_barrier_ms = 0.0
        slot_orch_sync_ms = 0.0
        slot_pre_infer_ms = 0.0

        slot0 = gpipe_slot(k, 0, pp_size, num_stages, num_microbatches)
        slot1 = gpipe_slot(k, 1, pp_size, num_stages, num_microbatches)
        my_slot = slot0 if pp_rank == 0 else slot1

        if my_slot is not None:
            _profile_add(profile, "n_busy_slots", 1.0)
        else:
            _profile_add(profile, "n_idle_slots", 1.0)

        if slot0 is not None and slot0[1] == 0:
            mb = slot0[0]
            t_meta = time.perf_counter()
            if pp_rank == 0:
                for m in tenants_for_mb(mb, seq_p_size, layout):
                    wan = _active_wan(runner, tenants[m].scheduler)
                    t_pre = time.perf_counter()
                    _ensure_tenant_pre(wan, tenants[m])
                    slot_pre_infer_ms += (time.perf_counter() - t_pre) * 1000.0
                    send_pre_metadata(tenants[m].pre, last_rank, pp_group, _tag_meta(m))
            if pp_rank == last_rank:
                for m in tenants_for_mb(mb, seq_p_size, layout):
                    tenants[m].pre = recv_pre_metadata(0, pp_group, device, _tag_meta(m))
            slot_meta_ms = (time.perf_counter() - t_meta) * 1000.0

        outputs: dict[tuple[int, int], torch.Tensor] = {}
        pending_noise: dict[int, torch.Tensor] = {}

        if my_slot is not None:
            mb, stage = my_slot
            assert mb < PP_GPIPE_MICROBATCHES
            mb_tenants = tenants_for_mb(mb, seq_p_size, layout)
            x_ins = {m: pending_x.pop(m, None) for m in mb_tenants}

            c0 = torch.cuda.Event(enable_timing=True)
            c0.record()
            for base, partner in sp_overlap_pairs(mb, seq_p_size, layout):
                tenant_a = tenants[base]
                tenant_b = tenants[partner]
                x_in_a = x_ins[base]
                x_in_b = x_ins[partner]
                if use_sp_dual:
                    x_out_a, x_out_b, orch_sync_ms = _run_stage_sp_dual(
                        runner,
                        tenant_a,
                        tenant_b,
                        stage,
                        num_stages,
                        layers_per_stage,
                        x_in_a,
                        x_in_b,
                        orch,
                        overlap=True,
                        defer_stream_sync=opts.defer_orch_stream_sync,
                        skip_stage_sync=opts.orch_sync_per_step,
                    )
                    slot_orch_sync_ms += orch_sync_ms
                else:
                    wan_a = _active_wan(runner, tenant_a.scheduler)
                    wan_b = _active_wan(runner, tenant_b.scheduler)
                    x_out_a = _run_stage_compute(
                        wan_a, tenant_a, stage, num_stages, layers_per_stage, x_in_a,
                    )
                    x_out_b = _run_stage_compute(
                        wan_b, tenant_b, stage, num_stages, layers_per_stage, x_in_b,
                    )
                x_ins[base] = x_out_a
                x_ins[partner] = x_out_b

            c1 = torch.cuda.Event(enable_timing=True)
            c1.record()
            slot_compute_ms = _cuda_elapsed_ms(c0, c1)

            if stage == num_stages - 1 and pp_rank == last_rank:
                p0 = torch.cuda.Event(enable_timing=True)
                p0.record()
                for m in mb_tenants:
                    pending_noise[m] = _active_wan(runner, tenants[m].scheduler).post_infer.infer(
                        x_ins[m], tenants[m].pre,
                    )[0]
                p1 = torch.cuda.Event(enable_timing=True)
                p1.record()
                _profile_add(profile, "post_infer_ms", _cuda_elapsed_ms(p0, p1))
            elif stage < num_stages - 1:
                for m in mb_tenants:
                    outputs[(m, stage)] = x_ins[m]

        if not opts.barrier_after_p2p and _slot_needs_barrier(slot0, slot1, opts.slot_barrier_mode):
            t_bar = time.perf_counter()
            _sync_step_group(pp_group, seq_parallel)
            slot_barrier_ms = (time.perf_counter() - t_bar) * 1000.0

        noise_mbs: list[int] = []
        for slot in (slot0, slot1):
            if slot is not None and slot[1] == num_stages - 1:
                noise_mbs.append(slot[0])
        for mb in sorted(set(noise_mbs)):
            t_noise = time.perf_counter()
            for m in tenants_for_mb(mb, seq_p_size, layout):
                if pp_rank == last_rank:
                    send_noise_pred(pending_noise[m], 0, pp_group, _tag_noise(m))
                if pp_rank == 0:
                    noise = recv_noise_pred(last_rank, pp_group, device, _tag_noise(m))
                    tenant = tenants[m]
                    tenant.scheduler.noise_pred_cond = noise
                    tenant.scheduler.noise_pred_uncond = None
                    tenant.scheduler.noise_pred_guided = noise
                    tenant.scheduler.noise_pred = noise
            slot_noise_ms += (time.perf_counter() - t_noise) * 1000.0

        send_plans: list[tuple[int, int]] = []
        for slot in (slot0, slot1):
            if slot is not None and slot[1] < num_stages - 1:
                send_plans.append((slot[0], slot[1]))
        send_plans.sort()
        act_batch: P2pAsyncSend | None = P2pAsyncSend() if opts.async_activation_p2p else None
        for mb, stage in send_plans:
            sender = pp_stage_owner(stage, pp_size)
            receiver = pp_stage_owner(stage + 1, pp_size)
            t_act = time.perf_counter()
            for m in tenants_for_mb(mb, seq_p_size, layout):
                tag = _tag_activation(m, stage)
                if opts.async_activation_p2p and pp_rank == sender:
                    tensor = outputs.get((m, stage))
                    assert tensor is not None and act_batch is not None
                    _async_send_activation(tensor, receiver, pp_group, tag, act_batch)
                elif opts.async_activation_p2p and pp_rank == receiver:
                    pending_x[m] = recv_activation(sender, pp_group, device, tag)
                else:
                    x_next = _paired_activation_p2p(
                        pp_rank,
                        pp_group,
                        device,
                        sender,
                        receiver,
                        m,
                        stage,
                        outputs.get((m, stage)) if pp_rank == sender else None,
                    )
                    if pp_rank == receiver:
                        assert x_next is not None
                        pending_x[m] = x_next
            if act_batch is not None and pp_rank == sender:
                act_batch.wait()
            slot_act_ms += (time.perf_counter() - t_act) * 1000.0

        if opts.barrier_after_p2p and _slot_needs_barrier(slot0, slot1, opts.slot_barrier_mode):
            t_bar = time.perf_counter()
            _sync_step_group(pp_group, seq_parallel)
            slot_barrier_ms = (time.perf_counter() - t_bar) * 1000.0

        slot_local_ms = (time.perf_counter() - t_slot) * 1000.0
        _profile_max_pp(profile, pp_group, "slot_critical_ms", slot_local_ms)
        _profile_max_pp(profile, pp_group, "slot_compute_ms", slot_compute_ms)
        _profile_max_pp(profile, pp_group, "slot_meta_ms", slot_meta_ms)
        _profile_max_pp(profile, pp_group, "slot_pre_infer_ms", slot_pre_infer_ms)
        _profile_max_pp(profile, pp_group, "slot_barrier_ms", slot_barrier_ms)
        _profile_max_pp(profile, pp_group, "slot_noise_p2p_ms", slot_noise_ms)
        _profile_max_pp(profile, pp_group, "slot_activation_p2p_ms", slot_act_ms)
        _profile_max_pp(profile, pp_group, "slot_orch_sync_ms", slot_orch_sync_ms)
        if my_slot is not None:
            _profile_add(profile, "stage_compute_ms", slot_compute_ms)

    if use_sp_dual and orch is not None and opts.orch_sync_per_step:
        _finish_orch_streams(orch, defer=opts.defer_orch_stream_sync)


def run_gpipe_quad_sp_pipeline(
    runner: MultiModelStruct,
    tenants: list[QuadPpTenantCtx],
    payloads: list[dict[str, Any]],
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int,
    *,
    seq_p_size: int = 2,
    layout: QuadTenantLayout = "oct",
    seq_parallel: bool,
    sp_overlap: bool,
    profile: dict[str, float] | None = None,
    minimal_step_sync: bool = False,
    pipeline_opts: QuadPipelineOpts | None = None,
    per_request_latency_s: dict[int, float] | None = None,
) -> dict[str, int] | None:
    n_tenants = num_quad_tenants(seq_p_size, layout)
    assert len(tenants) == len(payloads) == n_tenants
    wall_t0 = time.perf_counter()
    for tenant, payload in zip(tenants, payloads):
        tenant.scheduler.prepare(
            seed=int(payload["seed"]),
            latent_shape=payload["latent_shape"],
            image_encoder_output=payload["image_encoder_output"],
        )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    orch: A2AOrchestrator | None = None
    stats: dict[str, int] | None = None
    if sp_overlap and seq_parallel:
        orch = A2AOrchestrator(device)
        orch.install()

    n_steps = tenants[0].scheduler.infer_steps
    try:
        for step_index in range(n_steps):
            for tenant in tenants:
                tenant.scheduler.step_pre(step_index=step_index)
            if not minimal_step_sync:
                t_bar = time.perf_counter()
                _sync_step_group(pp_group, seq_parallel)
                _profile_add(profile, "step_barrier_ms", (time.perf_counter() - t_bar) * 1000.0)
            gpipe_quad_sp_pipeline_step(
                runner,
                tenants,
                pp_group,
                pp_rank,
                pp_size,
                num_layers,
                layers_per_stage,
                orch,
                seq_p_size=seq_p_size,
                layout=layout,
                seq_parallel=seq_parallel,
                sp_overlap=sp_overlap,
                profile=profile,
                pipeline_opts=pipeline_opts,
            )
            if pp_rank == 0:
                for tenant_id, tenant in enumerate(tenants):
                    tenant.scheduler.step_post()
                    if per_request_latency_s is not None and step_index == n_steps - 1:
                        per_request_latency_s[tenant_id] = time.perf_counter() - wall_t0
            if not minimal_step_sync:
                t_bar = time.perf_counter()
                _sync_step_group(pp_group, seq_parallel)
                _profile_add(profile, "step_barrier_ms", (time.perf_counter() - t_bar) * 1000.0)
        if orch is not None:
            stats = orch.stats_snapshot()
    finally:
        if orch is not None:
            orch.restore()
    return stats
