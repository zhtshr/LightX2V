"""TP fine-grained dual-tenant overlap: pump AR-free cross_ffn slices on each all_reduce."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTP


class _StopPump(Exception):
    """Peer hit an AR boundary while pumping (must not run AR during pump)."""


@dataclass
class CrossFFNPumpState:
    block_idx: int
    mid: Any
    pump_seg: int = 0  # next AR-free segment to pump (0,1,2)
    phase: str = "init"  # init | after_pre_cross | after_mid | done
    cross_o_done: bool = False
    x: Any = None
    attn_out: Any = None
    q: Any = None
    k_full: Any = None
    v_full: Any = None
    ctx_img: Any = None
    y_partial: Any = None

    def pumpable_remaining(self) -> int:
        return max(0, 4 - self.pump_seg)


@dataclass
class TPFineOrchestrator:
    device: torch.device
    enabled: bool = False
    overlap_enabled: bool = False
    pumping: bool = False
    comm_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream())
    compute_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream())
    ar_calls: int = 0
    ar_overlap_windows: int = 0
    pump_segments_run: int = 0
    pump_fn: Callable[[], bool] | None = None
    _orig: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> dict[str, int]:
        return {
            "all_reduce_calls": self.ar_calls,
            "all_reduce_overlap_windows": self.ar_overlap_windows,
            "pump_segments_run": self.pump_segments_run,
        }

    def install(self) -> None:
        if self._orig:
            return
        self._orig["mm_apply"] = MMWeightTP.apply
        self._orig["rms_apply"] = RMSWeightTP.apply
        orch = self

        def mm_patched(self_mm, input_tensor):
            out = orch._orig["mm_apply"](self_mm, input_tensor)
            if self_mm.split_dim == "row" and self_mm.tp_size > 1 and self_mm.tp_group is not None:
                if orch.pumping:
                    raise _StopPump
                orch._handle_ar(lambda async_op: dist.all_reduce(
                    out, op=dist.ReduceOp.SUM, group=self_mm.tp_group, async_op=async_op,
                ))
                if self_mm._row_split_bias is not None:
                    out = out + self_mm._row_split_bias
            return out

        def rms_patched(self_rms, input_tensor):
            if orch.pumping:
                raise _StopPump
            if getattr(self_rms, "use_p2p_norm", False):
                return orch._orig["rms_apply"](self_rms, input_tensor)
            local_sum = input_tensor.pow(2).sum(-1, keepdim=True)
            if self_rms.tp_size > 1 and self_rms.tp_group is not None:
                orch._handle_ar(lambda async_op: dist.all_reduce(
                    local_sum, op=dist.ReduceOp.SUM, group=self_rms.tp_group, async_op=async_op,
                ))
            hidden_dim = input_tensor.shape[-1] * self_rms.tp_size
            global_mean = local_sum / hidden_dim
            if self_rms.sensitive_layer_dtype != self_rms.infer_dtype:
                input_tensor = input_tensor * torch.rsqrt(global_mean.float() + self_rms.eps).to(self_rms.infer_dtype)
                input_tensor = (input_tensor * self_rms._get_actual_weight()).to(self_rms.infer_dtype)
            else:
                input_tensor = input_tensor * torch.rsqrt(global_mean + self_rms.eps)
                input_tensor = input_tensor * self_rms._get_actual_weight()
            return input_tensor

        MMWeightTP.apply = mm_patched
        RMSWeightTP.apply = rms_patched

    def restore(self) -> None:
        if "mm_apply" in self._orig:
            MMWeightTP.apply = self._orig["mm_apply"]
        if "rms_apply" in self._orig:
            RMSWeightTP.apply = self._orig["rms_apply"]
        self._orig.clear()

    def _handle_ar(self, launch: Callable[..., Any]) -> None:
        self.ar_calls += 1
        if not self.enabled or not self.overlap_enabled or self.pumping:
            with torch.cuda.stream(self.comm_stream):
                launch(async_op=False)
            return
        if self.pump_fn is None:
            with torch.cuda.stream(self.comm_stream):
                launch(async_op=False)
            return

        self.ar_overlap_windows += 1
        with torch.cuda.stream(self.comm_stream):
            work = launch(async_op=True)
        with torch.cuda.stream(self.compute_stream):
            while work is not None and not work.is_completed():
                progressed = self.pump_fn()
                if progressed:
                    self.pump_segments_run += 1
                else:
                    break
        if work is not None and hasattr(work, "wait"):
            work.wait()

    def set_pump(self, pump_fn: Callable[[], bool] | None) -> None:
        self.pump_fn = pump_fn


def _cross_context_split(ti: Any, context: torch.Tensor) -> tuple[Any, Any | None]:
    if ti.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and ti.config.get("use_image_encoder", True):
        return context[257:], context[:257]
    return context, None


def pump_cross_ffn_ar_free_segment(
    wan: Any,
    ti: Any,
    tenant: Any,
    state: CrossFFNPumpState,
) -> bool:
    """Execute one AR-free cross_ffn slice for overlap pumping."""
    if state.pump_seg >= 4 or state.phase == "done":
        return False

    block = wan.transformer_weights.blocks[state.block_idx]
    pc = block.compute_phases[1]
    pf = block.compute_phases[2]
    _, _, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = state.mid.mods
    n, d = ti.num_heads, ti.head_dim

    if state.pump_seg == 0:
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            x = tenant.x.to(ti.sensitive_layer_dtype) + state.mid.y_out.to(ti.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x = tenant.x.clone()
            x.add_(state.mid.y_out * gate_msa.squeeze())
        norm3_out = pc.norm3.apply(x)
        ctx, ctx_img = _cross_context_split(ti, tenant.pre_infer_out.context)
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            ctx = ctx.to(ti.infer_dtype)
            if ctx_img is not None:
                ctx_img = ctx_img.to(ti.infer_dtype)
        q = pc.cross_attn_q.apply(norm3_out)
        k_raw = pc.cross_attn_k.apply(ctx)
        v_full = pc.cross_attn_v.apply(ctx).view(-1, ti.global_num_heads, d)
        state.x, state.q, state.k_full, state.v_full, state.ctx_img = x, q, k_raw, v_full, ctx_img
        state.phase = "after_pre_cross"
        state.pump_seg = 1
        return True

    if state.pump_seg == 1 and state.attn_out is None:
        q = pc.cross_attn_norm_q.apply(state.q).view(-1, n, d)
        k_full = pc.cross_attn_norm_k.apply(state.k_full).view(-1, ti.global_num_heads, d)
        ctx_img = state.ctx_img
        if ti.tp_size > 1:
            hs = ti.tp_rank * n
            k, v = k_full[:, hs : hs + n, :], state.v_full[:, hs : hs + n, :]
        else:
            k, v = k_full, state.v_full
        if ti.cross_attn_cu_seqlens_q is None:
            ti.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32).cumsum(0)
        if ti.cross_attn_cu_seqlens_kv is None:
            ti.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]], device=k.device, dtype=torch.int32).cumsum(0)
        attn_out = pc.cross_attn_1.apply(
            q=q, k=k, v=v,
            cu_seqlens_q=ti.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=ti.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )
        if ctx_img is not None:
            k_img_full = pc.cross_attn_norm_k_img.apply(pc.cross_attn_k_img.apply(ctx_img)).view(-1, ti.global_num_heads, d)
            v_img_full = pc.cross_attn_v_img.apply(ctx_img).view(-1, ti.global_num_heads, d)
            if ti.tp_size > 1:
                hs = ti.tp_rank * n
                k_img, v_img = k_img_full[:, hs : hs + n, :], v_img_full[:, hs : hs + n, :]
            else:
                k_img, v_img = k_img_full, v_img_full
            if ti.cross_attn_cu_seqlens_kv_img is None:
                ti.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]], device=k_img.device, dtype=torch.int32).cumsum(0)
            attn_out = attn_out + pc.cross_attn_2.apply(
                q=q, k=k_img, v=v_img,
                cu_seqlens_q=ti.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=ti.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
        state.attn_out = attn_out
        state.pump_seg = 2
        return True

    if state.pump_seg == 2:
        if not state.cross_o_done:
            return False
        x = state.x
        if hasattr(pf, "smooth_norm2_weight"):
            nw = (1 + c_scale_msa.squeeze()) * pf.smooth_norm2_weight.tensor
            nb = c_shift_msa.squeeze() * pf.smooth_norm2_bias.tensor
            norm2_out = pf.norm2.apply(x)
            if ti.sensitive_layer_dtype != ti.infer_dtype:
                norm2_out = norm2_out.to(ti.sensitive_layer_dtype)
            norm2_out.mul_(nw).add_(nb)
        else:
            norm2_out = pf.norm2.apply(x)
            if ti.sensitive_layer_dtype != ti.infer_dtype:
                norm2_out = norm2_out.to(ti.sensitive_layer_dtype)
            norm2_out = ti.modulate_func(norm2_out, scale=c_scale_msa, shift=c_shift_msa).squeeze()
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            norm2_out = norm2_out.to(ti.infer_dtype)
        state.y_partial = norm2_out
        state.phase = "after_mid"
        state.pump_seg = 3
        return True

    if state.pump_seg == 3:
        if state.y_partial is None:
            return False
        y = torch.nn.functional.gelu(state.y_partial, approximate="tanh")
        state.y_partial = y
        state.pump_seg = 4
        return True

    return False


def complete_cross_ffn_from_state(
    wan: Any,
    ti: Any,
    tenant: Any,
    state: CrossFFNPumpState,
) -> None:
    """Finish cross_ffn on primary path (AR ops), continuing from pumped state."""
    if state.phase == "done":
        return
    block = wan.transformer_weights.blocks[state.block_idx]
    pc, pf = block.compute_phases[1], block.compute_phases[2]
    c_shift_msa, c_scale_msa, c_gate_msa = state.mid.mods[3], state.mid.mods[4], state.mid.mods[5]

    if state.phase == "init":
        pump_cross_ffn_ar_free_segment(wan, ti, tenant, state)

    if state.phase == "after_pre_cross":
        if state.attn_out is None:
            pump_cross_ffn_ar_free_segment(wan, ti, tenant, state)
        attn_out = pc.cross_attn_o.apply(state.attn_out)
        x = state.x.clone()
        x.add_(attn_out)
        state.cross_o_done = True
        if state.pump_seg < 3:
            if hasattr(pf, "smooth_norm2_weight"):
                nw = (1 + c_scale_msa.squeeze()) * pf.smooth_norm2_weight.tensor
                nb = c_shift_msa.squeeze() * pf.smooth_norm2_bias.tensor
                norm2_out = pf.norm2.apply(x)
                if ti.sensitive_layer_dtype != ti.infer_dtype:
                    norm2_out = norm2_out.to(ti.sensitive_layer_dtype)
                norm2_out.mul_(nw).add_(nb)
            else:
                norm2_out = pf.norm2.apply(x)
                if ti.sensitive_layer_dtype != ti.infer_dtype:
                    norm2_out = norm2_out.to(ti.sensitive_layer_dtype)
                norm2_out = ti.modulate_func(norm2_out, scale=c_scale_msa, shift=c_shift_msa).squeeze()
            if ti.sensitive_layer_dtype != ti.infer_dtype:
                norm2_out = norm2_out.to(ti.infer_dtype)
            y = pf.ffn_0.apply(norm2_out)
            y = torch.nn.functional.gelu(y, approximate="tanh")
            state.y_partial = y
        elif state.pump_seg == 3:
            y = pf.ffn_0.apply(state.y_partial)
            y = torch.nn.functional.gelu(y, approximate="tanh")
            state.y_partial = y
        state.x = x
        state.phase = "after_mid"

    if state.phase == "after_mid":
        y = pf.ffn_2.apply(state.y_partial)
        x = state.x
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            x = x.to(ti.sensitive_layer_dtype) + y.to(ti.sensitive_layer_dtype) * c_gate_msa.squeeze()
        else:
            x.add_(y * c_gate_msa.squeeze())
        tenant.x = x
        state.phase = "done"


def fine_overlap_main_blocks(
    model: Any,
    tenant_a: Any,
    tenant_b: Any,
    orch: TPFineOrchestrator,
    bind_tenant: Callable,
    run_self_attn_block: Callable,
    preload_blocks: Callable,
    *,
    overlap: bool,
) -> None:
    """Layer schedule with per-AR fine overlap (pump cross_ffn AR-free slices)."""
    wan_a, ti_a = bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = bind_tenant(model, tenant_b)
    preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    orch.enabled = True
    orch.overlap_enabled = False
    orch.set_pump(None)

    pump_states: dict[str, CrossFFNPumpState] = {}

    def _set_pump_for(peer: Any, block_idx: int, mid: Any) -> None:
        key = f"{peer.name}_{block_idx}"
        st = pump_states.get(key)
        if st is None or st.block_idx != block_idx:
            st = CrossFFNPumpState(block_idx=block_idx, mid=mid)
            pump_states[key] = st
        orch.set_pump(make_pump_fn(model, peer, bind_tenant, st, orch))

    def _finish_cross(tenant: Any, block_idx: int, mid: Any) -> None:
        key = f"{tenant.name}_{block_idx}"
        st = pump_states.get(key)
        if st is None:
            st = CrossFFNPumpState(block_idx=block_idx, mid=mid)
        wan, ti = bind_tenant(model, tenant)
        if overlap and st.pump_seg > 0:
            complete_cross_ffn_from_state(wan, ti, tenant, st)
        else:
            from scripts.disagg.run_phase3_dual_overlap_bench import _run_cross_ffn_block  # noqa: PLC0415
            _run_cross_ffn_block(wan, ti, tenant, block_idx, mid)
        pump_states.pop(key, None)

    mid_a = run_self_attn_block(wan_a, ti_a, tenant_a, 0)

    for k in range(num_blocks):
        if overlap:
            orch.overlap_enabled = True
            _set_pump_for(tenant_a, k, mid_a)
        else:
            orch.overlap_enabled = False
            orch.set_pump(None)

        wan_b, ti_b = bind_tenant(model, tenant_b)
        mid_b = run_self_attn_block(wan_b, ti_b, tenant_b, k)

        orch.overlap_enabled = False
        orch.set_pump(None)
        _finish_cross(tenant_a, k, mid_a)

        if k + 1 < num_blocks:
            if overlap:
                orch.overlap_enabled = True
                _set_pump_for(tenant_b, k, mid_b)
            wan_a, ti_a = bind_tenant(model, tenant_a)
            mid_a = run_self_attn_block(wan_a, ti_a, tenant_a, k + 1)
            orch.overlap_enabled = False
            orch.set_pump(None)
            _finish_cross(tenant_b, k, mid_b)
        else:
            orch.overlap_enabled = False
            orch.set_pump(None)

    orch.overlap_enabled = False
    orch.set_pump(None)
    orch.enabled = False
    orch.comm_stream.synchronize()
    orch.compute_stream.synchronize()


def make_pump_fn(
    model: Any,
    tenant: Any,
    bind_tenant: Callable,
    state: CrossFFNPumpState,
    orch: TPFineOrchestrator,
) -> Callable[[], bool]:
    def pump() -> bool:
        if state.pump_seg >= 3:
            return False
        wan, ti = bind_tenant(model, tenant)
        ti.block_idx = state.block_idx
        orch.pumping = True
        try:
            return pump_cross_ffn_ar_free_segment(wan, ti, tenant, state)
        except _StopPump:
            return False
        finally:
            orch.pumping = False

    return pump


def run_dual_fine_pipeline(
    model: Any,
    tenant_a: Any,
    tenant_b: Any,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    overlap: bool,
    steps: int | None,
    bind_tenant: Callable,
    run_self_attn_block: Callable,
    preload_blocks: Callable,
    pre_infer_tenant: Callable,
    finish_step_tenant: Callable,
    time_fn: Callable[[Callable[[], None]], float],
) -> tuple[float, dict[str, int]]:
    """Dual-tenant denoise with per-AR fine overlap on MMWeightTP/RMSWeightTP all_reduce."""
    import torch.distributed as dist  # noqa: PLC0415

    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = TPFineOrchestrator(device)
    orch.install()

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
    tenant_a.inputs = payload_a["inputs"]
    tenant_b.inputs = payload_b["inputs"]

    n_steps = steps if steps is not None else tenant_a.scheduler.infer_steps

    def _body() -> None:
        for step_index in range(n_steps):
            tenant_a.scheduler.step_pre(step_index=step_index)
            tenant_b.scheduler.step_pre(step_index=step_index)
            pre_infer_tenant(model, tenant_a)
            pre_infer_tenant(model, tenant_b)
            fine_overlap_main_blocks(
                model, tenant_a, tenant_b, orch,
                bind_tenant, run_self_attn_block, preload_blocks,
                overlap=overlap,
            )
            finish_step_tenant(model, tenant_a)
            finish_step_tenant(model, tenant_b)
            tenant_a.scheduler.step_post()
            tenant_b.scheduler.step_post()

    try:
        return time_fn(_body), orch.stats()
    finally:
        orch.restore()
