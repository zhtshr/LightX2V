"""Dual-tenant micro-phase overlap: A.comm async while B.comp (scheme 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Any, Callable

import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTP


class StepKind(Enum):
    COMP = "comp"
    COMM = "comm"


@dataclass
class TenantMicroRunner:
    section: str
    block_idx: int
    mid: Any = None
    step_idx: int = 0
    steps: list[tuple[StepKind, str]] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
    done_flag: bool = False

    def done(self) -> bool:
        return self.done_flag or self.step_idx >= len(self.steps)

    def peek_kind(self) -> StepKind | None:
        if self.done():
            return None
        return self.steps[self.step_idx][0]

    def advance_comp_only(
        self,
        model: Any,
        tenant: Any,
        bind_tenant: Callable,
        ensure_block: Callable,
        capture_ti_snap: Callable,
    ) -> bool:
        if self.peek_kind() != StepKind.COMP:
            return False
        wan, ti = bind_tenant(model, tenant)
        block = ensure_block(tenant, wan, ti, self.block_idx)
        ti.block_idx = self.block_idx
        _run_micro_step(self, wan, ti, tenant, block, capture_ti_snap)
        self.step_idx += 1
        if self.step_idx >= len(self.steps):
            self.done_flag = True
        return True

    def run_remaining(
        self,
        model: Any,
        tenant: Any,
        bind_tenant: Callable,
        ensure_block: Callable,
        capture_ti_snap: Callable,
        orch: TPMicroOrchestrator,
        *,
        overlap: bool,
    ) -> Any | None:
        wan, ti = bind_tenant(model, tenant)
        block = ensure_block(tenant, wan, ti, self.block_idx)
        ti.block_idx = self.block_idx
        orch.overlap_enabled = overlap
        while not self.done():
            _run_micro_step(self, wan, ti, tenant, block, capture_ti_snap)
            self.step_idx += 1
        orch.overlap_enabled = False
        if self.section == "self":
            return self.scratch.get("block_mid")
        return None

    @staticmethod
    def for_self(block_idx: int) -> TenantMicroRunner:
        return TenantMicroRunner(
            section="self",
            block_idx=block_idx,
            steps=[
                (StepKind.COMP, "sa_prep"),
                (StepKind.COMP, "sa_q_linear"),
                (StepKind.COMM, "sa_q_norm"),
                (StepKind.COMP, "sa_k_linear"),
                (StepKind.COMM, "sa_k_norm"),
                (StepKind.COMP, "sa_v_attn"),
                (StepKind.COMM, "sa_o"),
            ],
        )

    @staticmethod
    def for_cross(block_idx: int, mid: Any) -> TenantMicroRunner:
        return TenantMicroRunner(
            section="cross",
            block_idx=block_idx,
            mid=mid,
            steps=[
                (StepKind.COMP, "cx_prep"),
                (StepKind.COMP, "cx_q_linear"),
                (StepKind.COMM, "cx_q_norm"),
                (StepKind.COMP, "cx_k_linear"),
                (StepKind.COMM, "cx_k_norm"),
                (StepKind.COMP, "cx_v_attn"),
                (StepKind.COMM, "cx_o"),
                (StepKind.COMP, "cx_ffn_post"),
            ],
        )


@dataclass
class TPMicroOrchestrator:
    device: torch.device
    enabled: bool = False
    overlap_enabled: bool = False
    in_peer_comp: bool = False
    peer_runner: TenantMicroRunner | None = None
    comm_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream())
    compute_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream())
    ar_calls: int = 0
    ar_overlap_windows: int = 0
    peer_comp_steps: int = 0
    _orig: dict[str, Any] = field(default_factory=dict)
    _bind_tenant: Callable | None = None
    _ensure_block: Callable | None = None
    _capture_ti_snap: Callable | None = None
    _model: Any = None
    _peer_tenant: Any = None

    def stats(self) -> dict[str, int]:
        return {
            "all_reduce_calls": self.ar_calls,
            "all_reduce_overlap_windows": self.ar_overlap_windows,
            "peer_comp_steps": self.peer_comp_steps,
        }

    def set_peer_context(
        self,
        model: Any,
        peer_runner: TenantMicroRunner | None,
        bind_tenant: Callable,
        ensure_block: Callable,
        capture_ti_snap: Callable,
        peer_tenant: Any | None,
    ) -> None:
        self._model = model
        self.peer_runner = peer_runner
        self._bind_tenant = bind_tenant
        self._ensure_block = ensure_block
        self._capture_ti_snap = capture_ti_snap
        self._peer_tenant = peer_tenant

    def install(self) -> None:
        if self._orig:
            return
        self._orig["mm_apply"] = MMWeightTP.apply
        self._orig["rms_apply"] = RMSWeightTP.apply
        orch = self

        def mm_patched(self_mm, input_tensor):
            out = orch._orig["mm_apply"](self_mm, input_tensor)
            if self_mm.split_dim == "row" and self_mm.tp_size > 1 and self_mm.tp_group is not None:
                if orch.in_peer_comp:
                    raise RuntimeError("peer comp must not trigger row AR")
                orch._handle_ar(lambda async_op: dist.all_reduce(
                    out, op=dist.ReduceOp.SUM, group=self_mm.tp_group, async_op=async_op,
                ))
                if self_mm._row_split_bias is not None:
                    out = out + self_mm._row_split_bias
            return out

        def rms_patched(self_rms, input_tensor):
            if orch.in_peer_comp:
                raise RuntimeError("peer comp must not trigger RMS AR")
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

    def _advance_peer_comp(self) -> bool:
        if (
            not self.overlap_enabled
            or self.peer_runner is None
            or self._model is None
            or self._bind_tenant is None
            or self._ensure_block is None
            or self._capture_ti_snap is None
            or self._peer_tenant is None
        ):
            return False
        self.in_peer_comp = True
        try:
            progressed = self.peer_runner.advance_comp_only(
                self._model,
                self._peer_tenant,
                self._bind_tenant,
                self._ensure_block,
                self._capture_ti_snap,
            )
            if progressed:
                self.peer_comp_steps += 1
            return progressed
        finally:
            self.in_peer_comp = False

    def _handle_ar(self, launch: Callable[..., Any]) -> None:
        self.ar_calls += 1
        if not self.enabled or not self.overlap_enabled or self.in_peer_comp:
            with torch.cuda.stream(self.comm_stream):
                launch(async_op=False)
            return

        self.ar_overlap_windows += 1
        with torch.cuda.stream(self.comm_stream):
            work = launch(async_op=True)
        with torch.cuda.stream(self.compute_stream):
            while work is not None and not work.is_completed():
                if not self._advance_peer_comp():
                    break
        if work is not None and hasattr(work, "wait"):
            work.wait()


def _run_micro_step(
    runner: TenantMicroRunner,
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    capture_ti_snap: Callable,
) -> None:
    _, name = runner.steps[runner.step_idx]
    if runner.section == "self":
        _self_step(name, runner, wan, ti, tenant, block, capture_ti_snap)
    else:
        _cross_step(name, runner, wan, ti, tenant, block, capture_ti_snap)


def _self_step(
    name: str,
    runner: TenantMicroRunner,
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    capture_ti_snap: Callable,
) -> None:
    phase = block.compute_phases[0]
    pre = tenant.pre_infer_out
    x = tenant.x
    if name == "sa_prep":
        if hasattr(phase, "before_proj") and phase.before_proj.weight is not None:
            x = phase.before_proj.apply(x) + pre.x
        mods = ti.pre_process(phase.modulation, pre.embed0)
        shift_msa, scale_msa = mods[0], mods[1]
        if hasattr(phase, "smooth_norm1_weight"):
            nw = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            nb = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
            norm1_out = phase.norm1.apply(x)
            if ti.sensitive_layer_dtype != ti.infer_dtype:
                norm1_out = norm1_out.to(ti.sensitive_layer_dtype)
            norm1_out.mul_(nw).add_(nb)
        else:
            norm1_out = phase.norm1.apply(x)
            if ti.sensitive_layer_dtype != ti.infer_dtype:
                norm1_out = norm1_out.to(ti.sensitive_layer_dtype)
            norm1_out = ti.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            norm1_out = norm1_out.to(ti.infer_dtype)
        runner.scratch.update(x=x, mods=mods, norm1_out=norm1_out)
        tenant.x = x
        return

    if name == "sa_q_linear":
        runner.scratch["q_in"] = phase.self_attn_q.apply(runner.scratch["norm1_out"])
        return

    if name == "sa_q_norm":
        n, d = ti.num_heads, ti.head_dim
        q = phase.self_attn_norm_q.apply(runner.scratch["q_in"]).view(-1, n, d)
        runner.scratch["q"] = q
        return

    if name == "sa_k_linear":
        runner.scratch["k_in"] = phase.self_attn_k.apply(runner.scratch["norm1_out"])
        return

    if name == "sa_k_norm":
        n, d = ti.num_heads, ti.head_dim
        k = phase.self_attn_norm_k.apply(runner.scratch["k_in"]).view(-1, n, d)
        runner.scratch["k"] = k
        return

    if name == "sa_v_attn":
        n, d = ti.num_heads, ti.head_dim
        s = runner.scratch["q"].shape[0]
        v = phase.self_attn_v.apply(runner.scratch["norm1_out"]).view(s, n, d)
        q, k = runner.scratch["q"], runner.scratch["k"]
        q, k = ti.apply_rope_func(q, k, ti.cos_sin)
        if os.environ.get("LIGHTX2V_MOCK_ATTN") == "1":
            gemm = torch.randn(4096, 4096, device=q.device, dtype=torch.float16)
            torch.matmul(gemm, gemm)
            runner.scratch["attn_out"] = torch.randn(s, n * d, device=q.device, dtype=q.dtype)
            return
        if ti.self_attn_cu_seqlens_qkv is None:
            ti.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32).cumsum(0)
        attn_out = phase.self_attn_1.apply(
            q=q, k=k, v=v,
            cu_seqlens_q=ti.self_attn_cu_seqlens_qkv,
            cu_seqlens_kv=ti.self_attn_cu_seqlens_qkv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )
        runner.scratch["attn_out"] = attn_out
        return

    if name == "sa_o":
        y_out = phase.self_attn_o.apply(runner.scratch["attn_out"])
        from scripts.disagg.run_phase3_dual_overlap_bench import BlockMid  # noqa: PLC0415
        runner.scratch["block_mid"] = BlockMid(mods=runner.scratch["mods"], y_out=y_out)
        tenant.ti_snap = capture_ti_snap(ti)
        return

    raise RuntimeError(f"unknown self step {name}")


def _cross_context(ti: Any, context: torch.Tensor) -> tuple[Any, Any | None]:
    if ti.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and ti.config.get("use_image_encoder", True):
        return context[257:], context[:257]
    return context, None


def _cross_step(
    name: str,
    runner: TenantMicroRunner,
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    capture_ti_snap: Callable,
) -> None:
    pc, pf = block.compute_phases[1], block.compute_phases[2]
    mid = runner.mid
    _, _, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mid.mods
    n, d = ti.num_heads, ti.head_dim

    if name == "cx_prep":
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            x = tenant.x.to(ti.sensitive_layer_dtype) + mid.y_out.to(ti.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x = tenant.x.clone()
            x.add_(mid.y_out * gate_msa.squeeze())
        norm3_out = pc.norm3.apply(x)
        ctx, ctx_img = _cross_context(ti, tenant.pre_infer_out.context)
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            ctx = ctx.to(ti.infer_dtype)
            if ctx_img is not None:
                ctx_img = ctx_img.to(ti.infer_dtype)
        runner.scratch.update(x=x, norm3_out=norm3_out, ctx=ctx, ctx_img=ctx_img)
        return

    if name == "cx_q_linear":
        runner.scratch["q_in"] = pc.cross_attn_q.apply(runner.scratch["norm3_out"])
        return

    if name == "cx_q_norm":
        q = pc.cross_attn_norm_q.apply(runner.scratch["q_in"]).view(-1, n, d)
        runner.scratch["q"] = q
        return

    if name == "cx_k_linear":
        runner.scratch["k_in"] = pc.cross_attn_k.apply(runner.scratch["ctx"])
        return

    if name == "cx_k_norm":
        k_full = pc.cross_attn_norm_k.apply(runner.scratch["k_in"]).view(-1, ti.global_num_heads, d)
        runner.scratch["k_full"] = k_full
        return

    if name == "cx_v_attn":
        q = runner.scratch["q"]
        k_full = runner.scratch["k_full"]
        v_full = pc.cross_attn_v.apply(runner.scratch["ctx"]).view(-1, ti.global_num_heads, d)
        if ti.tp_size > 1:
            hs = ti.tp_rank * n
            k, v = k_full[:, hs : hs + n, :], v_full[:, hs : hs + n, :]
        else:
            k, v = k_full, v_full
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
        ctx_img = runner.scratch.get("ctx_img")
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
        runner.scratch["attn_out"] = attn_out
        return

    if name == "cx_o":
        x = runner.scratch["x"]
        attn_out = pc.cross_attn_o.apply(runner.scratch["attn_out"])
        x = x.clone()
        x.add_(attn_out)
        runner.scratch["x_ffn"] = x
        return

    if name == "cx_ffn_post":
        x = runner.scratch["x_ffn"]
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
        y = pf.ffn_2.apply(y)
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            x = x.to(ti.sensitive_layer_dtype) + y.to(ti.sensitive_layer_dtype) * c_gate_msa.squeeze()
        else:
            x.add_(y * c_gate_msa.squeeze())
        tenant.x = x
        tenant.ti_snap = capture_ti_snap(ti)
        return

    raise RuntimeError(f"unknown cross step {name}")


def micro_overlap_main_blocks(
    model: Any,
    tenant_a: Any,
    tenant_b: Any,
    orch: TPMicroOrchestrator,
    bind_tenant: Callable,
    ensure_block: Callable,
    preload_blocks: Callable,
    capture_ti_snap: Callable,
    *,
    overlap: bool,
) -> None:
    wan_a, ti_a = bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = bind_tenant(model, tenant_b)
    preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    orch.enabled = True

    def _run_self(tenant: Any, block_idx: int, peer_tenant: Any, peer_cross: TenantMicroRunner | None) -> Any:
        runner = TenantMicroRunner.for_self(block_idx)
        if overlap and peer_cross is not None:
            orch.set_peer_context(model, peer_cross, bind_tenant, ensure_block, capture_ti_snap, peer_tenant)
            return runner.run_remaining(
                model, tenant, bind_tenant, ensure_block, capture_ti_snap, orch, overlap=True,
            )
        orch.set_peer_context(model, None, bind_tenant, ensure_block, capture_ti_snap, None)
        return runner.run_remaining(
            model, tenant, bind_tenant, ensure_block, capture_ti_snap, orch, overlap=False,
        )

    def _finish_cross_runner(tenant: Any, runner: TenantMicroRunner | None) -> None:
        if runner is None or runner.done():
            return
        orch.set_peer_context(model, None, bind_tenant, ensure_block, capture_ti_snap, None)
        runner.run_remaining(
            model, tenant, bind_tenant, ensure_block, capture_ti_snap, orch, overlap=False,
        )

    def _run_cross_sync(tenant: Any, block_idx: int, mid: Any) -> None:
        runner = TenantMicroRunner.for_cross(block_idx, mid)
        orch.set_peer_context(model, None, bind_tenant, ensure_block, capture_ti_snap, None)
        runner.run_remaining(
            model, tenant, bind_tenant, ensure_block, capture_ti_snap, orch, overlap=False,
        )

    mid_a = _run_self(tenant_a, 0, tenant_b, None)

    for k in range(num_blocks):
        peer_a_cross = TenantMicroRunner.for_cross(k, mid_a) if overlap else None
        mid_b = _run_self(tenant_b, k, tenant_a, peer_a_cross)
        if overlap:
            _finish_cross_runner(tenant_a, peer_a_cross)
        else:
            _run_cross_sync(tenant_a, k, mid_a)

        if k + 1 < num_blocks:
            peer_b_cross = TenantMicroRunner.for_cross(k, mid_b) if overlap else None
            mid_a = _run_self(tenant_a, k + 1, tenant_b, peer_b_cross)
            if overlap:
                _finish_cross_runner(tenant_b, peer_b_cross)
            else:
                _run_cross_sync(tenant_b, k, mid_b)
        else:
            _run_cross_sync(tenant_b, k, mid_b)

    orch.set_peer_context(model, None, bind_tenant, ensure_block, capture_ti_snap, None)
    orch.enabled = False
    orch.comm_stream.synchronize()
    orch.compute_stream.synchronize()


def run_dual_micro_pipeline(
    model: Any,
    tenant_a: Any,
    tenant_b: Any,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    overlap: bool,
    steps: int | None,
    bind_tenant: Callable,
    ensure_block: Callable,
    preload_blocks: Callable,
    pre_infer_tenant: Callable,
    finish_step_tenant: Callable,
    capture_ti_snap: Callable,
    time_fn: Callable[[Callable[[], None]], float],
) -> tuple[float, dict[str, int]]:
    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = TPMicroOrchestrator(device)
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
            micro_overlap_main_blocks(
                model, tenant_a, tenant_b, orch,
                bind_tenant, ensure_block, preload_blocks, capture_ti_snap,
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
