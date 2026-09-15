"""6-phase dual-tenant pipeline: pure COMM phases overlap COMP phases.

Per tenant per layer:
  1 self COMP (through O matmul, no AR) | 2 self O row AR (NCCL)
  3 cross COMP (through O matmul)      | 4 cross O row AR (NCCL)
  5 FFN COMP (through ffn_2 matmul)    | 6 ffn_2 row AR + residual (NCCL)

Schedule (layer k):
  bootstrap A1(0)
  A2(k)∥B1(k), B2(k)∥A3(k), A4(k)∥B3(k), B4(k)∥A5(k), A6(k)∥B5(k), B6(k)∥A1(k+1)

Overlap: async **NCCL** row AR on comm_stream; peer COMP micro-steps on compute_stream.

Norm scalar all-reduce (small AR) must use ``tp_norm_p2p=True`` (IPC P2P), not NCCL.
Optional ``use_comm_p2p=True`` replaces row AR with P2P for experiments only.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

import scripts.disagg.tp_micro_overlap as micro
from scripts.disagg.tp_overlap_cuda_graph import PumpTailCudaGraph
from lightx2v.common.ops.tp_row_ar_nccl import resolve_row_ar_group
from lightx2v.utils.profiler import no_sync_profiling


def _sync_orch_streams(orch: PhasePumpOrch) -> None:
    """Pair-boundary wait: comm + compute streams only (not device-wide)."""
    orch.comm_stream.synchronize()
    orch.compute_stream.synchronize()


@dataclass
class BlockPipelineState:
    self_scratch: dict[str, Any] = field(default_factory=dict)
    cross_scratch: dict[str, Any] = field(default_factory=dict)
    mid: Any = None


_SELF_P1 = ("sa_prep", "sa_q_linear", "sa_q_norm", "sa_k_linear", "sa_k_norm", "sa_v_attn")
_CROSS_P3 = ("cx_prep", "cx_q_linear", "cx_q_norm", "cx_k_linear", "cx_k_norm", "cx_v_attn")
_FFN_P5 = ("ffn_norm2", "ffn0_gelu", "ffn2_linear")
_COMM_PHASES = frozenset({2, 4, 6})
_COMP_PHASES = frozenset({1, 3, 5})
# Heavy attention/GEMM steps deferred until row NCCL completes (avoid BW serial tail).
_DEFER_HEAVY_BEFORE: dict[int, str] = {1: "sa_v_attn", 3: "cx_v_attn"}


def _trim_self_for_comm(scratch: dict[str, Any]) -> None:
    keep = {k: scratch[k] for k in ("o_partial", "mods") if k in scratch}
    scratch.clear()
    scratch.update(keep)


def _trim_cross_for_comm(scratch: dict[str, Any]) -> None:
    keep = {k: scratch[k] for k in ("o_partial", "x") if k in scratch}
    scratch.clear()
    scratch.update(keep)


def _trim_cross_for_ffn_comm(scratch: dict[str, Any]) -> None:
    keep = {k: scratch[k] for k in ("y_partial", "x_ffn") if k in scratch}
    scratch.clear()
    scratch.update(keep)


def _mm_row_gemm_only(weight: Any, x: torch.Tensor) -> torch.Tensor:
    """Row-split matmul without TP all_reduce; AR runs in COMM phase only."""
    return weight._mm.apply(x)


def _row_bias(weight: Any) -> torch.Tensor | None:
    return getattr(weight, "_row_split_bias", None)


@dataclass
class PhaseCompRunner:
    """One COMP phase advanced step-by-step on compute_stream during peer AR."""

    phase: int
    block_idx: int
    tenant: Any
    model: Any
    bind_tenant: Callable
    ensure_block: Callable
    capture_ti_snap: Callable
    step_idx: int = 0

    def finished(self) -> bool:
        return self.step_idx >= len(self._steps())

    def _steps(self) -> tuple[str, ...]:
        if self.phase == 1:
            return _SELF_P1 + ("sa_o_linear",)
        if self.phase == 3:
            return _CROSS_P3 + ("cx_o_linear",)
        if self.phase == 5:
            return _FFN_P5
        raise ValueError(f"no comp runner for phase {self.phase}")

    def advance(self, orch: PhasePumpOrch) -> bool:
        if self.finished():
            return False
        name = self._steps()[self.step_idx]
        nvtx.range_push(f"pump_p{self.phase}_{name}")
        _run_comp_substep(
            self.phase, name, self.model, self.tenant, self.block_idx,
            self.bind_tenant, self.ensure_block, self.capture_ti_snap, orch,
        )
        nvtx.range_pop()
        self.step_idx += 1
        return not self.finished()

    def pump_remaining(self, orch: PhasePumpOrch) -> int:
        """Run all remaining COMP steps in one bind (overlap pump fast path)."""
        if self.finished():
            return 0
        remaining = len(self._steps()) - self.step_idx
        start_name = self._steps()[self.step_idx]
        nvtx.range_push(f"pump_p{self.phase}_bulk")
        _run_comp_from_step(
            self.phase,
            start_name,
            self.model,
            self.tenant,
            self.block_idx,
            self.bind_tenant,
            self.ensure_block,
            self.capture_ti_snap,
            orch,
        )
        nvtx.range_pop()
        self.step_idx = len(self._steps())
        return remaining

    def pump_until(self, before_step: str, orch: PhasePumpOrch) -> int:
        """Run COMP steps in [step_idx, before_step) — stop before heavy attn/o tail."""
        steps = self._steps()
        end_idx = steps.index(before_step)
        if self.step_idx >= end_idx:
            return 0
        n = end_idx - self.step_idx
        start_name = steps[self.step_idx]
        nvtx.range_push(f"pump_p{self.phase}_pre_{before_step}")
        _run_comp_step_range(
            self.phase,
            start_name,
            before_step,
            self.model,
            self.tenant,
            self.block_idx,
            self.bind_tenant,
            self.ensure_block,
            self.capture_ti_snap,
            orch,
        )
        nvtx.range_pop()
        self.step_idx = end_idx
        return n

    def reset_and_prep_until(self, before_step: str, orch: PhasePumpOrch) -> None:
        """Run from phase start through *before_step* and leave step_idx at boundary."""
        self.step_idx = 0
        start_name = self._steps()[0]
        _run_comp_step_range(
            self.phase,
            start_name,
            before_step,
            self.model,
            self.tenant,
            self.block_idx,
            self.bind_tenant,
            self.ensure_block,
            self.capture_ti_snap,
            orch,
        )
        self.step_idx = self._steps().index(before_step)


@dataclass
class PhasePumpOrch:
    device: torch.device
    enabled: bool = False
    use_comm_p2p: bool = False
    use_chunked_ar: bool = False
    use_stream_ar_wait: bool = False
    use_stream_fence: bool = False
    use_row_ready_event: bool = False
    use_defer_attn_post_ar: bool = False
    use_delay_nccl_until_prep: bool = False
    use_row_ar_nccl_group: bool = False
    use_high_priority_comm_stream: bool = False
    use_cuda_graph_staggered_pump: bool = False
    use_staggered_pump: bool = False
    in_peer_comp: bool = False
    peer_runner: PhaseCompRunner | None = None
    comm_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream())
    compute_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream())
    pump_tail_graph: PumpTailCudaGraph = field(default_factory=PumpTailCudaGraph)
    ar_calls: int = 0
    ar_overlap_windows: int = 0
    peer_comp_steps: int = 0

    def stats(self) -> dict[str, int]:
        return {
            "all_reduce_calls": self.ar_calls,
            "all_reduce_overlap_windows": self.ar_overlap_windows,
            "peer_comp_steps": self.peer_comp_steps,
        }

    def set_peer(self, runner: PhaseCompRunner | None) -> None:
        self.peer_runner = runner

    def capture_peer_pump_tail_graph(self) -> dict[str, Any]:
        """Capture attn+o tail subgraph (after q/k prep) for staggered overlap replay."""
        runner = self.peer_runner
        if runner is None:
            return {"captured": False, "error": "no peer runner"}
        heavy = _DEFER_HEAVY_BEFORE.get(runner.phase)
        if heavy is None:
            return {"captured": False, "error": f"no heavy step for phase {runner.phase}"}

        def prep() -> None:
            runner.reset_and_prep_until(heavy, self)

        def run_tail() -> None:
            _run_comp_step_range(
                runner.phase,
                heavy,
                None,
                runner.model,
                runner.tenant,
                runner.block_idx,
                runner.bind_tenant,
                runner.ensure_block,
                runner.capture_ti_snap,
                self,
            )

        ok = self.pump_tail_graph.try_capture(
            phase=runner.phase,
            start_name=heavy,
            run_tail=run_tail,
            compute_stream=self.compute_stream,
            prep=prep,
        )
        runner.step_idx = 0
        return {
            "captured": ok,
            "phase": runner.phase,
            "start_name": heavy,
            "error": self.pump_tail_graph.capture_error,
        }

    def on_comm_stream(self) -> torch.cuda.StreamContext:
        return torch.cuda.stream(self.comm_stream)

    def on_compute_stream(self) -> torch.cuda.StreamContext:
        return torch.cuda.stream(self.compute_stream)

    def _nccl_preamble(self, tensor: torch.Tensor, ready: torch.cuda.Event | None) -> None:
        """Stream-scoped deps before NCCL: avoid default-stream device sync."""
        if self.use_row_ready_event and ready is not None:
            self.comm_stream.wait_event(ready)
        if self.use_row_ready_event:
            tensor.record_stream(self.comm_stream)

    def _wait_ar_work(self, work: Any) -> None:
        """Wait for async AR on comm_stream only (avoid dist.Work.wait device-wide sync)."""
        if work is None:
            return
        if self.use_stream_ar_wait:
            self.comm_stream.synchronize()
            return
        if hasattr(work, "wait"):
            work.wait()

    def _maybe_fence_streams(self) -> None:
        if not self.use_stream_fence:
            return
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        self.comm_stream.wait_event(ready)
        self.compute_stream.wait_event(ready)

    def _advance_peer(self) -> bool:
        if self.peer_runner is None or self.peer_runner.finished():
            return False
        self.in_peer_comp = True
        try:
            with self.on_compute_stream():
                self.peer_runner.advance(self)
            self.peer_comp_steps += 1
        finally:
            self.in_peer_comp = False
        return True

    def _pump_peer_remaining(self) -> bool:
        if self.peer_runner is None or self.peer_runner.finished():
            return False
        self.in_peer_comp = True
        try:
            with self.on_compute_stream():
                n = self.peer_runner.pump_remaining(self)
            self.peer_comp_steps += n
        finally:
            self.in_peer_comp = False
        return n > 0

    def _pump_peer_until(self, before_step: str) -> bool:
        if self.peer_runner is None or self.peer_runner.finished():
            return False
        self.in_peer_comp = True
        try:
            with self.on_compute_stream():
                n = self.peer_runner.pump_until(before_step, self)
            self.peer_comp_steps += n
        finally:
            self.in_peer_comp = False
        return n > 0

    def _defer_heavy_before(self) -> str | None:
        if self.peer_runner is None:
            return None
        if self.use_cuda_graph_staggered_pump or self.use_staggered_pump:
            return _DEFER_HEAVY_BEFORE.get(self.peer_runner.phase)
        if self.use_delay_nccl_until_prep:
            return _DEFER_HEAVY_BEFORE.get(self.peer_runner.phase)
        if self.use_defer_attn_post_ar:
            return _DEFER_HEAVY_BEFORE.get(self.peer_runner.phase)
        return None

    def _handle_ar_chunked(
        self,
        tensor: torch.Tensor,
        launch_chunk: Callable[[torch.Tensor, bool], Any],
    ) -> None:
        """Interleave COMM chunks with peer COMP micro-steps (finer overlap windows)."""
        flat = tensor.contiguous().view(-1)
        n_steps = max(1, len(self.peer_runner._steps()) if self.peer_runner else 1)
        chunk_size = (flat.numel() + n_steps - 1) // n_steps

        self.ar_calls += 1
        self.ar_overlap_windows += 1

        for step_i in range(n_steps):
            start = step_i * chunk_size
            end = min(start + chunk_size, flat.numel())
            work = None
            if start < end:
                chunk = flat[start:end]
                with torch.cuda.stream(self.comm_stream):
                    work = launch_chunk(chunk, True)
            if self.peer_runner is not None and not self.peer_runner.finished():
                self._advance_peer()
            if work is not None and hasattr(work, "wait"):
                self._wait_ar_work(work)

        while self.peer_runner is not None and not self.peer_runner.finished():
            self._advance_peer()

    def _pump_peer_during_ar(self) -> None:
        """Bulk-run peer COMP while NCCL AR is in flight (non-chunked path)."""
        if self.peer_runner is None or self.peer_runner.finished():
            return
        self._pump_peer_remaining()

    def _handle_ar(self, launch: Callable[..., Any]) -> None:
        self.ar_calls += 1
        if not self.enabled or self.peer_runner is None:
            with torch.cuda.stream(self.comm_stream):
                launch(async_op=False)
            return
        self.ar_overlap_windows += 1
        with no_sync_profiling(enabled=True):
            heavy_before = self._defer_heavy_before()
            work = None
            if heavy_before is not None and (
                self.use_cuda_graph_staggered_pump or self.use_staggered_pump
            ):
                nvtx.range_push("overlap_ar_peer_pump_pre")
                with torch.cuda.stream(self.compute_stream):
                    self._pump_peer_until(heavy_before)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_comm_launch")
                with torch.cuda.stream(self.comm_stream):
                    work = launch(async_op=True)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_peer_pump_post")
                self.in_peer_comp = True
                try:
                    if self.pump_tail_graph.captured:
                        self.pump_tail_graph.replay(self.compute_stream)
                        if self.peer_runner is not None:
                            self.peer_runner.step_idx = len(self.peer_runner._steps())
                            n_tail = len(self.peer_runner._steps()) - self.peer_runner._steps().index(heavy_before)
                            self.peer_comp_steps += n_tail
                    else:
                        with torch.cuda.stream(self.compute_stream):
                            self._pump_peer_remaining()
                finally:
                    self.in_peer_comp = False
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_wait")
                self._wait_ar_work(work)
                nvtx.range_pop()
            elif heavy_before is not None and self.use_delay_nccl_until_prep:
                nvtx.range_push("overlap_ar_peer_pump_pre")
                with torch.cuda.stream(self.compute_stream):
                    self._pump_peer_until(heavy_before)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_comm_launch")
                with torch.cuda.stream(self.comm_stream):
                    work = launch(async_op=True)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_peer_pump_post")
                with torch.cuda.stream(self.compute_stream):
                    self._pump_peer_remaining()
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_wait")
                self._wait_ar_work(work)
                nvtx.range_pop()
            elif heavy_before is not None and self.use_defer_attn_post_ar:
                nvtx.range_push("overlap_ar_comm_launch")
                with torch.cuda.stream(self.comm_stream):
                    work = launch(async_op=True)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_peer_pump_pre")
                with torch.cuda.stream(self.compute_stream):
                    self._pump_peer_until(heavy_before)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_wait_mid")
                self._wait_ar_work(work)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_peer_pump_post")
                with torch.cuda.stream(self.compute_stream):
                    self._pump_peer_remaining()
                nvtx.range_pop()
            else:
                nvtx.range_push("overlap_ar_comm_launch")
                with torch.cuda.stream(self.comm_stream):
                    work = launch(async_op=True)
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_peer_pump")
                with torch.cuda.stream(self.compute_stream):
                    self._pump_peer_during_ar()
                nvtx.range_pop()
                nvtx.range_push("overlap_ar_wait")
                self._wait_ar_work(work)
                nvtx.range_pop()

    def pure_row_ar(
        self,
        tensor: torch.Tensor,
        group: Any,
        bias: torch.Tensor | None = None,
        *,
        row_ready: torch.cuda.Event | None = None,
    ) -> None:
        use_chunked = self.use_chunked_ar and self.enabled and self.peer_runner is not None
        self._nccl_preamble(tensor, row_ready)

        if self.use_comm_p2p:
            from lightx2v.common.ops.tp_p2p_allreduce import get_tp_p2p_allreduce

            ex = get_tp_p2p_allreduce(group)

            def launch_full(async_op: bool):
                return ex.all_reduce(tensor, self.comm_stream, async_op=async_op)

            def launch_chunk(chunk: torch.Tensor, async_op: bool):
                return ex.all_reduce(chunk, self.comm_stream, async_op=async_op)

            if use_chunked:
                self._handle_ar_chunked(tensor, launch_chunk)
            else:
                self._handle_ar(launch_full)
        else:
            ar_group = resolve_row_ar_group(group, use_dedicated=self.use_row_ar_nccl_group)

            def launch_full(async_op: bool):
                return dist.all_reduce(
                    tensor, op=dist.ReduceOp.SUM, group=ar_group, async_op=async_op,
                )

            def launch_chunk(chunk: torch.Tensor, async_op: bool):
                return dist.all_reduce(
                    chunk, op=dist.ReduceOp.SUM, group=ar_group, async_op=async_op,
                )

            if use_chunked:
                self._handle_ar_chunked(tensor, launch_chunk)
            else:
                self._handle_ar(launch_full)
        if bias is not None:
            with self.on_comm_stream():
                tensor.add_(bias)

    def drain_peer(self) -> None:
        if self.peer_runner is None:
            return
        with torch.cuda.stream(self.compute_stream):
            while not self.peer_runner.finished():
                self._pump_peer_remaining()


_ORCH: PhasePumpOrch | None = None


def _get_orch(device: torch.device) -> PhasePumpOrch:
    global _ORCH
    if _ORCH is None:
        _ORCH = PhasePumpOrch(device)
    return _ORCH


def configure_phase_pipeline(
    *,
    use_comm_p2p: bool = False,
    use_chunked_ar: bool = False,
    use_stream_ar_wait: bool = False,
    use_stream_fence: bool = False,
    use_row_ready_event: bool = False,
    use_defer_attn_post_ar: bool = False,
    use_delay_nccl_until_prep: bool = False,
    use_row_ar_nccl_group: bool = False,
    use_high_priority_comm_stream: bool = False,
    use_cuda_graph_staggered_pump: bool = False,
    use_staggered_pump: bool = False,
) -> None:
    """Tune COMM path. Default row AR is NCCL; ``use_comm_p2p`` is experimental P2P substitute."""
    orch = _get_orch(torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0"))
    orch.use_comm_p2p = use_comm_p2p
    orch.use_chunked_ar = use_chunked_ar
    orch.use_stream_ar_wait = use_stream_ar_wait
    orch.use_stream_fence = use_stream_fence
    orch.use_row_ready_event = use_row_ready_event
    orch.use_defer_attn_post_ar = use_defer_attn_post_ar
    orch.use_delay_nccl_until_prep = use_delay_nccl_until_prep
    orch.use_row_ar_nccl_group = use_row_ar_nccl_group
    orch.use_high_priority_comm_stream = use_high_priority_comm_stream
    orch.use_cuda_graph_staggered_pump = use_cuda_graph_staggered_pump
    orch.use_staggered_pump = use_staggered_pump
    if use_high_priority_comm_stream:
        orch.comm_stream = torch.cuda.Stream(priority=-1)
    else:
        orch.comm_stream = torch.cuda.Stream()
    if not use_cuda_graph_staggered_pump:
        orch.pump_tail_graph.reset()


def _record_row_ready(st: BlockPipelineState, orch: PhasePumpOrch, *, scratch: str = "self") -> None:
    if not orch.use_row_ready_event:
        return
    bag = st.self_scratch if scratch == "self" else st.cross_scratch
    ev = torch.cuda.Event(blocking=False)
    ev.record(orch.compute_stream)
    bag["_row_ready_evt"] = ev


def _take_row_ready(st: BlockPipelineState, *, scratch: str = "self") -> torch.cuda.Event | None:
    bag = st.self_scratch if scratch == "self" else st.cross_scratch
    ev = bag.pop("_row_ready_evt", None)
    return ev


def _pipe_states(tenant: Any) -> dict[int, BlockPipelineState]:
    if not hasattr(tenant, "pipe_states") or tenant.pipe_states is None:
        tenant.pipe_states = {}
    return tenant.pipe_states


def _state(tenant: Any, block_idx: int) -> BlockPipelineState:
    ps = _pipe_states(tenant)
    if block_idx not in ps:
        ps[block_idx] = BlockPipelineState()
    return ps[block_idx]


def _step_index(runner: micro.TenantMicroRunner, name: str) -> int:
    for i, (_, n) in enumerate(runner.steps):
        if n == name:
            return i
    raise KeyError(name)


def _run_named_steps(
    runner: micro.TenantMicroRunner,
    names: tuple[str, ...],
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    capture_ti_snap: Callable,
) -> None:
    for name in names:
        runner.step_idx = _step_index(runner, name)
        micro._run_micro_step(runner, wan, ti, tenant, block, capture_ti_snap)


def _self_o_linear(
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    st: BlockPipelineState,
) -> None:
    pf = block.compute_phases[0]
    st.self_scratch["o_partial"] = _mm_row_gemm_only(pf.self_attn_o, st.self_scratch["attn_out"])


def _self_o_comm(
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    st: BlockPipelineState,
    capture_ti_snap: Callable,
    orch: PhasePumpOrch,
) -> None:
    from scripts.disagg.run_phase3_dual_overlap_bench import BlockMid  # noqa: PLC0415

    pf = block.compute_phases[0]
    y_out = st.self_scratch["o_partial"]
    orch.pure_row_ar(
        y_out, pf.self_attn_o.tp_group, _row_bias(pf.self_attn_o),
        row_ready=_take_row_ready(st, scratch="self"),
    )
    st.mid = BlockMid(mods=st.self_scratch["mods"], y_out=y_out)
    tenant.ti_snap = capture_ti_snap(ti)


def _cross_o_linear(
    wan: Any,
    ti: Any,
    block: Any,
    st: BlockPipelineState,
) -> None:
    pc = block.compute_phases[1]
    st.cross_scratch["o_partial"] = _mm_row_gemm_only(pc.cross_attn_o, st.cross_scratch["attn_out"])


def _cross_o_comm(
    wan: Any,
    ti: Any,
    block: Any,
    st: BlockPipelineState,
    orch: PhasePumpOrch,
) -> None:
    pc = block.compute_phases[1]
    o = st.cross_scratch["o_partial"]
    orch.pure_row_ar(
        o, pc.cross_attn_o.tp_group, _row_bias(pc.cross_attn_o),
        row_ready=_take_row_ready(st, scratch="cross"),
    )
    x = st.cross_scratch["x"].clone()
    x.add_(o)
    st.cross_scratch["x_ffn"] = x


def _ffn_norm2(
    wan: Any,
    ti: Any,
    block: Any,
    st: BlockPipelineState,
) -> None:
    pf = block.compute_phases[2]
    mid = st.mid
    _, _, _, c_shift_msa, c_scale_msa, _ = mid.mods
    x = st.cross_scratch["x_ffn"]
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
    st.cross_scratch["norm2_out"] = norm2_out


def _ffn0_gelu(block: Any, st: BlockPipelineState) -> None:
    pf = block.compute_phases[2]
    y = pf.ffn_0.apply(st.cross_scratch["norm2_out"])
    st.cross_scratch["gelu_out"] = torch.nn.functional.gelu(y, approximate="tanh")


def _ffn2_linear(block: Any, st: BlockPipelineState) -> None:
    pf = block.compute_phases[2]
    st.cross_scratch["y_partial"] = _mm_row_gemm_only(pf.ffn_2, st.cross_scratch["gelu_out"])


def _ffn6_comm(
    wan: Any,
    ti: Any,
    tenant: Any,
    block: Any,
    st: BlockPipelineState,
    capture_ti_snap: Callable,
    orch: PhasePumpOrch,
) -> None:
    pf = block.compute_phases[2]
    mid = st.mid
    _, _, _, _, _, c_gate_msa = mid.mods
    x = st.cross_scratch["x_ffn"]
    y = st.cross_scratch["y_partial"]
    orch.pure_row_ar(
        y, pf.ffn_2.tp_group, _row_bias(pf.ffn_2),
        row_ready=_take_row_ready(st, scratch="cross"),
    )
    if ti.sensitive_layer_dtype != ti.infer_dtype:
        x = x.to(ti.sensitive_layer_dtype) + y.to(ti.sensitive_layer_dtype) * c_gate_msa.squeeze()
    else:
        x = x.clone()
        x.add_(y * c_gate_msa.squeeze())
    tenant.x = x
    tenant.ti_snap = capture_ti_snap(ti)


def _run_comp_substep(
    phase: int,
    name: str,
    model: Any,
    tenant: Any,
    block_idx: int,
    bind_tenant: Callable,
    ensure_block: Callable,
    capture_ti_snap: Callable,
    orch: PhasePumpOrch,
) -> None:
    with orch.on_compute_stream():
        wan, ti = bind_tenant(model, tenant)
        block = ensure_block(tenant, wan, ti, block_idx)
        ti.block_idx = block_idx
        st = _state(tenant, block_idx)

        if phase == 1:
            if name == "sa_o_linear":
                _self_o_linear(wan, ti, tenant, block, st)
                _record_row_ready(st, orch, scratch="self")
                return
            runner = micro.TenantMicroRunner.for_self(block_idx)
            runner.scratch = st.self_scratch
            runner.step_idx = _step_index(runner, name)
            micro._run_micro_step(runner, wan, ti, tenant, block, capture_ti_snap)
            st.self_scratch = runner.scratch
            return

        if phase == 3:
            if name == "cx_o_linear":
                _cross_o_linear(wan, ti, block, st)
                _record_row_ready(st, orch, scratch="cross")
                return
            if st.mid is None:
                raise RuntimeError(f"phase 3 needs mid block={block_idx}")
            runner = micro.TenantMicroRunner.for_cross(block_idx, st.mid)
            runner.scratch = st.cross_scratch
            runner.step_idx = _step_index(runner, name)
            micro._run_micro_step(runner, wan, ti, tenant, block, capture_ti_snap)
            st.cross_scratch = runner.scratch
            return

        if phase == 5:
            if name == "ffn_norm2":
                _ffn_norm2(wan, ti, block, st)
            elif name == "ffn0_gelu":
                _ffn0_gelu(block, st)
            elif name == "ffn2_linear":
                _ffn2_linear(block, st)
                _record_row_ready(st, orch, scratch="cross")
            return

        raise ValueError(f"bad comp substep phase={phase} name={name}")


def _phase_step_list(phase: int) -> tuple[str, ...]:
    if phase == 1:
        return _SELF_P1 + ("sa_o_linear",)
    if phase == 3:
        return _CROSS_P3 + ("cx_o_linear",)
    if phase == 5:
        return _FFN_P5
    raise ValueError(f"bad comp phase={phase}")


def _run_comp_step_range(
    phase: int,
    start_name: str,
    end_before: str | None,
    model: Any,
    tenant: Any,
    block_idx: int,
    bind_tenant: Callable,
    ensure_block: Callable,
    capture_ti_snap: Callable,
    orch: PhasePumpOrch,
) -> None:
    """Run COMP phase steps [start_name, end_before); end_before=None means through end."""
    with orch.on_compute_stream():
        wan, ti = bind_tenant(model, tenant)
        block = ensure_block(tenant, wan, ti, block_idx)
        ti.block_idx = block_idx
        st = _state(tenant, block_idx)
        all_steps = _phase_step_list(phase)
        idx = all_steps.index(start_name)
        end_idx = len(all_steps) if end_before is None else all_steps.index(end_before)
        if idx >= end_idx:
            return
        run_steps = all_steps[idx:end_idx]

        if phase == 1:
            runner = micro.TenantMicroRunner.for_self(block_idx)
            runner.scratch = st.self_scratch
            micro_steps = tuple(s for s in run_steps if s != "sa_o_linear")
            if micro_steps:
                _run_named_steps(runner, micro_steps, wan, ti, tenant, block, capture_ti_snap)
            if "sa_o_linear" in run_steps:
                _self_o_linear(wan, ti, tenant, block, st)
                _record_row_ready(st, orch, scratch="self")
            st.self_scratch = runner.scratch
            return

        if phase == 3:
            if st.mid is None:
                raise RuntimeError(f"phase 3 needs mid block={block_idx}")
            runner = micro.TenantMicroRunner.for_cross(block_idx, st.mid)
            runner.scratch = st.cross_scratch
            micro_steps = tuple(s for s in run_steps if s != "cx_o_linear")
            if micro_steps:
                _run_named_steps(runner, micro_steps, wan, ti, tenant, block, capture_ti_snap)
            if "cx_o_linear" in run_steps:
                _cross_o_linear(wan, ti, block, st)
                _record_row_ready(st, orch, scratch="cross")
            st.cross_scratch = runner.scratch
            return

        if phase == 5:
            for name in run_steps:
                if name == "ffn_norm2":
                    _ffn_norm2(wan, ti, block, st)
                elif name == "ffn0_gelu":
                    _ffn0_gelu(block, st)
                elif name == "ffn2_linear":
                    _ffn2_linear(block, st)
                    _record_row_ready(st, orch, scratch="cross")
            return

        raise ValueError(f"bad comp range phase={phase} start={start_name}")


def _run_comp_from_step(
    phase: int,
    start_name: str,
    model: Any,
    tenant: Any,
    block_idx: int,
    bind_tenant: Callable,
    ensure_block: Callable,
    capture_ti_snap: Callable,
    orch: PhasePumpOrch,
) -> None:
    """Run COMP phase from start_name through end in a single bind."""
    _run_comp_step_range(
        phase, start_name, None, model, tenant, block_idx,
        bind_tenant, ensure_block, capture_ti_snap, orch,
    )


def _make_peer_runner(
    model: Any,
    tenant: Any,
    block_idx: int,
    comp_phase: int,
    bind_tenant: Callable,
    ensure_block: Callable,
    capture_ti_snap: Callable,
) -> PhaseCompRunner:
    return PhaseCompRunner(
        phase=comp_phase,
        block_idx=block_idx,
        tenant=tenant,
        model=model,
        bind_tenant=bind_tenant,
        ensure_block=ensure_block,
        capture_ti_snap=capture_ti_snap,
    )


def run_tenant_phase(
    model: Any,
    tenant: Any,
    block_idx: int,
    phase: int,
    bind_tenant: Callable,
    ensure_block: Callable,
    capture_ti_snap: Callable,
) -> None:
    device = torch.device(
        f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
    )
    orch = _get_orch(device)
    wan, ti = bind_tenant(model, tenant)
    block = ensure_block(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    st = _state(tenant, block_idx)

    stream_ctx = (
        orch.on_compute_stream() if phase in _COMP_PHASES
        else orch.on_comm_stream() if phase in _COMM_PHASES
        else nullcontext()
    )
    with stream_ctx:
        if phase == 1:
            runner = micro.TenantMicroRunner.for_self(block_idx)
            runner.scratch = st.self_scratch
            _run_named_steps(runner, _SELF_P1, wan, ti, tenant, block, capture_ti_snap)
            _self_o_linear(wan, ti, tenant, block, st)
            _record_row_ready(st, orch, scratch="self")
            st.self_scratch = runner.scratch
            return

        if phase == 2:
            _trim_self_for_comm(st.self_scratch)
            _self_o_comm(wan, ti, tenant, block, st, capture_ti_snap, orch)
            return

        if phase == 3:
            if st.mid is None:
                raise RuntimeError(f"phase 3 needs mid block={block_idx}")
            runner = micro.TenantMicroRunner.for_cross(block_idx, st.mid)
            runner.scratch = st.cross_scratch
            _run_named_steps(runner, _CROSS_P3, wan, ti, tenant, block, capture_ti_snap)
            _cross_o_linear(wan, ti, block, st)
            _record_row_ready(st, orch, scratch="cross")
            st.cross_scratch = runner.scratch
            return

        if phase == 4:
            _trim_cross_for_comm(st.cross_scratch)
            _cross_o_comm(wan, ti, block, st, orch)
            return

        if phase == 5:
            for step in _FFN_P5:
                _run_comp_substep(
                    5, step, model, tenant, block_idx,
                    bind_tenant, ensure_block, capture_ti_snap, orch,
                )
            return

        if phase == 6:
            _trim_cross_for_ffn_comm(st.cross_scratch)
            _ffn6_comm(wan, ti, tenant, block, st, capture_ti_snap, orch)
            return

    raise ValueError(f"invalid phase {phase}")


def phase_pipeline_main_blocks(
    model: Any,
    tenant_a: Any,
    tenant_b: Any,
    bind_tenant: Callable,
    ensure_block: Callable,
    preload_blocks: Callable,
    capture_ti_snap: Callable,
    *,
    overlap: bool,
    pair_timings: list[dict[str, Any]] | None = None,
    stats: dict[str, int] | None = None,
) -> None:
    import time

    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = _get_orch(device)
    orch.enabled = False
    orch.set_peer(None)
    orch.ar_calls = 0
    orch.ar_overlap_windows = 0
    orch.peer_comp_steps = 0

    wan_a, ti_a = bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = bind_tenant(model, tenant_b)
    preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    tenant_a.pipe_states = {}
    tenant_b.pipe_states = {}

    def _pair(
        label: str,
        comm_t: Any,
        comm_b: int,
        comm_p: int,
        comp_t: Any,
        comp_b: int,
        comp_p: int,
    ) -> None:
        if comm_p not in _COMM_PHASES:
            raise ValueError(f"comm phase must be 2/4/6, got {comm_p}")

        t0 = time.perf_counter()
        if overlap:
            with no_sync_profiling(enabled=True):
                orch.enabled = True
                orch.set_peer(_make_peer_runner(
                    model, comp_t, comp_b, comp_p, bind_tenant, ensure_block, capture_ti_snap,
                ))
                run_tenant_phase(
                    model, comm_t, comm_b, comm_p, bind_tenant, ensure_block, capture_ti_snap,
                )
                orch.drain_peer()
                orch.enabled = False
                orch.set_peer(None)
        else:
            run_tenant_phase(
                model, comm_t, comm_b, comm_p, bind_tenant, ensure_block, capture_ti_snap,
            )
            run_tenant_phase(
                model, comp_t, comp_b, comp_p, bind_tenant, ensure_block, capture_ti_snap,
            )
        wall = time.perf_counter() - t0
        if pair_timings is not None:
            pair_timings.append({
                "pair": label,
                "wall_s": wall,
                "overlap": overlap,
            })

    run_tenant_phase(model, tenant_a, 0, 1, bind_tenant, ensure_block, capture_ti_snap)

    with no_sync_profiling(enabled=overlap):
        if overlap:
            orch._maybe_fence_streams()
        for k in range(num_blocks):
            _pair(f"A2B1_L{k}", tenant_a, k, 2, tenant_b, k, 1)
            _pair(f"B2A3_L{k}", tenant_b, k, 2, tenant_a, k, 3)
            _pair(f"A4B3_L{k}", tenant_a, k, 4, tenant_b, k, 3)
            _pair(f"B4A5_L{k}", tenant_b, k, 4, tenant_a, k, 5)
            _pair(f"A6B5_L{k}", tenant_a, k, 6, tenant_b, k, 5)
            if k + 1 < num_blocks:
                _pair(f"B6A1_L{k}", tenant_b, k, 6, tenant_a, k + 1, 1)
            else:
                run_tenant_phase(model, tenant_b, k, 6, bind_tenant, ensure_block, capture_ti_snap)

            tenant_a.pipe_states.pop(k, None)
            tenant_b.pipe_states.pop(k, None)

    orch.comm_stream.synchronize()
    orch.compute_stream.synchronize()
    if stats is not None:
        stats.update(orch.stats())


def run_dual_phase_pipeline(
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
) -> tuple[float, dict[str, Any]]:
    pair_timings: list[dict[str, Any]] = []
    pipe_stats: dict[str, int] = {}

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
        with no_sync_profiling(enabled=overlap):
            for step_index in range(n_steps):
                tenant_a.scheduler.step_pre(step_index=step_index)
                tenant_b.scheduler.step_pre(step_index=step_index)
                pre_infer_tenant(model, tenant_a)
                pre_infer_tenant(model, tenant_b)
                phase_pipeline_main_blocks(
                    model, tenant_a, tenant_b,
                    bind_tenant, ensure_block, preload_blocks, capture_ti_snap,
                    overlap=overlap,
                    pair_timings=pair_timings if step_index == n_steps - 1 else None,
                    stats=pipe_stats if step_index == n_steps - 1 else None,
                )
                finish_step_tenant(model, tenant_a)
                finish_step_tenant(model, tenant_b)
                tenant_a.scheduler.step_post()
                tenant_b.scheduler.step_post()

    elapsed = time_fn(_body)
    return elapsed, {
        "pair_timings": pair_timings,
        "overlap": overlap,
        **pipe_stats,
    }
