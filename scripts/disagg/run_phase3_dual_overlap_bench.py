#!/usr/bin/env python3
"""Phase 3: dual-tenant comm/compute overlap on real Wan denoise.

**Comm overlap mode (primary)**: patch ``dist.all_to_all*`` and ``dist.all_gather`` —
async comm on ``comm_stream``, run other tenant's cross_ffn on ``compute_stream``
during each collective wait.

Schedule per layer k:
  B self_attn(k): each all_to_all || A cross_ffn(k)
  A self_attn(k+1): each all_to_all || B cross_ffn(k)

Compare vs same schedule with overlap disabled, and dual back-to-back.

Run: torchrun --standalone --nproc_per_node=4 scripts/disagg/run_phase3_dual_overlap_bench.py ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1_module():
    bench_path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1_bench", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {bench_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


@dataclass
class BlockMid:
    mods: tuple[Any, ...]
    y_out: torch.Tensor


@dataclass
class TiSnap:
    self_attn_cu_seqlens_qkv: Any
    cross_attn_cu_seqlens_q: Any
    cross_attn_cu_seqlens_kv: Any
    cross_attn_cu_seqlens_kv_img: Any


@dataclass
class TenantCtx:
    name: str
    scheduler: WanScheduler
    inputs: dict[str, Any]
    pre_infer_out: Any = None
    x: torch.Tensor | None = None
    ti_snap: TiSnap | None = None
    block_cache: dict[int, Any] = field(default_factory=dict)


class _OnceCompute:
    """Run a compute chunk once (first a2a window starts cross_ffn; rest overlap tail)."""

    def __init__(self, fn: Callable[[], None]):
        self._fn = fn
        self._done = False

    def __call__(self) -> None:
        if self._done:
            return
        self._fn()
        self._done = True


class _SegmentCrossFFN:
    """Run cross_ffn once across input a2a windows; safe to call from each window."""

    def __init__(self, fn: Callable[[], None]):
        self._fn = fn
        self._done = False

    def __call__(self) -> None:
        if self._done:
            return
        self._fn()
        self._done = True

    def reset(self) -> None:
        self._done = False


class A2AOrchestrator:
    """Patch dist all_to_all* / all_gather to overlap comm with a compute callback."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = False
        self.overlap_enabled = False
        self.overlap_strategy = "legacy"  # legacy | segment
        self.other_compute_cb: Callable[[], None] | None = None
        self.in_other_compute = False
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)
        self.other_self_stream = torch.cuda.Stream(device=device)
        self.a2a_calls = 0
        self.all_gather_calls = 0
        self.all_reduce_calls = 0
        self.overlap_windows = 0
        self.a2a_overlap_windows = 0
        self.all_gather_overlap_windows = 0
        self.all_reduce_overlap_windows = 0
        self.a2a_in_self_attn = 0
        self.pending_other_self_fn: Callable[[], Any] | None = None
        self.other_self_thread: threading.Thread | None = None
        self.other_self_result: Any = None
        self.other_self_errors: list[BaseException] = []
        self._orig: dict[str, Callable[..., Any]] = {}
        # In-flight self_attn tenant; overlap callback must not leave ti_snap mutated.
        self.ti_ref: Any | None = None

    def reset_self_attn_counters(self) -> None:
        self.a2a_in_self_attn = 0

    def _launch_other_self_thread(self) -> None:
        if self.pending_other_self_fn is None or self.other_self_thread is not None:
            return
        fn = self.pending_other_self_fn
        errors = self.other_self_errors
        orch = self

        def _worker() -> None:
            try:
                with torch.cuda.stream(orch.other_self_stream):
                    orch.other_self_result = fn()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        self.other_self_thread = threading.Thread(target=_worker, name="segment-other-self")
        self.other_self_thread.start()

    def _join_other_self_thread(self) -> None:
        if self.other_self_thread is None:
            return
        self.other_self_thread.join()
        self.other_self_stream.synchronize()
        self.other_self_thread = None
        if self.other_self_errors:
            raise self.other_self_errors[0]

    def install(self) -> None:
        if self._orig:
            return
        for name in ("all_to_all_single", "all_to_all", "all_gather", "all_reduce"):
            if hasattr(dist, name):
                self._orig[name] = getattr(dist, name)
                setattr(dist, name, getattr(self, f"_patched_{name}"))

    def restore(self) -> None:
        for name, fn in self._orig.items():
            setattr(dist, name, fn)
        self._orig.clear()

    def stats_snapshot(self) -> dict[str, int]:
        return {
            "a2a_calls": self.a2a_calls,
            "all_gather_calls": self.all_gather_calls,
            "all_reduce_calls": self.all_reduce_calls,
            "overlap_windows": self.overlap_windows,
            "a2a_overlap_windows": self.a2a_overlap_windows,
            "all_gather_overlap_windows": self.all_gather_overlap_windows,
            "all_reduce_overlap_windows": self.all_reduce_overlap_windows,
        }

    def _maybe_overlap(self, comm_launch: Callable[[], Any], *, kind: str) -> Any:
        if kind == "all_gather":
            self.all_gather_calls += 1
        elif kind == "all_reduce":
            self.all_reduce_calls += 1
        else:
            self.a2a_calls += 1

        if not self.overlap_enabled or self.other_compute_cb is None or self.in_other_compute:
            with torch.cuda.stream(self.comm_stream):
                return comm_launch(async_op=False)

        if self.overlap_strategy == "segment":
            return self._maybe_overlap_segment(comm_launch, kind=kind)

        self.overlap_windows += 1
        if kind == "all_gather":
            self.all_gather_overlap_windows += 1
        elif kind == "all_reduce":
            self.all_reduce_overlap_windows += 1
        else:
            self.a2a_overlap_windows += 1

        with torch.cuda.stream(self.comm_stream):
            work = comm_launch(async_op=True)
        saved_snap: TiSnap | None = None
        if self.ti_ref is not None:
            saved_snap = _capture_ti_snap(self.ti_ref)
        self.in_other_compute = True
        try:
            with torch.cuda.stream(self.compute_stream):
                self.other_compute_cb()
        finally:
            self.in_other_compute = False
            if saved_snap is not None and self.ti_ref is not None:
                _apply_ti_snap(self.ti_ref, saved_snap)
        if work is not None and hasattr(work, "wait"):
            work.wait()
        return work

    def _maybe_overlap_segment(self, comm_launch: Callable[[], Any], *, kind: str) -> Any:
        """Profile-guided pairing: input a2a||cross_ffn; middle||other-self; output a2a||cross_ffn."""
        self.overlap_windows += 1
        if kind == "all_gather":
            self.all_gather_overlap_windows += 1
            with torch.cuda.stream(self.comm_stream):
                return comm_launch(async_op=False)

        self.a2a_overlap_windows += 1
        a2a_idx = self.a2a_in_self_attn
        self.a2a_in_self_attn += 1

        with torch.cuda.stream(self.comm_stream):
            work = comm_launch(async_op=True)

        saved_snap: TiSnap | None = None
        if self.ti_ref is not None:
            saved_snap = _capture_ti_snap(self.ti_ref)

        if a2a_idx < 3:
            # Input comm burst (~25ms): overlap other tenant cross_ffn (~32ms).
            self.in_other_compute = True
            try:
                with torch.cuda.stream(self.compute_stream):
                    self.other_compute_cb()
            finally:
                self.in_other_compute = False
                if saved_snap is not None and self.ti_ref is not None:
                    _apply_ti_snap(self.ti_ref, saved_snap)
            if work is not None and hasattr(work, "wait"):
                work.wait()
            if a2a_idx == 2:
                self.compute_stream.synchronize()
                self._launch_other_self_thread()
            return work

        if a2a_idx == 3:
            # Output a2a (~8ms): overlap other tenant cross_ffn tail if not done yet.
            self.in_other_compute = True
            try:
                with torch.cuda.stream(self.compute_stream):
                    self.other_compute_cb()
            finally:
                self.in_other_compute = False
                if saved_snap is not None and self.ti_ref is not None:
                    _apply_ti_snap(self.ti_ref, saved_snap)
            if work is not None and hasattr(work, "wait"):
                work.wait()
            return work

        if work is not None and hasattr(work, "wait"):
            work.wait()
        return work

    def _patched_all_to_all_single(self, output, input, group=None, async_op=False, **kwargs):
        if not self.enabled:
            return self._orig["all_to_all_single"](output, input, group=group, async_op=async_op, **kwargs)

        def launch(async_op: bool):
            return self._orig["all_to_all_single"](
                output, input, group=group, async_op=async_op, **kwargs,
            )

        return self._maybe_overlap(launch, kind="a2a")

    def _patched_all_to_all(self, output_tensor_list, input_tensor_list, group=None, async_op=False, **kwargs):
        if not self.enabled:
            return self._orig["all_to_all"](
                output_tensor_list, input_tensor_list, group=group, async_op=async_op, **kwargs,
            )

        def launch(async_op: bool):
            return self._orig["all_to_all"](
                output_tensor_list, input_tensor_list, group=group, async_op=async_op, **kwargs,
            )

        return self._maybe_overlap(launch, kind="a2a")

    def _patched_all_gather(self, tensor_list, tensor, group=None, async_op=False, **kwargs):
        if not self.enabled:
            return self._orig["all_gather"](tensor_list, tensor, group=group, async_op=async_op, **kwargs)

        def launch(async_op: bool):
            return self._orig["all_gather"](tensor_list, tensor, group=group, async_op=async_op, **kwargs)

        return self._maybe_overlap(launch, kind="all_gather")

    def _patched_all_reduce(self, tensor, op=None, group=None, async_op=False, **kwargs):
        if not self.enabled:
            return self._orig["all_reduce"](tensor, op=op, group=group, async_op=async_op, **kwargs)

        def launch(async_op: bool):
            return self._orig["all_reduce"](tensor, op=op, group=group, async_op=async_op, **kwargs)

        return self._maybe_overlap(launch, kind="all_reduce")


class _OverlapDistPatch:
    def __init__(self, comm_stream: torch.cuda.Stream):
        self.comm_stream = comm_stream
        self._orig: dict[str, Callable[..., Any]] = {}

    def install(self) -> None:
        for name in ("all_to_all", "all_gather", "all_reduce", "broadcast", "reduce_scatter"):
            if hasattr(dist, name):
                self._orig[name] = getattr(dist, name)
                setattr(dist, name, self._wrap(getattr(dist, name)))

    def restore(self) -> None:
        for name, fn in self._orig.items():
            setattr(dist, name, fn)
        self._orig.clear()

    def _wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        comm_stream = self.comm_stream

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with torch.cuda.stream(comm_stream):
                return fn(*args, **kwargs)

        return wrapped


@contextmanager
def _overlap_dist_patch(comm_stream: torch.cuda.Stream):
    patch = _OverlapDistPatch(comm_stream)
    patch.install()
    try:
        yield
    finally:
        patch.restore()


def _capture_ti_snap(ti: Any) -> TiSnap:
    return TiSnap(
        self_attn_cu_seqlens_qkv=ti.self_attn_cu_seqlens_qkv,
        cross_attn_cu_seqlens_q=ti.cross_attn_cu_seqlens_q,
        cross_attn_cu_seqlens_kv=ti.cross_attn_cu_seqlens_kv,
        cross_attn_cu_seqlens_kv_img=ti.cross_attn_cu_seqlens_kv_img,
    )


def _apply_ti_snap(ti: Any, snap: TiSnap | None) -> None:
    if snap is None:
        ti.reset_infer_states()
        return
    ti.self_attn_cu_seqlens_qkv = snap.self_attn_cu_seqlens_qkv
    ti.cross_attn_cu_seqlens_q = snap.cross_attn_cu_seqlens_q
    ti.cross_attn_cu_seqlens_kv = snap.cross_attn_cu_seqlens_kv
    ti.cross_attn_cu_seqlens_kv_img = snap.cross_attn_cu_seqlens_kv_img


_MODEL_INDEX_CACHE: dict[tuple[int, int], int] = {}


def _active_wan(model: Any, scheduler: WanScheduler) -> Any:
    if hasattr(model, "get_current_model_index") and hasattr(model, "model"):
        model.set_scheduler(scheduler)
        key = (id(scheduler), int(scheduler.step_index))
        idx = _MODEL_INDEX_CACHE.get(key)
        if idx is None:
            model.get_current_model_index()
            idx = int(model.cur_model_index)
            _MODEL_INDEX_CACHE[key] = idx
        else:
            model.cur_model_index = idx
        return model.model[idx]
    model.set_scheduler(scheduler)
    return model


def _bind_tenant(model: Any, tenant: TenantCtx) -> tuple[Any, Any]:
    wan = _active_wan(model, tenant.scheduler)
    ti = wan.transformer_infer
    ti.scheduler = tenant.scheduler
    ti.cos_sin = tenant.pre_infer_out.cos_sin
    _apply_ti_snap(ti, tenant.ti_snap)
    return wan, ti


def _ensure_block(tenant: TenantCtx, wan: Any, ti: Any, block_idx: int) -> Any:
    if block_idx in tenant.block_cache:
        return tenant.block_cache[block_idx]
    weights = wan.transformer_weights
    if hasattr(ti, "offload_manager"):
        om = ti.offload_manager
        blocks = weights.blocks
        if om.need_init_first_buffer:
            om.init_first_buffer(blocks)
        om.cuda_buffers[0].load_state_dict(blocks[block_idx].state_dict(), block_idx)
        om.cuda_load_stream.synchronize()
        block = om.cuda_buffers[0]
    else:
        block = weights.blocks[block_idx]
    tenant.block_cache[block_idx] = block
    return block


def _preload_blocks(tenant: TenantCtx, wan: Any, ti: Any, num_blocks: int) -> None:
    tenant.block_cache.clear()
    for i in range(num_blocks):
        _ensure_block(tenant, wan, ti, i)


def _run_self_attn_block(wan: Any, ti: Any, tenant: TenantCtx, block_idx: int) -> BlockMid:
    block = _ensure_block(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    x = tenant.x
    pre = tenant.pre_infer_out
    if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
        x = block.compute_phases[0].before_proj.apply(x) + pre.x
    mods = ti.pre_process(block.compute_phases[0].modulation, pre.embed0)
    y_out = ti.infer_self_attn(block.compute_phases[0], x, mods[0], mods[1])
    tenant.x = x
    tenant.ti_snap = _capture_ti_snap(ti)
    return BlockMid(mods=mods, y_out=y_out)


def _run_cross_ffn_block(wan: Any, ti: Any, tenant: TenantCtx, block_idx: int, mid: BlockMid) -> None:
    block = _ensure_block(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mid.mods
    x, attn_out = ti.infer_cross_attn(
        block.compute_phases[1], tenant.x, tenant.pre_infer_out.context, mid.y_out, gate_msa,
    )
    y = ti.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
    tenant.x = ti.post_process(x, y, c_gate_msa, tenant.pre_infer_out)
    tenant.ti_snap = _capture_ti_snap(ti)


def _run_parallel_overlap(
    compute_fn: Callable[[], None],
    comm_fn: Callable[[], None],
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
) -> float:
    """Launch compute and comm worker threads on all ranks; return wall seconds."""
    if dist.is_initialized():
        dist.barrier()

    errors: list[BaseException] = []

    def _compute_worker() -> None:
        try:
            with torch.cuda.stream(compute_stream):
                compute_fn()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _comm_worker() -> None:
        try:
            with _overlap_dist_patch(comm_stream):
                with torch.cuda.stream(comm_stream):
                    comm_fn()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    _sync()
    t0 = time.perf_counter()
    tc = threading.Thread(target=_compute_worker, name="overlap-compute")
    tm = threading.Thread(target=_comm_worker, name="overlap-comm")
    tc.start()
    tm.start()
    tc.join()
    tm.join()
    compute_stream.synchronize()
    comm_stream.synchronize()
    wall = time.perf_counter() - t0
    if errors:
        raise errors[0]
    return wall


def _run_serial_phases(compute_fn: Callable[[], None], comm_fn: Callable[[], None]) -> float:
    _sync()
    t0 = time.perf_counter()
    compute_fn()
    comm_fn()
    _sync()
    return time.perf_counter() - t0


def _layer_pipeline_main_blocks(
    model: Any,
    tenant_a: TenantCtx,
    tenant_b: TenantCtx,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    *,
    overlap: bool,
    pair_timings: list[dict[str, float]] | None = None,
) -> None:
    wan_a, ti_a = _bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    _preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = _bind_tenant(model, tenant_b)
    _preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    mid_a = _run_self_attn_block(wan_a, ti_a, tenant_a, 0)

    for k in range(num_blocks):
        def a_cross() -> None:
            w, ti = _bind_tenant(model, tenant_a)
            _run_cross_ffn_block(w, ti, tenant_a, k, mid_a)

        mid_b_holder: list[BlockMid | None] = [None]

        def b_self() -> None:
            w, ti = _bind_tenant(model, tenant_b)
            mid_b_holder[0] = _run_self_attn_block(w, ti, tenant_b, k)

        if overlap:
            wall = _run_parallel_overlap(a_cross, b_self, compute_stream, comm_stream)
            if pair_timings is not None:
                pair_timings.append({"pair": f"crossA_selfB_L{k}", "parallel_wall_s": wall})
        else:
            wall = _run_serial_phases(a_cross, b_self)
            if pair_timings is not None:
                pair_timings.append({"pair": f"crossA_selfB_L{k}", "serial_wall_s": wall})

        mid_b = mid_b_holder[0]
        assert mid_b is not None

        if k + 1 < num_blocks:
            def a_self() -> None:
                nonlocal mid_a
                w, ti = _bind_tenant(model, tenant_a)
                mid_a = _run_self_attn_block(w, ti, tenant_a, k + 1)

            def b_cross() -> None:
                w, ti = _bind_tenant(model, tenant_b)
                _run_cross_ffn_block(w, ti, tenant_b, k, mid_b)

            if overlap:
                wall = _run_parallel_overlap(b_cross, a_self, compute_stream, comm_stream)
                if pair_timings is not None:
                    pair_timings.append({"pair": f"crossB_selfA_L{k}", "parallel_wall_s": wall})
            else:
                wall = _run_serial_phases(b_cross, a_self)
                if pair_timings is not None:
                    pair_timings.append({"pair": f"crossB_selfA_L{k}", "serial_wall_s": wall})
        else:
            w, ti = _bind_tenant(model, tenant_b)
            _run_cross_ffn_block(w, ti, tenant_b, k, mid_b)


def _a2a_overlap_main_blocks(
    model: Any,
    tenant_a: TenantCtx,
    tenant_b: TenantCtx,
    orch: A2AOrchestrator,
    *,
    overlap: bool,
    overlap_strategy: str = "legacy",
) -> None:
    """Per-layer schedule with optional all_to_all-window overlap via *orch*."""
    wan_a, ti_a = _bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    _preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = _bind_tenant(model, tenant_b)
    _preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    orch.enabled = True
    orch.overlap_enabled = False
    orch.overlap_strategy = overlap_strategy if overlap else "legacy"
    orch.other_compute_cb = None
    orch.pending_other_self_fn = None

    mid_a = _run_self_attn_block(wan_a, ti_a, tenant_a, 0)

    for k in range(num_blocks):
        if overlap:
            orch.overlap_enabled = True
            if overlap_strategy == "segment":
                orch.other_compute_cb = _SegmentCrossFFN(
                    lambda k=k, m=mid_a: _run_cross_ffn_block(
                        *_bind_tenant(model, tenant_a), tenant_a, k, m,
                    ),
                )
                if k + 1 < num_blocks:
                    def _other_a_self(k=k) -> BlockMid:
                        w, ti = _bind_tenant(model, tenant_a)
                        return _run_self_attn_block(w, ti, tenant_a, k + 1)

                    orch.pending_other_self_fn = _other_a_self
            else:
                orch.pending_other_self_fn = None
                orch.other_compute_cb = _OnceCompute(
                    lambda k=k, m=mid_a: _run_cross_ffn_block(
                        *_bind_tenant(model, tenant_a), tenant_a, k, m,
                    ),
                )
        else:
            orch.overlap_enabled = False
            orch.other_compute_cb = None
            orch.pending_other_self_fn = None

        wan_b, ti_b = _bind_tenant(model, tenant_b)
        if overlap:
            orch.ti_ref = ti_b
            orch.reset_self_attn_counters()
        try:
            mid_b = _run_self_attn_block(wan_b, ti_b, tenant_b, k)
        finally:
            if overlap:
                orch.ti_ref = None
        if overlap and overlap_strategy == "segment":
            orch._join_other_self_thread()
            if orch.other_self_result is not None:
                mid_a = orch.other_self_result
                orch.other_self_result = None
                orch.pending_other_self_fn = None

        if not overlap:
            wan_a, ti_a = _bind_tenant(model, tenant_a)
            _run_cross_ffn_block(wan_a, ti_a, tenant_a, k, mid_a)

        if k + 1 < num_blocks:
            if overlap and overlap_strategy == "segment" and orch.pending_other_self_fn is None:
                # A self(k+1) finished in background during B self(k).
                wan_b, ti_b = _bind_tenant(model, tenant_b)
                _run_cross_ffn_block(wan_b, ti_b, tenant_b, k, mid_b)
            elif overlap:
                orch.overlap_enabled = True
                if overlap_strategy == "segment":
                    orch.other_compute_cb = _SegmentCrossFFN(
                        lambda k=k, mb=mid_b: _run_cross_ffn_block(
                            *_bind_tenant(model, tenant_b), tenant_b, k, mb,
                        ),
                    )

                    def _other_b_self(k=k) -> BlockMid:
                        w, ti = _bind_tenant(model, tenant_b)
                        return _run_self_attn_block(w, ti, tenant_b, k + 1)

                    orch.pending_other_self_fn = _other_b_self
                else:
                    orch.pending_other_self_fn = None
                    orch.other_compute_cb = _OnceCompute(
                        lambda k=k, mb=mid_b: _run_cross_ffn_block(
                            *_bind_tenant(model, tenant_b), tenant_b, k, mb,
                        ),
                    )

                wan_a, ti_a = _bind_tenant(model, tenant_a)
                if overlap:
                    orch.ti_ref = ti_a
                    orch.reset_self_attn_counters()
                try:
                    mid_a = _run_self_attn_block(wan_a, ti_a, tenant_a, k + 1)
                finally:
                    if overlap:
                        orch.ti_ref = None
                if overlap_strategy == "segment":
                    orch._join_other_self_thread()
                    if orch.other_self_result is not None:
                        mid_b = orch.other_self_result
                        orch.other_self_result = None
                        orch.pending_other_self_fn = None
            else:
                orch.overlap_enabled = False
                orch.other_compute_cb = None
                orch.pending_other_self_fn = None

                wan_a, ti_a = _bind_tenant(model, tenant_a)
                mid_a = _run_self_attn_block(wan_a, ti_a, tenant_a, k + 1)

            if not overlap:
                wan_b, ti_b = _bind_tenant(model, tenant_b)
                _run_cross_ffn_block(wan_b, ti_b, tenant_b, k, mid_b)
        else:
            orch.overlap_enabled = False
            orch.other_compute_cb = None
            orch.pending_other_self_fn = None
            wan_b, ti_b = _bind_tenant(model, tenant_b)
            _run_cross_ffn_block(wan_b, ti_b, tenant_b, k, mid_b)

    orch.overlap_enabled = False
    orch.other_compute_cb = None
    orch.pending_other_self_fn = None
    orch.enabled = False
    orch.compute_stream.synchronize()
    orch.comm_stream.synchronize()
    orch.other_self_stream.synchronize()
    _sync()


def _single_tenant_blocks_decomposed(
    model: Any,
    tenant: TenantCtx,
    orch: A2AOrchestrator | None = None,
) -> None:
    """One tenant layer split; optional a2a patch (same as dual path, no cross-tenant overlap)."""
    wan, ti = _bind_tenant(model, tenant)
    num_blocks = len(wan.transformer_weights.blocks)
    _preload_blocks(tenant, wan, ti, num_blocks)

    if orch is not None:
        orch.enabled = True
        orch.overlap_enabled = False
        orch.other_compute_cb = None

    mid = _run_self_attn_block(wan, ti, tenant, 0)
    for k in range(num_blocks):
        wan, ti = _bind_tenant(model, tenant)
        _run_cross_ffn_block(wan, ti, tenant, k, mid)
        if k + 1 < num_blocks:
            wan, ti = _bind_tenant(model, tenant)
            mid = _run_self_attn_block(wan, ti, tenant, k + 1)

    if orch is not None:
        orch.enabled = False
        orch.compute_stream.synchronize()
        orch.comm_stream.synchronize()
    _sync()


def _run_single_decomposed_once(
    model: Any,
    tenant: TenantCtx,
    payload: dict[str, Any],
    *,
    patch_a2a: bool,
    steps: int,
    orch: A2AOrchestrator | None = None,
    install_orch: bool = False,
) -> None:
    tenant.scheduler.prepare(
        seed=int(payload["seed"]),
        latent_shape=payload["latent_shape"],
        image_encoder_output=payload["image_encoder_output"],
    )
    tenant.inputs = payload["inputs"]
    local_orch = orch
    owns_orch = False
    if patch_a2a and local_orch is None:
        device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
        local_orch = A2AOrchestrator(device)
        local_orch.install()
        owns_orch = True
    try:
        for step_index in range(steps):
            tenant.scheduler.step_pre(step_index=step_index)
            _pre_infer_tenant(model, tenant)
            _single_tenant_blocks_decomposed(
                model, tenant, local_orch if patch_a2a else None,
            )
            _finish_step_tenant(model, tenant)
            tenant.scheduler.step_post()
    finally:
        if owns_orch and local_orch is not None:
            local_orch.restore()


def _summarize_samples(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"n": 0, "mean_s": None, "min_s": None, "max_s": None, "stdev_s": None}
    out: dict[str, float | int | None] = {
        "n": len(samples),
        "mean_s": statistics.mean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
    }
    out["stdev_s"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return out


def _bench_single_path_alignment(
    model: Any,
    scheduler: WanScheduler,
    tenant: TenantCtx,
    payload: dict[str, Any],
    *,
    measure_steps: int,
    align_iters: int,
) -> dict[str, Any]:
    """Interleaved timing: model.infer vs decomposed plain vs decomposed+a2a patch."""
    paths = ("model_infer", "decomposed_plain", "decomposed_patch")
    rotate = [
        paths,
        (paths[1], paths[2], paths[0]),
        (paths[2], paths[0], paths[1]),
    ]
    samples: dict[str, list[float]] = {p: [] for p in paths}

    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = A2AOrchestrator(device)
    orch.install()
    try:
        for round_idx in range(align_iters):
            for path_name in rotate[round_idx % len(rotate)]:
                if path_name == "model_infer":
                    samples[path_name].append(
                        _time_fn(lambda: _run_denoise_serial(model, scheduler, payload)),
                    )
                elif path_name == "decomposed_plain":
                    samples[path_name].append(
                        _time_fn(
                            lambda: _run_single_decomposed_once(
                                model, tenant, payload, patch_a2a=False, steps=measure_steps,
                            ),
                        ),
                    )
                else:
                    samples[path_name].append(
                        _time_fn(
                            lambda: _run_single_decomposed_once(
                                model, tenant, payload, patch_a2a=True, steps=measure_steps, orch=orch,
                            ),
                        ),
                    )
    finally:
        orch.restore()

    stats = {name: _summarize_samples(samples[name]) for name in paths}
    return {
        "align_iters": align_iters,
        "measure_steps": measure_steps,
        "rotation": "3-way round-robin per round",
        "samples_s": samples,
        "stats": stats,
    }


def _ensure_wan_gpu(wan: Any) -> None:
    if wan.config.get("cpu_offload", False):
        wan.pre_weight.to_cuda()
        wan.transformer_weights.non_block_weights_to_cuda()


def _pre_infer_tenant(model: Any, tenant: TenantCtx) -> None:
    wan = _active_wan(model, tenant.scheduler)
    _ensure_wan_gpu(wan)
    pre = wan.pre_infer.infer(wan.pre_weight, tenant.inputs)
    if wan.config["seq_parallel"]:
        pre = wan._seq_parallel_pre_process(pre)
    tenant.pre_infer_out = pre
    tenant.x = pre.x
    tenant.ti_snap = None
    tenant.block_cache.clear()


def _finish_step_tenant(model: Any, tenant: TenantCtx) -> None:
    wan, ti = _bind_tenant(model, tenant)
    x = ti.infer_non_blocks(wan.transformer_weights, tenant.x, tenant.pre_infer_out.embed)
    if wan.config["seq_parallel"]:
        x = wan._seq_parallel_post_process(x)
    noise_pred = wan.post_infer.infer(x, tenant.pre_infer_out)[0]
    tenant.scheduler.noise_pred_cond = noise_pred
    tenant.scheduler.noise_pred_uncond = None
    tenant.scheduler.noise_pred_guided = noise_pred
    tenant.scheduler.noise_pred = noise_pred


def _one_step_blocks(
    model: Any,
    tenant_a: TenantCtx,
    tenant_b: TenantCtx,
    step_index: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    *,
    overlap: bool,
    pair_timings: list[dict[str, float]] | None = None,
) -> None:
    tenant_a.scheduler.step_pre(step_index=step_index)
    tenant_b.scheduler.step_pre(step_index=step_index)
    _pre_infer_tenant(model, tenant_a)
    _pre_infer_tenant(model, tenant_b)
    _layer_pipeline_main_blocks(
        model, tenant_a, tenant_b, compute_stream, comm_stream,
        overlap=overlap, pair_timings=pair_timings,
    )
    _finish_step_tenant(model, tenant_a)
    _finish_step_tenant(model, tenant_b)
    tenant_a.scheduler.step_post()
    tenant_b.scheduler.step_post()


def _run_denoise_serial(model: Any, scheduler: WanScheduler, payload: dict[str, Any]) -> None:
    scheduler.prepare(
        seed=int(payload["seed"]),
        latent_shape=payload["latent_shape"],
        image_encoder_output=payload["image_encoder_output"],
    )
    for step_index in range(scheduler.infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.set_scheduler(scheduler)
        model.infer(payload["inputs"])
        scheduler.step_post()
    _sync()


def _time_fn(fn: Callable[[], None]) -> float:
    _sync()
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _run_dual_layer_pipeline(
    model: Any,
    tenant_a: TenantCtx,
    tenant_b: TenantCtx,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    overlap: bool,
    steps: int | None = None,
) -> tuple[float, list[dict[str, float]]]:
    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device)

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
    pair_timings: list[dict[str, float]] = []

    def _body() -> None:
        for step_index in range(n_steps):
            _one_step_blocks(
                model, tenant_a, tenant_b, step_index,
                compute_stream, comm_stream,
                overlap=overlap,
                pair_timings=pair_timings if step_index == 0 else None,
            )

    return _time_fn(_body), pair_timings


def _run_single_a2a_pipeline(
    model: Any,
    tenant: TenantCtx,
    payload: dict[str, Any],
    *,
    steps: int | None = None,
) -> tuple[float, dict[str, Any]]:
    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = A2AOrchestrator(device)
    orch.install()
    n_steps = steps if steps is not None else tenant.scheduler.infer_steps

    def _body() -> None:
        _run_single_decomposed_once(
            model, tenant, payload, patch_a2a=True, steps=n_steps, orch=orch,
        )

    try:
        return _time_fn(_body), orch.stats_snapshot()
    finally:
        orch.restore()


def _run_dual_a2a_pipeline(
    model: Any,
    tenant_a: TenantCtx,
    tenant_b: TenantCtx,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    overlap: bool,
    overlap_strategy: str = "legacy",
    steps: int | None = None,
) -> tuple[float, dict[str, Any]]:
    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = A2AOrchestrator(device)
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
            _pre_infer_tenant(model, tenant_a)
            _pre_infer_tenant(model, tenant_b)
            _a2a_overlap_main_blocks(
                model, tenant_a, tenant_b, orch,
                overlap=overlap, overlap_strategy=overlap_strategy,
            )
            _finish_step_tenant(model, tenant_a)
            _finish_step_tenant(model, tenant_b)
            tenant_a.scheduler.step_post()
            tenant_b.scheduler.step_post()

    try:
        wall = _time_fn(_body)
        stats = orch.stats_snapshot()
        stats["overlap_strategy"] = overlap_strategy if overlap else "serial"
        return wall, stats
    finally:
        orch.restore()


def _summarize_pair_timings(pair_timings: list[dict[str, float]]) -> dict[str, Any]:
    parallel = [p["parallel_wall_s"] for p in pair_timings if "parallel_wall_s" in p]
    serial = [p["serial_wall_s"] for p in pair_timings if "serial_wall_s" in p]
    out: dict[str, Any] = {"num_pairs": len(parallel) or len(serial)}
    if parallel:
        out["parallel_mean_s"] = statistics.mean(parallel)
        out["parallel_sum_s"] = sum(parallel)
    if serial:
        out["serial_mean_s"] = statistics.mean(serial)
        out["serial_sum_s"] = sum(serial)
    if parallel and serial and len(parallel) == len(serial):
        saved = [s - p for s, p in zip(serial, parallel)]
        out["per_pair_saved_mean_s"] = statistics.mean(saved)
        out["per_pair_saved_sum_s"] = sum(saved)
        out["overlap_fraction_of_serial"] = sum(saved) / sum(serial) if sum(serial) > 0 else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--config_json", default="")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v", choices=("i2v", "t2v"))
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument(
        "--inputs_cache_b",
        default="",
        help="Second encoder cache for tenant B (mixed-resolution dual overlap). "
        "If empty, tenant B reuses tenant A latent_shape/inputs.",
    )
    parser.add_argument("--resolution_a", default="", help="Label for tenant A resolution (e.g. 512x512)")
    parser.add_argument("--resolution_b", default="", help="Label for tenant B resolution (e.g. 1024x1024)")
    parser.add_argument("--phase3_json", default="/root/zht/LightX2V/save_results/optimization_study/p3_sp_seqp4.json")
    parser.add_argument(
        "--negative_prompt",
        default="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )
    parser.add_argument("--seed_a", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=0, help="0 = all infer_steps")
    parser.add_argument(
        "--align_iters",
        type=int,
        default=3,
        help="Interleaved rounds for model.infer vs decomposed plain vs decomposed+a2a patch",
    )
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    p1 = _load_phase1_module()
    study_dir = Path("/root/zht/LightX2V/save_results/optimization_study")
    cfg_path = Path(args.config_json) if args.config_json else study_dir / f"baseline_seqp{args.seq_p_size}.json"

    ns = argparse.Namespace(
        config_json=str(cfg_path),
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
            if args.task == "t2v"
            else "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
        ),
        seed=args.seed_a,
        seq_p_size=args.seq_p_size,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt=args.negative_prompt,
    )
    config = p1._load_config(ns)
    seed_all(args.seed_a)
    p1._init_distributed(config)

    payload_a = p1._prepare_inputs_cache(
        config=config, cache_path=Path(args.inputs_cache),
        prompt=ns.prompt, image_path=ns.image_path, seed=args.seed_a, force=False,
        task=args.task, negative_prompt=args.negative_prompt,
    )
    payload_a = p1._prepare_payload_on_device(payload_a)
    if args.inputs_cache_b:
        payload_b = p1._prepare_inputs_cache(
            config=config, cache_path=Path(args.inputs_cache_b),
            prompt=ns.prompt, image_path=ns.image_path, seed=args.seed_b, force=False,
            task=args.task, negative_prompt=args.negative_prompt,
        )
        payload_b = p1._prepare_payload_on_device(payload_b)
    else:
        payload_b = p1._prepare_payload_on_device({
            "seed": args.seed_b,
            "latent_shape": payload_a["latent_shape"],
            "image_encoder_output": payload_a["image_encoder_output"],
            "inputs": payload_a["inputs"],
        })

    model = load_wan_transformer(config)
    scheduler_a = WanScheduler(config)
    scheduler_b = WanScheduler(config)
    model.set_scheduler(scheduler_a)
    tenant_a = TenantCtx("A", scheduler_a, payload_a["inputs"])
    tenant_b = TenantCtx("B", scheduler_b, payload_b["inputs"])

    comm_ratio = 0.35
    phase3_path = Path(args.phase3_json)
    if phase3_path.is_file():
        comm_ratio = float(json.loads(phase3_path.read_text()).get("one_step_profile", {}).get("comm_cuda_ratio", comm_ratio))

    measure_steps = args.measure_steps or scheduler_a.infer_steps

    for _ in range(args.warmup):
        _run_denoise_serial(model, scheduler_a, payload_a)

    dual_serial_s = _time_fn(lambda: (
        _run_denoise_serial(model, scheduler_a, payload_a),
        _run_denoise_serial(model, scheduler_b, payload_b),
    ))

    single_s: float | None = None
    single_decomposed_s: float | None = None
    single_decomposed_plain_s: float | None = None
    single_path_alignment: dict[str, Any] | None = None
    dual_decomposed_b2b_s: float | None = None
    if args.seq_p_size > 1 and dist.is_initialized():
        if args.align_iters > 0:
            single_path_alignment = _bench_single_path_alignment(
                model, scheduler_a, tenant_a, payload_a,
                measure_steps=measure_steps, align_iters=args.align_iters,
            )
            align_stats = single_path_alignment["stats"]
            single_s = align_stats["model_infer"]["mean_s"]
            single_decomposed_plain_s = align_stats["decomposed_plain"]["mean_s"]
            single_decomposed_s = align_stats["decomposed_patch"]["mean_s"]
        else:
            single_s = _time_fn(lambda: _run_denoise_serial(model, scheduler_a, payload_a))
            single_decomposed_s, _ = _run_single_a2a_pipeline(
                model, tenant_a, payload_a, steps=measure_steps,
            )
            single_decomposed_plain_s = _time_fn(
                lambda: _run_single_decomposed_once(
                    model, tenant_a, payload_a, patch_a2a=False, steps=measure_steps,
                ),
            )

        def _dual_decomposed_b2b() -> None:
            _run_single_decomposed_once(
                model, tenant_a, payload_a, patch_a2a=True, steps=measure_steps,
            )
            _run_single_decomposed_once(
                model, tenant_b, payload_b, patch_a2a=True, steps=measure_steps,
            )

        dual_decomposed_b2b_s = _time_fn(_dual_decomposed_b2b)
    else:
        single_s = _time_fn(lambda: _run_denoise_serial(model, scheduler_a, payload_a))

    dual_a2a_serial_s: float | None = None
    dual_a2a_overlap_s: float | None = None
    a2a_serial_stats: dict[str, Any] = {}
    a2a_overlap_stats: dict[str, Any] = {}
    dual_layer_serial_s: float | None = None
    dual_layer_overlap_s: float | None = None
    pair_serial_timings: list[dict[str, float]] = []
    pair_overlap_timings: list[dict[str, float]] = []
    err: str | None = None
    layer_err: str | None = None

    if args.seq_p_size > 1 and dist.is_initialized():
        try:
            dual_a2a_serial_s, a2a_serial_stats = _run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b,
                overlap=False, steps=measure_steps,
            )
            dual_a2a_overlap_s, a2a_overlap_stats = _run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b,
                overlap=True, steps=measure_steps,
            )
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if is_main_process():
                print(f"dual a2a pipeline failed: {exc}")

        if err is None and not args.inputs_cache_b:
            try:
                dual_layer_serial_s, pair_serial_timings = _run_dual_layer_pipeline(
                    model, tenant_a, tenant_b, payload_a, payload_b,
                    overlap=False, steps=measure_steps,
                )
                dual_layer_overlap_s, pair_overlap_timings = _run_dual_layer_pipeline(
                    model, tenant_a, tenant_b, payload_a, payload_b,
                    overlap=True, steps=measure_steps,
                )
            except Exception as exc:  # noqa: BLE001
                layer_err = str(exc)
                if is_main_process():
                    print(f"dual layer pipeline failed: {exc}")

    # One-step apples-to-apples: same decomposition, serial vs threaded overlap.
    one_step_a2a_serial_s: float | None = None
    one_step_a2a_overlap_s: float | None = None
    one_step_serial_s: float | None = None
    one_step_overlap_s: float | None = None
    one_step_pair_stats: dict[str, Any] = {}
    one_step_err: str | None = None
    if err is None and args.seq_p_size > 1 and dist.is_initialized():
        try:
            one_step_a2a_serial_s, a2a_one_serial = _run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b, overlap=False, steps=1,
            )
            one_step_a2a_overlap_s, a2a_one_overlap = _run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b, overlap=True, steps=1,
            )
            if not args.inputs_cache_b:
                one_step_serial_s, pts = _run_dual_layer_pipeline(
                    model, tenant_a, tenant_b, payload_a, payload_b, overlap=False, steps=1,
                )
                one_step_overlap_s, pto = _run_dual_layer_pipeline(
                    model, tenant_a, tenant_b, payload_a, payload_b, overlap=True, steps=1,
                )
                combined = []
                for a, b in zip(pts, pto):
                    combined.append({**a, **b})
                one_step_pair_stats = _summarize_pair_timings(combined)
            a2a_serial_stats = {**a2a_serial_stats, "one_step": a2a_one_serial}
            a2a_overlap_stats = {**a2a_overlap_stats, "one_step": a2a_one_overlap}
        except Exception as exc:  # noqa: BLE001
            one_step_err = str(exc)
            if is_main_process():
                print(f"one-step dual pipeline failed: {exc}")

    pure_compute_dual_s = (single_decomposed_s or single_s) * 2.0 * (1.0 - comm_ratio)

    def rps(n: float, t: float | None) -> float | None:
        return n / t if t and t > 0 else None

    fair_dual_baseline_s = dual_decomposed_b2b_s or dual_serial_s

    speedup_a2a_vs_serial = (
        dual_a2a_serial_s / dual_a2a_overlap_s
        if dual_a2a_serial_s and dual_a2a_overlap_s and dual_a2a_overlap_s > 0
        else None
    )
    speedup_a2a_one_step = (
        one_step_a2a_serial_s / one_step_a2a_overlap_s
        if one_step_a2a_serial_s and one_step_a2a_overlap_s and one_step_a2a_overlap_s > 0
        else None
    )
    speedup_vs_layer_serial = (
        dual_layer_serial_s / dual_layer_overlap_s
        if dual_layer_serial_s and dual_layer_overlap_s and dual_layer_overlap_s > 0
        else None
    )
    speedup_one_step = (
        one_step_serial_s / one_step_overlap_s
        if one_step_serial_s and one_step_overlap_s and one_step_overlap_s > 0
        else None
    )

    result = {
        "seq_p_size": args.seq_p_size,
        "task": args.task,
        "model_cls": args.model_cls,
        "model_path": args.model_path,
        "config_json": str(cfg_path),
        "mixed_resolution": bool(args.inputs_cache_b),
        "resolution_a": args.resolution_a or None,
        "resolution_b": args.resolution_b or None,
        "inputs_cache_a": args.inputs_cache,
        "inputs_cache_b": args.inputs_cache_b or None,
        "latent_shape_a": payload_a.get("latent_shape"),
        "latent_shape_b": payload_b.get("latent_shape"),
        "enable_cfg": config.get("enable_cfg", False),
        "cpu_offload": config.get("cpu_offload", False),
        "offload_granularity": config.get("offload_granularity"),
        "unload_modules": config.get("unload_modules", False),
        "measure_steps": measure_steps,
        "comm_cuda_ratio": comm_ratio,
        "single_transformer_s": single_s,
        "single_decomposed_s": single_decomposed_s,
        "single_decomposed_plain_s": single_decomposed_plain_s,
        "single_path_alignment": single_path_alignment,
        "dual_back_to_back_s": dual_serial_s,
        "dual_decomposed_back_to_back_s": dual_decomposed_b2b_s,
        "fair_dual_baseline_s": fair_dual_baseline_s,
        "dual_a2a_serial_s": dual_a2a_serial_s,
        "dual_a2a_overlap_s": dual_a2a_overlap_s,
        "one_step_a2a_serial_s": one_step_a2a_serial_s,
        "one_step_a2a_overlap_s": one_step_a2a_overlap_s,
        "a2a_serial_stats": a2a_serial_stats,
        "a2a_overlap_stats": a2a_overlap_stats,
        "dual_layer_serial_s": dual_layer_serial_s,
        "dual_layer_overlap_s": dual_layer_overlap_s,
        "one_step_layer_serial_s": one_step_serial_s,
        "one_step_layer_overlap_s": one_step_overlap_s,
        "one_step_pair_overlap_stats": one_step_pair_stats,
        "error": err,
        "layer_error": layer_err,
        "one_step_error": one_step_err,
        "throughput": {
            "single_rps": rps(1, single_s),
            "single_decomposed_rps": rps(1, single_decomposed_s),
            "dual_back_to_back_rps": rps(2, dual_serial_s),
            "dual_decomposed_back_to_back_rps": rps(2, dual_decomposed_b2b_s),
            "dual_a2a_serial_rps": rps(2, dual_a2a_serial_s),
            "dual_a2a_overlap_rps": rps(2, dual_a2a_overlap_s),
            "dual_layer_serial_rps": rps(2, dual_layer_serial_s),
            "dual_layer_overlap_rps": rps(2, dual_layer_overlap_s),
            "pure_compute_ceiling_rps": rps(2, pure_compute_dual_s),
        },
        "speedup": {
            "a2a_overlap_vs_a2a_serial": speedup_a2a_vs_serial,
            "a2a_overlap_vs_fair_dual_baseline": (
                fair_dual_baseline_s / dual_a2a_overlap_s
                if fair_dual_baseline_s and dual_a2a_overlap_s else None
            ),
            "a2a_serial_vs_fair_dual_baseline": (
                fair_dual_baseline_s / dual_a2a_serial_s
                if fair_dual_baseline_s and dual_a2a_serial_s else None
            ),
            "a2a_overlap_vs_back_to_back": (
                dual_serial_s / dual_a2a_overlap_s
                if dual_serial_s and dual_a2a_overlap_s else None
            ),
            "one_step_a2a_overlap_vs_serial": speedup_a2a_one_step,
            "layer_overlap_vs_layer_serial": speedup_vs_layer_serial,
            "layer_overlap_vs_fair_dual_baseline": (
                fair_dual_baseline_s / dual_layer_overlap_s
                if fair_dual_baseline_s and dual_layer_overlap_s else None
            ),
            "layer_overlap_vs_back_to_back": (
                dual_serial_s / dual_layer_overlap_s
                if dual_serial_s and dual_layer_overlap_s else None
            ),
            "one_step_overlap_vs_one_step_serial": speedup_one_step,
        },
        "interpretation": {
            "single_transformer_s": "model.infer(); mean of interleaved alignment runs when align_iters>0",
            "single_decomposed_plain_s": "Layer-split decomposed path without a2a patch (isolates decomposition overhead)",
            "single_decomposed_s": "Layer-split decomposed path with a2a patch (same as dual tenant path)",
            "single_path_alignment": "Round-robin interleaved samples for model.infer / plain / patch",
            "fair_dual_baseline_s": "Two single_decomposed (patch) runs back-to-back (primary fair baseline for 2 req)",
            "dual_a2a_serial": "Same pipeline; each all_to_all runs synchronously (patch on, overlap off)",
            "dual_a2a_overlap": "Each all_to_all/all_gather async on comm_stream; other tenant cross_ffn on compute_stream",
            "dual_layer_serial": "Layer-granularity phase split, phases sequential",
            "dual_layer_overlap": "Layer-granularity, two threads on dual CUDA streams",
            "dual_back_to_back": "Two full model.infer() runs, no cross-tenant overlap",
            "meaningful_speedup": "a2a_overlap_vs_a2a_serial > 1.0; a2a_overlap_vs_fair_dual_baseline > 1.0",
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
