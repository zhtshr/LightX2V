#!/usr/bin/env python3
"""Measure which collectives truly overlap with cross_ffn in legacy dual overlap.

Per overlap window records (CUDA events on comm_stream vs compute_stream):
  - comm_ms, compute_ms, window_wall_ms
  - compute_cb_ran / compute_cb_noop (_OnceCompute skip)
  - gpu_overlap_ms (intersection of comm and compute GPU intervals)
  - saved_vs_serial_ms = comm_ms + effective_compute_ms - window_wall_ms

Run:
  torchrun --standalone --nproc_per_node=4 \\
    scripts/disagg/run_phase3_overlap_comm_truth_profile.py \\
    --config_json save_results/optimization_study/baseline_moe_i2v_480_sla_triton_seqp4.json \\
    --layer_start 1 --layer_end 6 \\
    --output_json save_results/optimization_study/p3_overlap_comm_truth_moe_480_sla_seqp4.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_phase3():
    path = Path(__file__).with_name("run_phase3_dual_overlap_bench.py")
    spec = importlib.util.spec_from_file_location("phase3", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _CountingOnceCompute:
    def __init__(self, fn: Callable[[], None]):
        self._fn = fn
        self._done = False
        self.last_call_ran = False
        self.last_call_noop = False

    def __call__(self) -> None:
        if self._done:
            self.last_call_ran = False
            self.last_call_noop = True
            return
        self.last_call_ran = True
        self.last_call_noop = False
        self._fn()
        self._done = True


class InstrumentedA2AOrchestrator:
    """Drop-in legacy overlap orchestrator with per-window CUDA timing."""

    def __init__(self, device: torch.device, *, capture_ti_snap: Callable, apply_ti_snap: Callable):
        self.device = device
        self.enabled = False
        self.overlap_enabled = False
        self.other_compute_cb: Any = None
        self.in_other_compute = False
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)
        self.ti_ref: Any | None = None
        self.a2a_in_self_attn = 0
        self.phase_label = ""
        self.records: list[dict[str, Any]] = []
        self._orig: dict[str, Callable[..., Any]] = {}
        self._capture_ti_snap = capture_ti_snap
        self._apply_ti_snap = apply_ti_snap

    def reset_self_attn_counters(self) -> None:
        self.a2a_in_self_attn = 0

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

    def _maybe_overlap(self, comm_launch: Callable[[], Any], *, kind: str) -> Any:
        if not self.overlap_enabled or self.other_compute_cb is None or self.in_other_compute:
            with torch.cuda.stream(self.comm_stream):
                return comm_launch(async_op=False)

        a2a_idx = self.a2a_in_self_attn if kind == "a2a" else -1
        if kind == "a2a":
            self.a2a_in_self_attn += 1

        cb_holder = self.other_compute_cb
        cb_holder.last_call_ran = False
        cb_holder.last_call_noop = False

        ev_comm_launch = torch.cuda.Event(enable_timing=True)
        ev_comm_done = torch.cuda.Event(enable_timing=True)
        ev_comp_start = torch.cuda.Event(enable_timing=True)
        ev_comp_done = torch.cuda.Event(enable_timing=True)
        t_wall0 = time.perf_counter()

        with torch.cuda.stream(self.comm_stream):
            ev_comm_launch.record(self.comm_stream)
            work = comm_launch(async_op=True)

        saved_snap = None
        if self.ti_ref is not None:
            saved_snap = self._capture_ti_snap(self.ti_ref)

        self.in_other_compute = True
        try:
            with torch.cuda.stream(self.compute_stream):
                ev_comp_start.record(self.compute_stream)
                cb_holder()
                ev_comp_done.record(self.compute_stream)
        finally:
            self.in_other_compute = False
            if saved_snap is not None and self.ti_ref is not None:
                self._apply_ti_snap(self.ti_ref, saved_snap)

        cb_ran = bool(cb_holder.last_call_ran)
        cb_noop = bool(cb_holder.last_call_noop)

        with torch.cuda.stream(self.comm_stream):
            if work is not None and hasattr(work, "wait"):
                work.wait()
            ev_comm_done.record(self.comm_stream)

        self.comm_stream.synchronize()
        self.compute_stream.synchronize()
        wall_ms = (time.perf_counter() - t_wall0) * 1000.0

        comm_ms = ev_comm_launch.elapsed_time(ev_comm_done)
        comp_ms = ev_comp_start.elapsed_time(ev_comp_done) if cb_ran and not cb_noop else 0.0

        # GPU interval overlap: comm [launch, done] vs compute [start, done]
        # Use elapsed from common reference ev_comm_launch
        comp_start_off = ev_comm_launch.elapsed_time(ev_comp_start)
        comp_end_off = ev_comm_launch.elapsed_time(ev_comp_done)
        comm_end_off = comm_ms
        overlap_start = max(0.0, comp_start_off)
        overlap_end = min(comm_end_off, comp_end_off)
        gpu_overlap_ms = max(0.0, overlap_end - overlap_start)

        effective_comp = comp_ms if cb_ran else 0.0
        serial_ms = comm_ms + effective_comp
        saved_ms = max(0.0, serial_ms - wall_ms)

        seg = "input_q" if a2a_idx == 0 else "input_k" if a2a_idx == 1 else "input_v" if a2a_idx == 2 else "output" if a2a_idx == 3 else "all_gather"

        self.records.append({
            "phase": self.phase_label,
            "kind": kind,
            "a2a_idx": a2a_idx,
            "segment": seg,
            "comm_ms": round(comm_ms, 3),
            "compute_ms": round(effective_comp, 3),
            "window_wall_ms": round(wall_ms, 3),
            "gpu_overlap_ms": round(gpu_overlap_ms, 3),
            "saved_vs_serial_ms": round(saved_ms, 3),
            "compute_cb_ran": cb_ran,
            "compute_cb_noop": cb_noop,
        })
        return work

    def _patched_all_to_all_single(self, output, input, group=None, async_op=False, **kwargs):
        if not self.enabled:
            return self._orig["all_to_all_single"](output, input, group=group, async_op=async_op, **kwargs)

        def launch(async_op: bool):
            return self._orig["all_to_all_single"](output, input, group=group, async_op=async_op, **kwargs)

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


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def filt(key: str, val: Any) -> list[dict[str, Any]]:
        return [r for r in records if r.get(key) == val]

    def mean_of(rows: list[dict[str, Any]], key: str) -> float:
        return statistics.mean(r[key] for r in rows) if rows else 0.0

    by_seg: dict[str, Any] = {}
    for seg in ("input_q", "input_k", "input_v", "output", "all_gather"):
        rows = [r for r in records if r["segment"] == seg]
        ran = [r for r in rows if r["compute_cb_ran"]]
        noop = [r for r in rows if r["compute_cb_noop"]]
        by_seg[seg] = {
            "windows": len(rows),
            "compute_ran": len(ran),
            "compute_noop": len(noop),
            "comm_ms_mean": round(mean_of(rows, "comm_ms"), 3),
            "compute_ms_mean_when_ran": round(mean_of(ran, "compute_ms"), 3),
            "window_wall_ms_mean": round(mean_of(rows, "window_wall_ms"), 3),
            "gpu_overlap_ms_mean": round(mean_of(rows, "gpu_overlap_ms"), 3),
            "saved_vs_serial_ms_mean": round(mean_of(rows, "saved_vs_serial_ms"), 3),
            "saved_vs_serial_ms_sum": round(sum(r["saved_vs_serial_ms"] for r in rows), 3),
        }

    all_ran = [r for r in records if r["compute_cb_ran"]]
    all_noop = [r for r in records if r["compute_cb_noop"]]
    return {
        "total_overlap_windows": len(records),
        "windows_compute_ran": len(all_ran),
        "windows_compute_noop": len(all_noop),
        "total_saved_vs_serial_ms": round(sum(r["saved_vs_serial_ms"] for r in records), 3),
        "total_gpu_overlap_ms": round(sum(r["gpu_overlap_ms"] for r in records), 3),
        "by_segment": by_seg,
        "first_compute_ran_sample": all_ran[0] if all_ran else None,
    }


def _profile_legacy_overlap_layers(
    model: Any,
    p3: Any,
    tenant_a: Any,
    tenant_b: Any,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    orch: InstrumentedA2AOrchestrator,
    *,
    layer_start: int,
    layer_end: int,
) -> list[dict[str, Any]]:
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
    tenant_a.scheduler.step_pre(step_index=0)
    tenant_b.scheduler.step_pre(step_index=0)
    p3._pre_infer_tenant(model, tenant_a)
    p3._pre_infer_tenant(model, tenant_b)

    wan_a, ti_a = p3._bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    p3._preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = p3._bind_tenant(model, tenant_b)
    p3._preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    orch.enabled = True
    orch.overlap_enabled = False
    orch.records = []

    mid_a = p3._run_self_attn_block(wan_a, ti_a, tenant_a, 0)

    end_layer = min(layer_end, num_blocks)
    for k in range(layer_start, end_layer):
        cb_b_self = _CountingOnceCompute(
            lambda k=k, m=mid_a: p3._run_cross_ffn_block(
                *p3._bind_tenant(model, tenant_a), tenant_a, k, m,
            ),
        )
        orch.overlap_enabled = True
        orch.other_compute_cb = cb_b_self
        orch.phase_label = f"B_self_L{k}"
        wan_b, ti_b = p3._bind_tenant(model, tenant_b)
        orch.ti_ref = ti_b
        orch.reset_self_attn_counters()
        try:
            mid_b = p3._run_self_attn_block(wan_b, ti_b, tenant_b, k)
        finally:
            orch.ti_ref = None

        if k + 1 < num_blocks:
            cb_a_self = _CountingOnceCompute(
                lambda k=k, mb=mid_b: p3._run_cross_ffn_block(
                    *p3._bind_tenant(model, tenant_b), tenant_b, k, mb,
                ),
            )
            orch.overlap_enabled = True
            orch.other_compute_cb = cb_a_self
            orch.phase_label = f"A_self_L{k + 1}"
            wan_a, ti_a = p3._bind_tenant(model, tenant_a)
            orch.ti_ref = ti_a
            orch.reset_self_attn_counters()
            try:
                mid_a = p3._run_self_attn_block(wan_a, ti_a, tenant_a, k + 1)
            finally:
                orch.ti_ref = None

    orch.overlap_enabled = False
    orch.enabled = False
    orch.compute_stream.synchronize()
    orch.comm_stream.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return orch.records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--layer_start", type=int, default=1)
    parser.add_argument("--layer_end", type=int, default=6)
    parser.add_argument("--warmup_layers", type=int, default=1)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    p1 = _load_phase1()
    p3 = _load_phase3()
    ns = argparse.Namespace(
        seq_p_size=args.seq_p_size,
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
    )
    config = p1._load_config(ns)
    p1._init_distributed(config)
    seed_all(42)

    payload_a = p1._prepare_inputs_cache(
        config, Path(args.inputs_cache), prompt="bench", image_path="", seed=42, task=args.task,
    )
    payload_a = p1._prepare_payload_on_device(payload_a)
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    tenant_a = p3.TenantCtx("A", WanScheduler(config), payload_a["inputs"])
    tenant_b = p3.TenantCtx("B", WanScheduler(config), payload_b["inputs"])

    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = InstrumentedA2AOrchestrator(
        device,
        capture_ti_snap=p3._capture_ti_snap,
        apply_ti_snap=p3._apply_ti_snap,
    )
    orch.install()
    try:
        if args.warmup_layers > 0:
            _profile_legacy_overlap_layers(
                model, p3, tenant_a, tenant_b, payload_a, payload_b, orch,
                layer_start=0, layer_end=args.warmup_layers,
            )
            orch.records = []
            if dist.is_initialized():
                dist.barrier()

        records = _profile_legacy_overlap_layers(
            model, p3, tenant_a, tenant_b, payload_a, payload_b, orch,
            layer_start=args.layer_start, layer_end=args.layer_end,
        )
    finally:
        orch.restore()

    summary = _summarize_records(records)
    result = {
        "tag": "legacy_overlap_comm_truth",
        "config_json": args.config_json,
        "resolution": f"{config.get('target_height')}x{config.get('target_width')}",
        "self_attn_type": config.get("self_attn_1_type"),
        "seq_p_size": config.get("parallel", {}).get("seq_p_size", 1),
        "layer_range": [args.layer_start, args.layer_end],
        "summary": summary,
        "per_window": records,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if is_main_process():
        print(json.dumps({"summary": summary, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
