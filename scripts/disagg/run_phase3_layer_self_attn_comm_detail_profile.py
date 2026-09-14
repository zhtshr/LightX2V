#!/usr/bin/env python3
"""Fine-grained self-attn comm breakdown: per all_to_all / all_gather timing.

Within each layer's self_attn, records:
  - wall time of each collective (call order preserved)
  - gap_ms between consecutive collectives (local compute between comm bursts)
  - segment totals: input_a2a (first 3), middle_gap, output_a2a (4th), all_gather

Run (MoE 480p SLA P=4):
  torchrun --standalone --nproc_per_node=4 \\
    scripts/disagg/run_phase3_layer_self_attn_comm_detail_profile.py \\
    --config_json save_results/optimization_study/baseline_moe_i2v_480_sla_triton_seqp4.json \\
    --tag moe_sla --output_json save_results/optimization_study/p3_self_attn_comm_detail_moe_480_sla_seqp4.json
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


class CommDetailTracker:
  def __init__(self) -> None:
    self.active = False
    self.calls: list[dict[str, Any]] = []
    self._orig: dict[str, Callable[..., Any]] = {}
    self._layer_start_event: torch.cuda.Event | None = None
    self._since_last_comm_end: torch.cuda.Event | None = None

  def begin_layer(self) -> None:
    self.calls = []
    self._layer_start_event = torch.cuda.Event(enable_timing=True)
    self._layer_start_event.record()
    self._since_last_comm_end = self._layer_start_event

  def install(self) -> None:
    if self._orig:
      return
    tracker = self

    def wrap(name: str, kind: str) -> Callable[..., Any]:
      orig = getattr(dist, name)

      def patched(*args: Any, **kwargs: Any) -> Any:
        if not tracker.active:
          return orig(*args, **kwargs)

        gap_ms = 0.0
        if tracker._since_last_comm_end is not None:
          gap_end = torch.cuda.Event(enable_timing=True)
          gap_end.record()
          gap_end.synchronize()
          tracker._since_last_comm_end.synchronize()
          gap_ms = tracker._since_last_comm_end.elapsed_time(gap_end)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig(*args, **kwargs)
        if kwargs.get("async_op") and out is not None and hasattr(out, "wait"):
          out.wait()
        end.record()
        start.synchronize()
        end.synchronize()
        comm_ms = start.elapsed_time(end)

        tracker.calls.append({
          "idx": len(tracker.calls),
          "kind": kind,
          "op": name,
          "gap_before_ms": round(gap_ms, 3),
          "comm_ms": round(comm_ms, 3),
        })
        tracker._since_last_comm_end = end
        return out

      return patched

    for name in ("all_to_all_single", "all_to_all"):
      if hasattr(dist, name):
        self._orig[name] = getattr(dist, name)
        setattr(dist, name, wrap(name, "a2a"))
    if hasattr(dist, "all_gather"):
      self._orig["all_gather"] = dist.all_gather
      dist.all_gather = wrap("all_gather", "all_gather")

  def restore(self) -> None:
    for name, fn in self._orig.items():
      setattr(dist, name, fn)
    self._orig.clear()

  def finish_layer(self, self_attn_ms: float) -> dict[str, Any]:
    tail_gap_ms = 0.0
    if self._since_last_comm_end is not None:
      tail_end = torch.cuda.Event(enable_timing=True)
      tail_end.record()
      tail_end.synchronize()
      self._since_last_comm_end.synchronize()
      tail_gap_ms = self._since_last_comm_end.elapsed_time(tail_end)

    comm_sum_ms = sum(c["comm_ms"] for c in self.calls)
    gap_sum_ms = sum(c["gap_before_ms"] for c in self.calls) + tail_gap_ms

    a2a_calls = [c for c in self.calls if c["kind"] == "a2a"]
    ag_calls = [c for c in self.calls if c["kind"] == "all_gather"]

    input_a2a = a2a_calls[:3]
    output_a2a = a2a_calls[3:]

    def _sum_ms(items: list[dict[str, Any]], key: str) -> float:
      return sum(x[key] for x in items)

    input_comm_ms = _sum_ms(input_a2a, "comm_ms")
    output_comm_ms = _sum_ms(output_a2a, "comm_ms")
    ag_comm_ms = _sum_ms(ag_calls, "comm_ms")

  # gap before first a2a
    gap_before_first = a2a_calls[0]["gap_before_ms"] if a2a_calls else 0.0
  # gap between 3rd a2a end and 4th a2a start
    gap_input_to_output = 0.0
    if len(a2a_calls) >= 4:
      gap_input_to_output = a2a_calls[3]["gap_before_ms"]
  # gap between last a2a and all_gather
    gap_a2a_to_ag = ag_calls[0]["gap_before_ms"] if ag_calls else 0.0

    return {
      "self_attn_ms": round(self_attn_ms, 3),
      "comm_sum_ms": round(comm_sum_ms, 3),
      "gap_sum_ms": round(gap_sum_ms, 3),
      "residual_ms": round(max(0.0, self_attn_ms - comm_sum_ms - gap_sum_ms), 3),
      "num_a2a": len(a2a_calls),
      "num_all_gather": len(ag_calls),
      "input_a2a_ms": round(input_comm_ms, 3),
      "middle_gap_ms": round(gap_input_to_output, 3),
      "output_a2a_ms": round(output_comm_ms, 3),
      "all_gather_ms": round(ag_comm_ms, 3),
      "gap_before_first_a2a_ms": round(gap_before_first, 3),
      "gap_a2a_to_all_gather_ms": round(gap_a2a_to_ag, 3),
      "tail_after_last_comm_ms": round(tail_gap_ms, 3),
      "per_call": list(self.calls),
    }


def _cuda_elapsed(fn: Callable[[], Any]) -> float:
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  fn()
  end.record()
  start.synchronize()
  end.synchronize()
  return start.elapsed_time(end)


def _reduce_max(x: float) -> float:
  if not dist.is_initialized():
    return x
  t = torch.tensor([x], device="cuda", dtype=torch.float64)
  dist.all_reduce(t, op=dist.ReduceOp.MAX)
  return float(t.item())


def _profile_layers(
    model: Any,
    p3: Any,
    tenant: Any,
    payload: dict[str, Any],
    tracker: CommDetailTracker,
    *,
    layer_start: int,
    layer_end: int,
) -> list[dict[str, Any]]:
  tenant.scheduler.prepare(
    seed=int(payload["seed"]),
    latent_shape=payload["latent_shape"],
    image_encoder_output=payload["image_encoder_output"],
  )
  tenant.scheduler.step_pre(step_index=0)
  p3._pre_infer_tenant(model, tenant)

  wan, ti = p3._bind_tenant(model, tenant)
  num_blocks = len(wan.transformer_weights.blocks)
  p3._preload_blocks(tenant, wan, ti, num_blocks)

  rows: list[dict[str, Any]] = []
  end_layer = min(layer_end, num_blocks)

  for layer in range(layer_start, end_layer):
    tracker.begin_layer()
    tracker.active = True
    mid_holder: list[Any] = []

    def _fn() -> None:
      mid_holder.append(p3._run_self_attn_block(wan, ti, tenant, layer))

    self_ms = _cuda_elapsed(_fn)
    tracker.active = False
    detail = tracker.finish_layer(self_ms)
    detail["layer"] = layer
    rows.append(detail)

  p3._finish_step_tenant(model, tenant)
  tenant.scheduler.step_post()
  return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
  return statistics.mean(r[key] for r in rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  def per_call_mean(idx: int, key: str) -> float | None:
    vals = []
    for r in rows:
      calls = r.get("per_call", [])
      if idx < len(calls):
        vals.append(calls[idx][key])
    return round(statistics.mean(vals), 3) if vals else None

  return {
    "layers_profiled": len(rows),
    "self_attn_ms_mean": round(_mean(rows, "self_attn_ms"), 3),
    "comm_sum_ms_mean": round(_mean(rows, "comm_sum_ms"), 3),
    "gap_sum_ms_mean": round(_mean(rows, "gap_sum_ms"), 3),
    "input_a2a_ms_mean": round(_mean(rows, "input_a2a_ms"), 3),
    "middle_gap_ms_mean": round(_mean(rows, "middle_gap_ms"), 3),
    "output_a2a_ms_mean": round(_mean(rows, "output_a2a_ms"), 3),
    "all_gather_ms_mean": round(_mean(rows, "all_gather_ms"), 3),
    "gap_before_first_a2a_ms_mean": round(_mean(rows, "gap_before_first_a2a_ms"), 3),
    "gap_a2a_to_all_gather_ms_mean": round(_mean(rows, "gap_a2a_to_all_gather_ms"), 3),
    "tail_after_last_comm_ms_mean": round(_mean(rows, "tail_after_last_comm_ms"), 3),
    "comm_pct_of_self": round(100 * _mean(rows, "comm_sum_ms") / max(_mean(rows, "self_attn_ms"), 1e-6), 1),
    "middle_gap_pct_of_self": round(100 * _mean(rows, "middle_gap_ms") / max(_mean(rows, "self_attn_ms"), 1e-6), 1),
    "per_call_comm_ms_mean": {
      "a2a_0_input_q": per_call_mean(0, "comm_ms"),
      "a2a_1_input_k": per_call_mean(1, "comm_ms"),
      "a2a_2_input_v": per_call_mean(2, "comm_ms"),
      "a2a_3_output": per_call_mean(3, "comm_ms"),
      "all_gather_4": per_call_mean(4, "comm_ms"),
    },
    "per_call_gap_before_ms_mean": {
      "a2a_0": per_call_mean(0, "gap_before_ms"),
      "a2a_1": per_call_mean(1, "gap_before_ms"),
      "a2a_2": per_call_mean(2, "gap_before_ms"),
      "a2a_3": per_call_mean(3, "gap_before_ms"),
      "all_gather": per_call_mean(4, "gap_before_ms"),
    },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seq_p_size", type=int, default=0)
  parser.add_argument("--config_json", required=True)
  parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
  parser.add_argument("--task", default="i2v", choices=("i2v", "t2v"))
  parser.add_argument("--model_cls", default="wan2.2_moe")
  parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
  parser.add_argument("--tag", default="")
  parser.add_argument("--layer_start", type=int, default=1, help="skip layer 0 warmup spike")
  parser.add_argument("--layer_end", type=int, default=40)
  parser.add_argument("--warmup_layers", type=int, default=2)
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

  payload = p1._prepare_inputs_cache(
    config, Path(args.inputs_cache), prompt="bench", image_path="", seed=42, task=args.task,
  )
  payload = p1._prepare_payload_on_device(payload)

  scheduler = WanScheduler(config)
  model = load_wan_transformer(config)
  model.scheduler = scheduler
  tenant = p3.TenantCtx(name="A", scheduler=scheduler, inputs=payload["inputs"])

  tracker = CommDetailTracker()
  tracker.install()
  try:
    if args.warmup_layers > 0:
      _profile_layers(
        model, p3, tenant, payload, tracker,
        layer_start=0, layer_end=args.warmup_layers,
      )
      if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
      dist.barrier()
    t0 = time.perf_counter()
    rows = _profile_layers(
      model, p3, tenant, payload, tracker,
      layer_start=args.layer_start, layer_end=args.layer_end,
    )
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    if dist.is_initialized():
      dist.barrier()
    wall = time.perf_counter() - t0
  finally:
    tracker.restore()

  summary = _summarize(rows)
  result = {
    "tag": args.tag or Path(args.config_json).stem,
    "config_json": args.config_json,
    "seq_p_size": config.get("parallel", {}).get("seq_p_size", 1),
    "self_attn_type": config.get("self_attn_1_type"),
    "resolution": f"{config.get('target_height')}x{config.get('target_width')}",
    "layer_range": [args.layer_start, args.layer_end],
    "wall_s_profiled_layers": wall,
    "summary": summary,
    "per_layer": rows,
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
