#!/usr/bin/env python3
"""Per-layer self-attn vs cross-attn+FFN wall time (SP decomposed path).

Measures overlap-relevant segments:
  - self_attn block (includes Ulysses all_to_all inside infer_self_attn)
  - cross_ffn block (cross_attn + FFN + post_process; no Ulysses)
  - self_attn comm vs compute (comm = patched dist collectives during self_attn)

Run (MoE 480p P=4):
  torchrun --standalone --nproc_per_node=4 \\
    scripts/disagg/run_phase3_layer_self_cross_profile.py \\
    --config_json save_results/optimization_study/baseline_seqp4.json \\
    --tag dense --output_json save_results/optimization_study/p3_layer_profile_moe_480_dense_seqp4.json
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


class CommTracker:
    def __init__(self) -> None:
        self.active = False
        self.comm_s = 0.0
        self.a2a_calls = 0
        self.all_gather_calls = 0
        self._orig: dict[str, Callable[..., Any]] = {}

    def install(self) -> None:
        if self._orig:
            return
        tracker = self

        def wrap(name: str, kind: str) -> Callable[..., Any]:
            orig = getattr(dist, name)

            def patched(*args: Any, **kwargs: Any) -> Any:
                if not tracker.active:
                    return orig(*args, **kwargs)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = orig(*args, **kwargs)
                if kwargs.get("async_op") and out is not None and hasattr(out, "wait"):
                    out.wait()
                end.record()
                start.synchronize()
                end.synchronize()
                tracker.comm_s += start.elapsed_time(end) / 1000.0
                if kind == "a2a":
                    tracker.a2a_calls += 1
                else:
                    tracker.all_gather_calls += 1
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

    def flush_async(self) -> None:
        return


def _cuda_elapsed(fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    start.synchronize()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def _reduce_max(x: float) -> float:
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _profile_one_step(
    model: Any,
    p3: Any,
    tenant: Any,
    payload: dict[str, Any],
    tracker: CommTracker,
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

    def _run_self_timed(layer: int) -> tuple[Any, dict[str, float | int]]:
        tracker.active = True
        tracker.comm_s = 0.0
        tracker.a2a_calls = 0
        tracker.all_gather_calls = 0
        mid_holder: list[Any] = []

        def _fn() -> None:
            mid_holder.append(p3._run_self_attn_block(wan, ti, tenant, layer))

        self_s = _cuda_elapsed(_fn)
        tracker.flush_async()
        tracker.active = False
        metrics = {
            "self_attn_s": self_s,
            "comm_s": tracker.comm_s,
            "a2a_calls": tracker.a2a_calls,
            "all_gather_calls": tracker.all_gather_calls,
        }
        return mid_holder[0], metrics

    def _run_cross_timed(layer: int, mid: Any) -> dict[str, float]:
        tracker.active = False
        block = p3._ensure_block(tenant, wan, ti, layer)
        ti.block_idx = layer
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mid.mods
        x_holder: list[Any] = []
        attn_holder: list[Any] = []

        def _cross() -> None:
            x, attn_out = ti.infer_cross_attn(
                block.compute_phases[1], tenant.x, tenant.pre_infer_out.context, mid.y_out, gate_msa,
            )
            x_holder.append(x)
            attn_holder.append(attn_out)

        cross_attn_s = _cuda_elapsed(_cross)
        x, attn_out = x_holder[0], attn_holder[0]

        def _ffn_post() -> None:
            y = ti.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
            tenant.x = ti.post_process(x, y, c_gate_msa, tenant.pre_infer_out)

        ffn_post_s = _cuda_elapsed(_ffn_post)
        return {"cross_attn_s": cross_attn_s, "ffn_post_s": ffn_post_s}

    mid, self_m = _run_self_timed(0)
    for layer in range(num_blocks):
        cross_m = _run_cross_timed(layer, mid)

        self_s = _reduce_max(self_m["self_attn_s"])
        comm_s = _reduce_max(self_m["comm_s"])
        cross_attn_s = _reduce_max(cross_m["cross_attn_s"])
        ffn_post_s = _reduce_max(cross_m["ffn_post_s"])
        cross_ffn_s = cross_attn_s + ffn_post_s
        compute_s = max(0.0, self_s - comm_s)

        rows.append({
            "layer": layer,
            "self_attn_ms": round(self_s * 1000, 3),
            "self_attn_comm_ms": round(comm_s * 1000, 3),
            "self_attn_compute_ms": round(compute_s * 1000, 3),
            "cross_attn_ms": round(cross_attn_s * 1000, 3),
            "ffn_post_ms": round(ffn_post_s * 1000, 3),
            "cross_ffn_ms": round(cross_ffn_s * 1000, 3),
            "a2a_calls": int(self_m["a2a_calls"]),
            "all_gather_calls": int(self_m["all_gather_calls"]),
            "cross_covers_self": cross_ffn_s >= self_s,
            "cross_covers_comm": cross_ffn_s >= comm_s,
            "cross_attn_covers_comm": cross_attn_s >= comm_s,
        })

        if layer + 1 < num_blocks:
            mid, self_m = _run_self_timed(layer + 1)

    p3._finish_step_tenant(model, tenant)
    tenant.scheduler.step_post()
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return statistics.mean(r[key] for r in rows)

    cross_covers_self = sum(1 for r in rows if r["cross_covers_self"])
    cross_covers_comm = sum(1 for r in rows if r["cross_covers_comm"])
    cross_attn_covers_comm = sum(1 for r in rows if r["cross_attn_covers_comm"])
    return {
        "layers": len(rows),
        "self_attn_ms_mean": round(mean("self_attn_ms"), 3),
        "self_attn_comm_ms_mean": round(mean("self_attn_comm_ms"), 3),
        "self_attn_compute_ms_mean": round(mean("self_attn_compute_ms"), 3),
        "cross_attn_ms_mean": round(mean("cross_attn_ms"), 3),
        "ffn_post_ms_mean": round(mean("ffn_post_ms"), 3),
        "cross_ffn_ms_mean": round(mean("cross_ffn_ms"), 3),
        "comm_pct_of_self_mean": round(100 * mean("self_attn_comm_ms") / max(mean("self_attn_ms"), 1e-6), 1),
        "cross_attn_over_self_ratio_mean": round(mean("cross_attn_ms") / max(mean("self_attn_ms"), 1e-6), 3),
        "cross_over_self_ratio_mean": round(mean("cross_ffn_ms") / max(mean("self_attn_ms"), 1e-6), 3),
        "layers_cross_covers_self": cross_covers_self,
        "layers_cross_covers_comm": cross_covers_comm,
        "layers_cross_attn_covers_comm": cross_attn_covers_comm,
        "a2a_calls_per_layer_mean": round(statistics.mean(r["a2a_calls"] for r in rows), 2),
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
    parser.add_argument("--warmup_steps", type=int, default=0)
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

    tracker = CommTracker()
    tracker.install()
    try:
        for _ in range(args.warmup_steps):
            _profile_one_step(model, p3, tenant, payload, tracker)
            if dist.is_initialized():
                dist.barrier()

        if dist.is_initialized():
            dist.barrier()
        t0 = time.perf_counter()
        rows = _profile_one_step(model, p3, tenant, payload, tracker)
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
        "wall_s_one_step": wall,
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
