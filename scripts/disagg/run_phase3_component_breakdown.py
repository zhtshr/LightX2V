#!/usr/bin/env python3
"""Measure per-component CUDA wall time for one denoise step (self/cross/ffn/post)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class CudaTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self._starts: dict[str, torch.cuda.Event] = {}
        self._ends: dict[str, torch.cuda.Event] = {}

    def start(self, key: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._starts[key] = e

    def stop(self, key: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._ends[key] = e

    def sync_total(self, key: str) -> float:
        if key not in self._starts or key not in self._ends:
            return 0.0
        self._starts[key].synchronize()
        self._ends[key].synchronize()
        ms = self._starts[key].elapsed_time(self._ends[key])
        self.totals[key] = self.totals.get(key, 0.0) + ms / 1000.0
        return ms / 1000.0


def _patch_infer(ti: Any, timer: CudaTimer) -> None:
    orig_self = ti.infer_self_attn
    orig_cross = ti.infer_cross_attn
    orig_ffn = ti.infer_ffn

    def timed_self(phase, x, shift_msa, scale_msa):
        timer.start("self_attn")
        out = orig_self(phase, x, shift_msa, scale_msa)
        timer.stop("self_attn")
        return out

    def timed_cross(phase, x, context, y_out, gate_msa):
        timer.start("cross_attn")
        out = orig_cross(phase, x, context, y_out, gate_msa)
        timer.stop("cross_attn")
        return out

    def timed_ffn(phase, x, attn_out, c_shift_msa, c_scale_msa):
        timer.start("ffn")
        out = orig_ffn(phase, x, attn_out, c_shift_msa, c_scale_msa)
        timer.stop("ffn")
        return out

    ti.infer_self_attn = MethodType(lambda _, *a, **k: timed_self(*a, **k), ti)
    ti.infer_cross_attn = MethodType(lambda _, *a, **k: timed_cross(*a, **k), ti)
    ti.infer_ffn = MethodType(lambda _, *a, **k: timed_ffn(*a, **k), ti)


def _patch_post_gather(model: Any, timer: CudaTimer) -> None:
    orig = model._seq_parallel_post_process

    def timed_post(x):
        timer.start("post_all_gather")
        out = orig(x)
        timer.stop("post_all_gather")
        return out

    model._seq_parallel_post_process = MethodType(timed_post, model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_p_size", type=int, required=True)
    parser.add_argument("--config_json", default="/root/zht/LightX2V/save_results/optimization_study/baseline_seqp4.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    p1 = _load_phase1()
    config = p1._load_config(argparse.Namespace(
        seq_p_size=args.seq_p_size,
        config_json=args.config_json,
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
    ))
    # keep config_json defaults (phase3 uses cpu_offload=block)
    p1._init_distributed(config)
    seed_all(42)
    rank = dist.get_rank() if dist.is_initialized() else 0

    payload = p1._prepare_inputs_cache(
        config, Path(args.inputs_cache), prompt="bench", image_path="", seed=42, task="i2v",
    )
    payload = p1._prepare_payload_on_device(payload)

    scheduler = WanScheduler(config)
    model = load_wan_transformer(config)
    model.scheduler = scheduler

    timer = CudaTimer()
    _patch_infer(model.transformer_infer, timer)
    if config.get("seq_parallel"):
        _patch_post_gather(model, timer)

    scheduler.prepare(seed=payload["seed"], latent_shape=payload["latent_shape"], image_encoder_output=payload["image_encoder_output"])
    scheduler.step_pre(step_index=0)

    if dist.is_initialized():
        dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.infer(payload["inputs"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    wall = time.perf_counter() - t0
    scheduler.step_post()

    # finalize cuda event totals
    for key in list(timer._starts):
        if key in timer._ends:
            timer.sync_total(key)

    result = {
        "seq_p_size": args.seq_p_size,
        "rank": rank,
        "wall_s": wall,
        "components_s": dict(sorted(timer.totals.items())),
        "component_sum_s": sum(timer.totals.values()),
        "unaccounted_s": max(0.0, wall - sum(timer.totals.values())),
    }
    out = args.out_json or str(
        Path("/root/zht/LightX2V/save_results/optimization_study")
        / f"p3_component_breakdown_seqp{args.seq_p_size}.json"
    )
    Path(out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if is_main_process():
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
