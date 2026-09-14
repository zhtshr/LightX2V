#!/usr/bin/env python3
"""Measured TP=2 overhead breakdown: comm vs compute, module timers, cross K/V split."""

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
from torch.profiler import ProfilerActivity, profile

from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTP
from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all

_COMM_KEYWORDS = (
    "nccl",
    "all_to_all",
    "alltoall",
    "dist.all",
    "broadcast",
    "reduce_scatter",
    "all_gather",
    "barrier",
    "isend",
    "irecv",
    "send_recv",
    "p2p",
)


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

    def start(self, key: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._starts[key] = e

    def stop(self, key: str) -> None:
        if key not in self._starts:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        start = self._starts.pop(key)
        start.synchronize()
        end.synchronize()
        self.totals[key] = self.totals.get(key, 0.0) + start.elapsed_time(end) / 1000.0


def _is_comm_event(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in _COMM_KEYWORDS)


def _profile_one_step(model: Any, scheduler: WanScheduler, payload: dict[str, Any], step_index: int) -> dict[str, float]:
    seed = int(payload["seed"])
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]

    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    scheduler.step_pre(step_index=step_index)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof:
        model.infer(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scheduler.step_post()

    comm_cuda_us = 0.0
    compute_cuda_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type.name != "CUDA":
            continue
        if _is_comm_event(evt.key):
            comm_cuda_us += evt.device_time_total
        else:
            compute_cuda_us += evt.device_time_total
    total = comm_cuda_us + compute_cuda_us
    return {
        "step_index": step_index,
        "comm_cuda_us": comm_cuda_us,
        "compute_cuda_us": compute_cuda_us,
        "comm_cuda_ratio": (comm_cuda_us / total) if total > 0 else 0.0,
        "comm_cuda_s": comm_cuda_us / 1e6,
        "compute_cuda_s": compute_cuda_us / 1e6,
    }


def _patch_mm_tp(timer: CudaTimer) -> Any:
    orig = MMWeightTP.apply

    def patched(self, input_tensor):
        timer.start("tp_mm_gemm")
        output = self._mm.apply(input_tensor)
        timer.stop("tp_mm_gemm")
        if self.split_dim == "row" and self.tp_size > 1 and self.tp_group is not None:
            timer.start("tp_row_allreduce")
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
            timer.stop("tp_row_allreduce")
            if self._row_split_bias is not None:
                output = output + self._row_split_bias
        return output

    MMWeightTP.apply = patched
    return orig


def _patch_rms_tp(timer: CudaTimer) -> Any:
    orig = RMSWeightTP.apply

    def patched(self, input_tensor):
        local_sum = input_tensor.pow(2).sum(-1, keepdim=True)
        if self.tp_size > 1 and self.tp_group is not None:
            timer.start("tp_rms_allreduce")
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.tp_group)
            timer.stop("tp_rms_allreduce")
        hidden_dim = input_tensor.shape[-1] * self.tp_size
        global_mean = local_sum / hidden_dim
        if self.sensitive_layer_dtype != self.infer_dtype:
            input_tensor = input_tensor * torch.rsqrt(global_mean.float() + self.eps).to(self.infer_dtype)
            input_tensor = (input_tensor * self._get_actual_weight()).to(self.infer_dtype)
        else:
            input_tensor = input_tensor * torch.rsqrt(global_mean + self.eps)
            input_tensor = input_tensor * self._get_actual_weight()
        return input_tensor

    RMSWeightTP.apply = patched
    return orig


def _patch_transformer_infer(ti: Any, timer: CudaTimer) -> None:
    orig_self = ti.infer_self_attn
    orig_cross = ti.infer_cross_attn
    orig_ffn = ti.infer_ffn

    def timed_self(phase, x, shift_msa, scale_msa):
        timer.start("self_attn")
        out = orig_self(phase, x, shift_msa, scale_msa)
        timer.stop("self_attn")
        return out

    def timed_cross(phase, x, context, y_out, gate_msa):
        if ti.sensitive_layer_dtype != ti.infer_dtype:
            x = x.to(ti.sensitive_layer_dtype) + y_out.to(ti.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        timer.start("cross_norm3")
        norm3_out = phase.norm3.apply(x)
        timer.stop("cross_norm3")

        n, d = ti.num_heads, ti.head_dim
        timer.start("cross_q_path")
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        timer.stop("cross_q_path")

        timer.start("cross_kv_path")
        k_full = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, ti.global_num_heads, d)
        v_full = phase.cross_attn_v.apply(context).view(-1, ti.global_num_heads, d)
        timer.stop("cross_kv_path")

        if ti.tp_size > 1:
            head_start = ti.tp_rank * n
            k = k_full[:, head_start : head_start + n, :]
            v = v_full[:, head_start : head_start + n, :]
        else:
            k, v = k_full, v_full

        if ti.cross_attn_cu_seqlens_q is None:
            ti.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]], device=q.device).cumsum(0, dtype=torch.int32)
        if ti.cross_attn_cu_seqlens_kv is None:
            ti.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]], device=k.device).cumsum(0, dtype=torch.int32)

        timer.start("cross_attn_kernel")
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=ti.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=ti.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )
        timer.stop("cross_attn_kernel")

        timer.start("cross_o")
        attn_out = phase.cross_attn_o.apply(attn_out)
        timer.stop("cross_o")

        return x, attn_out

    def timed_ffn(phase, x, attn_out, c_shift_msa, c_scale_msa):
        timer.start("ffn")
        out = orig_ffn(phase, x, attn_out, c_shift_msa, c_scale_msa)
        timer.stop("ffn")
        return out

    ti.infer_self_attn = MethodType(lambda _, *a, **k: timed_self(*a, **k), ti)
    ti.infer_cross_attn = MethodType(lambda _, *a, **k: timed_cross(*a, **k), ti)
    ti.infer_ffn = MethodType(lambda _, *a, **k: timed_ffn(*a, **k), ti)


def _patch_wan_models(model: Any, timer: CudaTimer) -> list[Any]:
    patched: list[Any] = []
    if hasattr(model, "model") and isinstance(model.model, list):
        for sub in model.model:
            if sub is not None and hasattr(sub, "transformer_infer"):
                _patch_transformer_infer(sub.transformer_infer, timer)
                patched.append(sub)
    elif hasattr(model, "transformer_infer"):
        _patch_transformer_infer(model.transformer_infer, timer)
        patched.append(model)
    return patched


def _bench_with_prepare_timer(p1, scheduler, model, payload, timer: CudaTimer) -> float:
    seed = int(payload["seed"])
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    timer.start("scheduler_prepare")
    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    timer.stop("scheduler_prepare")

    infer_steps = scheduler.infer_steps
    for step_index in range(infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def _write_markdown(out_md: Path, result: dict[str, Any]) -> None:
    wall = result["transformer_compute_s"]
    mod = result["module_timers_4step_s"]
    ar = result["allreduce_timers_4step_s"]
    prof = result["one_step_torch_profiler"]
    comm_ar = ar.get("tp_row_allreduce", 0.0) + ar.get("tp_rms_allreduce", 0.0)
    module_sum = sum(mod.values()) or 1.0
    scale = wall / module_sum

    lines = [
        "# Wan2.2-Distill TP=2 — 延迟开销实测分解",
        "",
        "> 由 `scripts/disagg/run_tp2_overhead_profile.py` 实测生成。",
        "> 模块 Event 原始加总可能 > 墙钟；占比用 `模块/加总×墙钟` 归一化。",
        "",
        "## 端到端墙钟（实测）",
        "",
        f"| `transformer_compute_s`（4 step） | **{wall:.3f}s** |",
        f"| 每步均值 | **{wall / 4:.3f}s** |",
        "",
        "## NCCL all-reduce（独立跑，实测，可直接占墙钟）",
        "",
        "| 分项 | 累计 (s) | 占墙钟 % |",
        "|---|---:|---:|",
        f"| row-split AR | {ar.get('tp_row_allreduce', 0.0):.3f} | {_pct(ar.get('tp_row_allreduce', 0.0), wall):.1f}% |",
        f"| RMS AR | {ar.get('tp_rms_allreduce', 0.0):.3f} | {_pct(ar.get('tp_rms_allreduce', 0.0), wall):.1f}% |",
        f"| **AR 合计** | **{comm_ar:.3f}** | **{_pct(comm_ar, wall):.1f}%** |",
        "",
        "## 模块耗时（Event 归一化到墙钟）",
        "",
        "| 模块 | 归一化 (s) | 占墙钟 % |",
        "|---|---:|---:|",
    ]
    for key, val in sorted(mod.items(), key=lambda kv: -kv[1]):
        norm = val * scale
        lines.append(f"| {key} | {norm:.3f} | {_pct(norm, wall):.1f}% |")

    lines.extend(
        [
            "",
            "## 单步 Torch Profiler（实测）",
            "",
            f"| comm / (comm+compute) | **{100 * prof['comm_cuda_ratio']:.1f}%** |",
            f"| comm CUDA (1 step) | {prof['comm_cuda_s']:.3f}s |",
            f"| compute CUDA (1 step) | {prof['compute_cuda_s']:.3f}s |",
            "",
            f"JSON: `{result.get('output_json', '')}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_json",
        default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_tp_fair_bench.json",
    )
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--tensor_p_size", type=int, default=2)
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--output_json", default="/root/zht/LightX2V/save_results/optimization_study/wan22_tp2_overhead_profile.json")
    parser.add_argument("--output_md", default="/root/zht/LightX2V/save_results/optimization_study/wan22_tp2_overhead_analysis.md")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    p1 = _load_phase1()
    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        seed=42,
        seq_p_size=1,
        tensor_p_size=args.tensor_p_size,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )

    config = p1._load_config(ns)
    seed_all(42)
    p1._init_distributed(config)

    payload = p1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=ns.prompt,
        image_path=ns.image_path,
        seed=42,
        force=False,
        task="i2v",
        negative_prompt=ns.negative_prompt,
    )
    payload = p1._prepare_payload_on_device(payload)

    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)

    for _ in range(args.warmup):
        p1._bench_transformer_once(scheduler, model, payload)

    # Pass 1: module breakdown (no MM/RMS patch)
    mod_timer = CudaTimer()
    _patch_wan_models(model, mod_timer)
    if dist.is_initialized():
        dist.barrier()
    wall = _bench_with_prepare_timer(p1, scheduler, model, payload, mod_timer)
    if dist.is_initialized():
        dist.barrier()

    # Pass 2: all-reduce only (fresh timer, class patch)
    ar_timer = CudaTimer()
    orig_mm = _patch_mm_tp(ar_timer)
    orig_rms = _patch_rms_tp(ar_timer)
    try:
        if dist.is_initialized():
            dist.barrier()
        _bench_with_prepare_timer(p1, scheduler, model, payload, CudaTimer())
        if dist.is_initialized():
            dist.barrier()

        step_profile = _profile_one_step(model, scheduler, payload, step_index=0)
    finally:
        MMWeightTP.apply = orig_mm
        RMSWeightTP.apply = orig_rms

    result = {
        "tensor_p_size": args.tensor_p_size,
        "config_json": args.config_json,
        "warmup_iters": args.warmup,
        "transformer_compute_s": wall,
        "per_step_mean_s": wall / 4,
        "module_timers_4step_s": dict(sorted(mod_timer.totals.items())),
        "module_timers_sum_s": sum(mod_timer.totals.values()),
        "allreduce_timers_4step_s": dict(sorted(ar_timer.totals.items())),
        "allreduce_timers_sum_s": sum(ar_timer.totals.values()),
        "one_step_torch_profiler": step_profile,
        "output_json": args.output_json,
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _write_markdown(Path(args.output_md), result)
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")
        print(f"wrote {args.output_md}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
