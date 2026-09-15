#!/usr/bin/env python3
"""Per-layer TP compute vs all_reduce profiling for dual-overlap slice feasibility.

Measures per block (40 layers):
  - self_attn / cross_ffn wall time (CUDA events)
  - all_reduce count & cumulative ms (MMWeightTP + RMSWeightTP hook)
  - compute gaps between consecutive AR calls (overlap-able segments)

Run (256x256 example):
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_layer_comm_slice_profile.py \\
    --config_json configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json \\
    --inputs_cache save_results/optimization_study/phase1_encoder_inputs_256x256.pt \\
    --output_json save_results/optimization_study/wan22_256_tp2_layer_slice_profile.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTP
from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all

_ctx_phase: ContextVar[str] = ContextVar("tp_profile_phase", default="other")


@dataclass
class BlockStats:
    block_idx: int
    self_attn_s: float = 0.0
    cross_ffn_s: float = 0.0
    block_total_s: float = 0.0
    ar_s: float = 0.0
    ar_count: int = 0
    compute_gap_s: list[float] = field(default_factory=list)
    ar_s_by_phase: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    ar_count_by_phase: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class LayerSliceProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.blocks: dict[int, BlockStats] = {}
        self._cur_block = 0
        self._last_marker: torch.cuda.Event | None = None
        self._in_ar = False
        self.global_ar_s = 0.0
        self.global_ar_count = 0
        self.global_compute_gap_s: list[float] = []

    def _block(self, idx: int) -> BlockStats:
        if idx not in self.blocks:
            self.blocks[idx] = BlockStats(block_idx=idx)
        return self.blocks[idx]

    def set_block(self, idx: int) -> None:
        self._cur_block = idx
        self._sync_marker()

    def _sync_marker(self) -> None:
        if not torch.cuda.is_available():
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        e.synchronize()
        self._last_marker = e

    def _elapsed_since_marker_ms(self) -> float:
        if self._last_marker is None:
            return 0.0
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._last_marker.synchronize()
        end.synchronize()
        return self._last_marker.elapsed_time(end)

    def record_compute_gap(self) -> None:
        if self._in_ar:
            return
        gap_ms = self._elapsed_since_marker_ms()
        if gap_ms <= 0:
            self._sync_marker()
            return
        gap_s = gap_ms / 1000.0
        blk = self._block(self._cur_block)
        blk.compute_gap_s.append(gap_s)
        self.global_compute_gap_s.append(gap_s)
        self._sync_marker()

    def record_ar(self, duration_s: float) -> None:
        phase = _ctx_phase.get()
        blk = self._block(self._cur_block)
        blk.ar_s += duration_s
        blk.ar_count += 1
        blk.ar_s_by_phase[phase] += duration_s
        blk.ar_count_by_phase[phase] += 1
        self.global_ar_s += duration_s
        self.global_ar_count += 1
        self._sync_marker()

    def time_region(self, block_idx: int, key: str) -> Any:
        """Context manager via helper object."""
        return _CudaRegionTimer(self, block_idx, key)


class _CudaRegionTimer:
    def __init__(self, prof: LayerSliceProfiler, block_idx: int, key: str) -> None:
        self.prof = prof
        self.block_idx = block_idx
        self.key = key
        self._start: torch.cuda.Event | None = None

    def __enter__(self) -> _CudaRegionTimer:
        self.prof.set_block(self.block_idx)
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *args: Any) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        assert self._start is not None
        self._start.synchronize()
        end.synchronize()
        dt = self._start.elapsed_time(end) / 1000.0
        blk = self.prof._block(self.block_idx)
        if self.key == "self_attn":
            blk.self_attn_s += dt
        elif self.key == "cross_ffn":
            blk.cross_ffn_s += dt
        elif self.key == "block":
            blk.block_total_s += dt
        self.prof._sync_marker()


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_mm_tp(prof: LayerSliceProfiler) -> Any:
    orig = MMWeightTP.apply

    def patched(self, input_tensor):
        prof.record_compute_gap()
        prof._in_ar = False
        out = self._mm.apply(input_tensor)
        if self.split_dim == "row" and self.tp_size > 1 and self.tp_group is not None:
            prof._in_ar = True
            if torch.cuda.is_available():
                s = torch.cuda.Event(enable_timing=True)
                s.record()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            if torch.cuda.is_available():
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                s.synchronize()
                e.synchronize()
                prof.record_ar(s.elapsed_time(e) / 1000.0)
            else:
                prof.record_ar(0.0)
            prof._in_ar = False
            if self._row_split_bias is not None:
                out = out + self._row_split_bias
        else:
            prof._sync_marker()
        return out

    MMWeightTP.apply = patched
    return orig


def _patch_rms_tp(prof: LayerSliceProfiler) -> Any:
    orig = RMSWeightTP.apply

    def patched(self, input_tensor):
        local_sum = input_tensor.pow(2).sum(-1, keepdim=True)
        if self.tp_size > 1 and self.tp_group is not None:
            prof.record_compute_gap()
            prof._in_ar = True
            if torch.cuda.is_available():
                s = torch.cuda.Event(enable_timing=True)
                s.record()
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.tp_group)
            if torch.cuda.is_available():
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                s.synchronize()
                e.synchronize()
                prof.record_ar(s.elapsed_time(e) / 1000.0)
            else:
                prof.record_ar(0.0)
            prof._in_ar = False
        hidden_dim = input_tensor.shape[-1] * self.tp_size
        global_mean = local_sum / hidden_dim
        if self.sensitive_layer_dtype != self.infer_dtype:
            input_tensor = input_tensor * torch.rsqrt(global_mean.float() + self.eps).to(self.infer_dtype)
            input_tensor = (input_tensor * self._get_actual_weight()).to(self.infer_dtype)
        else:
            input_tensor = input_tensor * torch.rsqrt(global_mean + self.eps)
            input_tensor = input_tensor * self._get_actual_weight()
        prof._sync_marker()
        return input_tensor

    RMSWeightTP.apply = patched
    return orig


def _patch_transformer_infer(ti: Any, prof: LayerSliceProfiler) -> None:
    orig_block = ti.infer_block
    orig_self = ti.infer_self_attn
    orig_cross = ti.infer_cross_attn
    orig_ffn = ti.infer_ffn

    def infer_block(block, x, pre_infer_out):
        idx = ti.block_idx
        with prof.time_region(idx, "block"):
            token_self = _ctx_phase.set("self_attn")
            try:
                with prof.time_region(idx, "self_attn"):
                    if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
                        x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x
                    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ti.pre_process(
                        block.compute_phases[0].modulation, pre_infer_out.embed0,
                    )
                    y_out = orig_self(block.compute_phases[0], x, shift_msa, scale_msa)
            finally:
                _ctx_phase.reset(token_self)

            token_cross = _ctx_phase.set("cross_ffn")
            try:
                with prof.time_region(idx, "cross_ffn"):
                    x, attn_out = orig_cross(block.compute_phases[1], x, pre_infer_out.context, y_out, gate_msa)
                    y = orig_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
                    x = ti.post_process(x, y, c_gate_msa, pre_infer_out)
                    if hasattr(block.compute_phases[2], "after_proj"):
                        pre_infer_out.adapter_args["hints"].append(block.compute_phases[2].after_proj.apply(x))
                    if ti.has_post_adapter:
                        x = ti.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)
            finally:
                _ctx_phase.reset(token_cross)
        return x

    ti.infer_block = MethodType(lambda _, *a, **k: infer_block(*a, **k), ti)


def _patch_wan_models(model: Any, prof: LayerSliceProfiler) -> None:
    subs = model.model if hasattr(model, "model") and isinstance(model.model, list) else [model]
    for sub in subs:
        if sub is not None and hasattr(sub, "transformer_infer"):
            _patch_transformer_infer(sub.transformer_infer, prof)


def _run_one_step(model: Any, scheduler: WanScheduler, payload: dict[str, Any], step_index: int = 0) -> None:
    scheduler.prepare(
        seed=int(payload["seed"]),
        latent_shape=payload["latent_shape"],
        image_encoder_output=payload["image_encoder_output"],
    )
    scheduler.step_pre(step_index=step_index)
    model.infer(payload["inputs"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scheduler.step_post()


def _summarize_block(b: BlockStats) -> dict[str, Any]:
    gaps = b.compute_gap_s
    gap_sum = sum(gaps)
    block_compute_est = max(b.block_total_s - b.ar_s, 0.0)
    return {
        "block_idx": b.block_idx,
        "block_total_ms": round(b.block_total_s * 1000, 3),
        "self_attn_ms": round(b.self_attn_s * 1000, 3),
        "cross_ffn_ms": round(b.cross_ffn_s * 1000, 3),
        "ar_ms": round(b.ar_s * 1000, 3),
        "ar_count": b.ar_count,
        "ar_pct": round(100 * b.ar_s / b.block_total_s, 1) if b.block_total_s > 0 else 0.0,
        "compute_gap_sum_ms": round(gap_sum * 1000, 3),
        "compute_gap_count": len(gaps),
        "compute_gap_mean_ms": round(statistics.mean(gaps) * 1000, 3) if gaps else 0.0,
        "compute_gap_max_ms": round(max(gaps) * 1000, 3) if gaps else 0.0,
        "compute_gap_p50_ms": round(statistics.median(gaps) * 1000, 3) if gaps else 0.0,
        "ar_ms_self_attn": round(b.ar_s_by_phase.get("self_attn", 0) * 1000, 3),
        "ar_ms_cross_ffn": round(b.ar_s_by_phase.get("cross_ffn", 0) * 1000, 3),
        "ar_count_self_attn": b.ar_count_by_phase.get("self_attn", 0),
        "ar_count_cross_ffn": b.ar_count_by_phase.get("cross_ffn", 0),
        "block_compute_est_ms": round(block_compute_est * 1000, 3),
    }


def _overlap_feasibility(rows: list[dict[str, Any]], step_wall_s: float) -> dict[str, Any]:
    total_ar = sum(r["ar_ms"] for r in rows) / 1000.0
    total_block = sum(r["block_total_ms"] for r in rows) / 1000.0
    all_gaps = []
    for r in rows:
        # approximate: use mean * count
        if r["compute_gap_count"]:
            all_gaps.extend([r["compute_gap_mean_ms"] / 1000.0] * r["compute_gap_count"])

    # Per-layer: can one cross_ffn (~cross_ffn_ms) cover one self_attn AR burst?
    # SP-style: overlap uses full cross_ffn during self_attn AR windows
    cross_ffn_ms = [r["cross_ffn_ms"] for r in rows]
    self_ar_ms = [r["ar_ms_self_attn"] for r in rows]
    layers_cross_covers_self_ar = sum(
        1 for c, a in zip(cross_ffn_ms, self_ar_ms) if c >= a and a > 0
    )

    # Dual-tenant: tenant B gap while tenant A AR — need gap >= typical AR
    gap_max_ms = max((r["compute_gap_max_ms"] for r in rows), default=0.0)
    ar_mean_ms = (total_ar * 1000 / max(sum(r["ar_count"] for r in rows), 1))

    # Theoretical overlap per layer if we fill every AR with max preceding gap compute
    overlap_upper_per_step_s = sum(r["compute_gap_sum_ms"] for r in rows) / 1000.0
    # Realistic: one cross_ffn per layer during self_attn (OnceCompute)
    once_per_layer_s = sum(min(c, a) / 1000.0 for c, a in zip(cross_ffn_ms, self_ar_ms))

    return {
        "one_step_wall_s": step_wall_s,
        "per_layer_total_ms_mean": round(statistics.mean(r["block_total_ms"] for r in rows), 3),
        "per_layer_ar_ms_mean": round(statistics.mean(r["ar_ms"] for r in rows), 3),
        "per_layer_ar_count_mean": round(statistics.mean(r["ar_count"] for r in rows), 1),
        "step_ar_ms_total": round(total_ar * 1000, 3),
        "step_ar_pct_of_blocks": round(100 * total_ar / total_block, 1) if total_block else 0,
        "compute_gap_global_mean_ms": round(statistics.mean(all_gaps) * 1000, 3) if all_gaps else 0,
        "compute_gap_global_max_ms": round(gap_max_ms, 3),
        "ar_mean_per_call_ms": round(ar_mean_ms, 3),
        "layers_cross_ffn_covers_self_ar_count": layers_cross_covers_self_ar,
        "layers_total": len(rows),
        "sp_style_once_overlap_save_ms_per_step": round(once_per_layer_s * 1000, 3),
        "sp_style_once_overlap_pct_of_step": round(100 * once_per_layer_s / step_wall_s, 1) if step_wall_s else 0,
        "theoretical_max_gap_fill_ms_per_step": round(overlap_upper_per_step_s * 1000, 3),
        "theoretical_max_gap_fill_pct_of_step": round(100 * overlap_upper_per_step_s / step_wall_s, 1) if step_wall_s else 0,
    }


def _write_md(out_md: Path, result: dict[str, Any]) -> None:
    feas = result["feasibility"]
    rows = result["per_layer"]
    lines = [
        "# Wan TP=2 — 每层 Compute / AR 切片分析",
        "",
        f"Config: `{result['config_json']}` | 1 denoise step | wall **{feas['one_step_wall_s']:.3f}s**",
        "",
        "## 汇总",
        "",
        f"- 每层 AR 均值: **{feas['per_layer_ar_ms_mean']:.2f} ms** ({feas['per_layer_ar_count_mean']:.1f} 次/层)",
        f"- AR 占 block 时间: **{feas['step_ar_pct_of_blocks']:.1f}%**",
        f"- AR 单次均值: **{feas['ar_mean_per_call_ms']:.3f} ms**",
        f"- 无 AR 计算间隙: mean **{feas['compute_gap_global_mean_ms']:.3f} ms**, max **{feas['compute_gap_global_max_ms']:.3f} ms**",
        f"- SP 风格（每层叠一次 cross_ffn）可省: **{feas['sp_style_once_overlap_save_ms_per_step']:.1f} ms** ({feas['sp_style_once_overlap_pct_of_step']:.1f}%/step)",
        f"- 理论间隙全填满上界: **{feas['theoretical_max_gap_fill_ms_per_step']:.1f} ms** ({feas['theoretical_max_gap_fill_pct_of_step']:.1f}%/step)",
        f"- cross_ffn 时长 ≥ self_attn AR 的层数: **{feas['layers_cross_ffn_covers_self_ar_count']}/{feas['layers_total']}**",
        "",
        "## 逐层（ms）",
        "",
        "| L | block | self_attn | cross_ffn | AR | AR% | #AR | gap_mean | gap_max | AR_self | AR_cross |",
        "|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['block_idx']} | {r['block_total_ms']:.1f} | {r['self_attn_ms']:.1f} | "
            f"{r['cross_ffn_ms']:.1f} | {r['ar_ms']:.1f} | {r['ar_pct']:.0f}% | {r['ar_count']} | "
            f"{r['compute_gap_mean_ms']:.2f} | {r['compute_gap_max_ms']:.2f} | "
            f"{r['ar_ms_self_attn']:.1f} | {r['ar_ms_cross_ffn']:.1f} |"
        )
    lines.extend(["", "## 切片 overlap 判断", "", result["analysis_text"], ""])
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _analysis_text(feas: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    once_pct = feas["sp_style_once_overlap_pct_of_step"]
    max_pct = feas["theoretical_max_gap_fill_pct_of_step"]
    ar_cross = sum(r["ar_ms_cross_ffn"] for r in rows)
    ar_self = sum(r["ar_ms_self_attn"] for r in rows)
    lines = [
        f"1. **AR 分布**：self_attn 段 AR {ar_self:.1f} ms，cross_ffn 段 AR {ar_cross:.1f} ms；"
        f"cross_ffn 并非无通信，SP 假设不成立。",
        f"2. **间隙尺度**：平均计算间隙 {feas['compute_gap_global_mean_ms']:.2f} ms vs AR 单次 {feas['ar_mean_per_call_ms']:.2f} ms；"
        f"最大间隙 {feas['compute_gap_global_max_ms']:.2f} ms。",
        f"3. **SP 风格 OnceCompute**：每层用 cross_ffn 盖 self_attn AR，上界约 **{once_pct:.1f}%**/step；"
        f"与 dual overlap 实测（~0% 增益）一致。",
        f"4. **细粒度切片上界**：若每个 AR 前间隙都能叠满另一 tenant 计算，约 **{max_pct:.1f}%**/step；"
        "需要 per-AR 调度 + 无 collective 切片，工程复杂。",
    ]
    if max_pct < 30:
        lines.append("5. **结论**：切片 overlap **理论收益有限**（<30%/step），且需重写 infer 为 AR 感知状态机。")
    elif once_pct < 15:
        lines.append(
            "5. **结论**：粗粒度（每层一次）overlap 收益 <15%；细粒度切片值得原型，"
            f"上界 ~{max_pct:.0f}%，但 cross_ffn 内 AR 仍限制双 tenant 互叠。"
        )
    else:
        lines.append(f"5. **结论**：细粒度切片有 **~{max_pct:.0f}%**/step 潜力，建议实现 per-AR 进度调度原型。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--tensor_p_size", type=int, default=2)
    parser.add_argument("--inputs_cache", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--step_index", type=int, default=0)
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

    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    prof = LayerSliceProfiler(device)

    for _ in range(args.warmup):
        _run_one_step(model, scheduler, payload, step_index=args.step_index)
        prof = LayerSliceProfiler(device)

    orig_mm = _patch_mm_tp(prof)
    orig_rms = _patch_rms_tp(prof)
    _patch_wan_models(model, prof)
    prof._sync_marker()

    try:
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_one_step(model, scheduler, payload, step_index=args.step_index)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_wall = time.perf_counter() - t0
        if dist.is_initialized():
            dist.barrier()
    finally:
        MMWeightTP.apply = orig_mm
        RMSWeightTP.apply = orig_rms

    rows = [_summarize_block(prof.blocks[i]) for i in sorted(prof.blocks)]
    feas = _overlap_feasibility(rows, step_wall)
    analysis = _analysis_text(feas, rows)

    result = {
        "config_json": args.config_json,
        "tensor_p_size": args.tensor_p_size,
        "inputs_cache": args.inputs_cache,
        "step_index": args.step_index,
        "one_step_wall_s": step_wall,
        "per_layer": rows,
        "feasibility": feas,
        "analysis_text": analysis,
        "global": {
            "ar_count": prof.global_ar_count,
            "ar_s": prof.global_ar_s,
            "compute_gap_count": len(prof.global_compute_gap_s),
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        md_path = Path(args.output_md) if args.output_md else out.with_suffix(".md")
        _write_md(md_path, result)
        print(json.dumps({"feasibility": feas, "analysis": analysis}, indent=2))
        print(f"wrote {out}")
        print(f"wrote {md_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
