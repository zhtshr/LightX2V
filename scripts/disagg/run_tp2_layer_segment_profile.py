#!/usr/bin/env python3
"""Per-layer segment compute vs all_reduce timing for one TP=2 request.

Breaks each block into micro-phases (same split as tp_micro_overlap):
  self: prep, q_linear, q_norm, k_linear, k_norm, v_attn, o
  cross: prep, q_linear, q_norm, k_linear, k_norm, v_attn, o, ffn_post

Each segment reports wall_ms, compute_ms (wall - AR), comm_ms (AR sum), ar_count.

Run (256x256):
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_layer_segment_profile.py \\
    --config_json configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json \\
    --inputs_cache save_results/optimization_study/phase1_encoder_inputs_256x256.pt \\
    --output_json save_results/optimization_study/wan22_256_tp2_layer_segment_profile.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTP
from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all

_ctx_segment: ContextVar[str | None] = ContextVar("segment_profile_name", default=None)
_ctx_block: ContextVar[int] = ContextVar("segment_profile_block", default=-1)


@dataclass
class SegmentAccum:
    segment: str
    section: str
    total_s: float = 0.0
    comm_s: float = 0.0
    ar_count: int = 0
    samples: int = 0

    @property
    def compute_s(self) -> float:
        return max(self.total_s - self.comm_s, 0.0)


class SegmentProfiler:
    def __init__(self) -> None:
        self.by_key: dict[tuple[int, str], SegmentAccum] = {}
        self._seg_comm_s = 0.0
        self._seg_ar_count = 0
        self._in_ar = False

    def _acc(self, block_idx: int, segment: str, section: str) -> SegmentAccum:
        key = (block_idx, segment)
        if key not in self.by_key:
            self.by_key[key] = SegmentAccum(segment=segment, section=section)
        return self.by_key[key]

    def begin_segment(self, block_idx: int, segment: str, section: str) -> _SegmentTimer:
        return _SegmentTimer(self, block_idx, segment, section)

    def record_ar(self, duration_s: float) -> None:
        self._seg_comm_s += duration_s
        self._seg_ar_count += 1

    def reset_segment_ar(self) -> None:
        self._seg_comm_s = 0.0
        self._seg_ar_count = 0


class _SegmentTimer:
    def __init__(self, prof: SegmentProfiler, block_idx: int, segment: str, section: str) -> None:
        self.prof = prof
        self.block_idx = block_idx
        self.segment = segment
        self.section = section
        self._start: torch.cuda.Event | None = None
        self._tok_seg: Any = None
        self._tok_blk: Any = None

    def __enter__(self) -> _SegmentTimer:
        self.prof.reset_segment_ar()
        self._tok_seg = _ctx_segment.set(self.segment)
        self._tok_blk = _ctx_block.set(self.block_idx)
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *args: Any) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        assert self._start is not None
        self._start.synchronize()
        end.synchronize()
        total_s = self._start.elapsed_time(end) / 1000.0
        acc = self.prof._acc(self.block_idx, self.segment, self.section)
        acc.total_s += total_s
        acc.comm_s += self.prof._seg_comm_s
        acc.ar_count += self.prof._seg_ar_count
        acc.samples += 1
        if self._tok_seg is not None:
            _ctx_segment.reset(self._tok_seg)
        if self._tok_blk is not None:
            _ctx_block.reset(self._tok_blk)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_mm(prof: SegmentProfiler) -> Any:
    orig = MMWeightTP.apply

    def patched(self, input_tensor):
        prof._in_ar = False
        out = self._mm.apply(input_tensor)
        if self.split_dim == "row" and self.tp_size > 1 and self.tp_group is not None:
            prof._in_ar = True
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            s.synchronize()
            e.synchronize()
            prof.record_ar(s.elapsed_time(e) / 1000.0)
            prof._in_ar = False
            if self._row_split_bias is not None:
                out = out + self._row_split_bias
        return out

    MMWeightTP.apply = patched
    return orig


def _patch_rms(prof: SegmentProfiler) -> Any:
    orig = RMSWeightTP.apply

    def patched(self, input_tensor):
        local_sum = input_tensor.pow(2).sum(-1, keepdim=True)
        if self.tp_size > 1 and self.tp_group is not None:
            prof._in_ar = True
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.tp_group)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            s.synchronize()
            e.synchronize()
            prof.record_ar(s.elapsed_time(e) / 1000.0)
            prof._in_ar = False
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


def _run_layer_decomposed(
    model: Any,
    tenant: Any,
    prof: SegmentProfiler,
    block_idx: int,
    ov: Any,
    micro: Any,
) -> None:
    bind = ov._bind_tenant
    ensure = ov._ensure_block
    snap = ov._capture_ti_snap
    run_step = micro._run_micro_step

    self_runner = micro.TenantMicroRunner.for_self(block_idx)
    wan, ti = bind(model, tenant)
    block = ensure(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx

    for kind, name in self_runner.steps:
        with prof.begin_segment(block_idx, name, "self_attn"):
            run_step(self_runner, wan, ti, tenant, block, snap)
        self_runner.step_idx += 1

    mid = self_runner.scratch["block_mid"]
    cross_runner = micro.TenantMicroRunner.for_cross(block_idx, mid)
    wan, ti = bind(model, tenant)
    block = ensure(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx

    for kind, name in cross_runner.steps:
        with prof.begin_segment(block_idx, name, "cross_ffn"):
            run_step(cross_runner, wan, ti, tenant, block, snap)
        cross_runner.step_idx += 1


def _run_one_step_decomposed(
    model: Any,
    scheduler: WanScheduler,
    tenant: Any,
    payload: dict[str, Any],
    prof: SegmentProfiler,
    ov: Any,
    micro: Any,
    step_index: int,
) -> None:
    scheduler.prepare(
        seed=int(payload["seed"]),
        latent_shape=payload["latent_shape"],
        image_encoder_output=payload["image_encoder_output"],
    )
    scheduler.step_pre(step_index=step_index)
    ov._pre_infer_tenant(model, tenant)
    wan, ti = ov._bind_tenant(model, tenant)
    num_blocks = len(wan.transformer_weights.blocks)
    ov._preload_blocks(tenant, wan, ti, num_blocks)
    for block_idx in range(num_blocks):
        _run_layer_decomposed(model, tenant, prof, block_idx, ov, micro)
    ov._finish_step_tenant(model, tenant)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scheduler.step_post()


SEGMENT_ORDER = [
    ("self_attn", "sa_prep"),
    ("self_attn", "sa_q_linear"),
    ("self_attn", "sa_q_norm"),
    ("self_attn", "sa_k_linear"),
    ("self_attn", "sa_k_norm"),
    ("self_attn", "sa_v_attn"),
    ("self_attn", "sa_o"),
    ("cross_ffn", "cx_prep"),
    ("cross_ffn", "cx_q_linear"),
    ("cross_ffn", "cx_q_norm"),
    ("cross_ffn", "cx_k_linear"),
    ("cross_ffn", "cx_k_norm"),
    ("cross_ffn", "cx_v_attn"),
    ("cross_ffn", "cx_o"),
    ("cross_ffn", "cx_ffn_post"),
]

SEGMENT_LABELS = {
    "sa_prep": "self prep (before_proj+norm1+mod)",
    "sa_q_linear": "self Q linear",
    "sa_q_norm": "self Q norm (+AR)",
    "sa_k_linear": "self K linear",
    "sa_k_norm": "self K norm (+AR)",
    "sa_v_attn": "self V linear + attention",
    "sa_o": "self O linear (+AR)",
    "cx_prep": "cross prep (residual+norm3)",
    "cx_q_linear": "cross Q linear",
    "cx_q_norm": "cross Q norm (+AR)",
    "cx_k_linear": "cross K linear",
    "cx_k_norm": "cross K norm (+AR)",
    "cx_v_attn": "cross V + attention",
    "cx_o": "cross O linear (+AR)",
    "cx_ffn_post": "FFN (norm2+ffn0+gelu+ffn2)",
}


def _segment_row(block_idx: int, acc: SegmentAccum) -> dict[str, Any]:
    return {
        "block_idx": block_idx,
        "section": acc.section,
        "segment": acc.segment,
        "label": SEGMENT_LABELS.get(acc.segment, acc.segment),
        "total_ms": round(acc.total_s * 1000, 3),
        "compute_ms": round(acc.compute_s * 1000, 3),
        "comm_ms": round(acc.comm_s * 1000, 3),
        "ar_count": acc.ar_count,
    }


def _aggregate_segments(prof: SegmentProfiler, num_blocks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, seg in SEGMENT_ORDER:
        totals = [prof.by_key[(b, seg)].total_s for b in range(num_blocks) if (b, seg) in prof.by_key]
        comms = [prof.by_key[(b, seg)].comm_s for b in range(num_blocks) if (b, seg) in prof.by_key]
        counts = [prof.by_key[(b, seg)].ar_count for b in range(num_blocks) if (b, seg) in prof.by_key]
        if not totals:
            continue
        total_ms = statistics.mean(totals) * 1000
        comm_ms = statistics.mean(comms) * 1000
        rows.append({
            "section": section,
            "segment": seg,
            "label": SEGMENT_LABELS.get(seg, seg),
            "total_ms_mean": round(total_ms, 3),
            "compute_ms_mean": round(total_ms - comm_ms, 3),
            "comm_ms_mean": round(comm_ms, 3),
            "ar_count_mean": round(statistics.mean(counts), 2),
            "total_ms_sum_40L": round(sum(totals) * 1000, 3),
            "comm_ms_sum_40L": round(sum(comms) * 1000, 3),
        })
    return rows


def _layer_totals(prof: SegmentProfiler, block_idx: int) -> dict[str, float]:
    segs = [acc for (b, _), acc in prof.by_key.items() if b == block_idx]
    total = sum(s.total_s for s in segs) * 1000
    comm = sum(s.comm_s for s in segs) * 1000
    self_ms = sum(s.total_s for s in segs if s.section == "self_attn") * 1000
    cross_ms = sum(s.total_s for s in segs if s.section == "cross_ffn") * 1000
    return {
        "block_total_ms": round(total, 3),
        "compute_ms": round(total - comm, 3),
        "comm_ms": round(comm, 3),
        "self_attn_ms": round(self_ms, 3),
        "cross_ffn_ms": round(cross_ms, 3),
    }


def _write_md(path: Path, result: dict[str, Any]) -> None:
    agg = result["segment_summary_mean_per_layer"]
    layer0 = result["layer0_segments"]
    feas = result["layer_summary"]
    lines = [
        "# Wan TP=2 — 单层分段 Compute / Comm 实测",
        "",
        f"Config: `{result['config_json']}` | 1 request | 1 denoise step (idx={result['step_index']})",
        f"| wall **{result['one_step_wall_s']:.3f}s**",
        "",
        "## 单层汇总（40 层均值）",
        "",
        f"- block total: **{feas['block_total_ms_mean']:.2f} ms**",
        f"- self_attn: **{feas['self_attn_ms_mean']:.2f} ms** (comm {feas['self_attn_comm_ms_mean']:.2f} ms)",
        f"- cross_ffn: **{feas['cross_ffn_ms_mean']:.2f} ms** (comm {feas['cross_ffn_comm_ms_mean']:.2f} ms)",
        f"- comm 占比: **{feas['comm_pct_mean']:.1f}%**",
        "",
        "## 分段均值（每层）",
        "",
        "| section | segment | compute (ms) | comm (ms) | total (ms) | #AR | 40L comm sum |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        lines.append(
            f"| {r['section']} | {r['label']} | {r['compute_ms_mean']:.3f} | "
            f"{r['comm_ms_mean']:.3f} | {r['total_ms_mean']:.3f} | {r['ar_count_mean']:.1f} | "
            f"{r['comm_ms_sum_40L']:.1f} |"
        )
    lines.extend([
        "",
        "## Layer 0 明细（ms）",
        "",
        "| segment | compute | comm | total | #AR |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in layer0:
        lines.append(
            f"| {r['label']} | {r['compute_ms']:.3f} | {r['comm_ms']:.3f} | "
            f"{r['total_ms']:.3f} | {r['ar_count']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--tensor_p_size", type=int, default=2)
    parser.add_argument("--inputs_cache", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--step_index", type=int, default=0)
    args = parser.parse_args()

    here = Path(__file__).parent
    p1 = _load_module("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load_module("overlap", here / "run_phase3_dual_overlap_bench.py")
    micro = _load_module("micro", here / "tp_micro_overlap.py")

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
    tenant = ov.TenantCtx("profile", scheduler, payload["inputs"])

    for _ in range(args.warmup):
        ov._run_denoise_serial(model, scheduler, payload)

    prof = SegmentProfiler()
    orig_mm = _patch_mm(prof)
    orig_rms = _patch_rms(prof)

    try:
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_one_step_decomposed(
            model, scheduler, tenant, payload, prof, ov, micro, args.step_index,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_wall = time.perf_counter() - t0
        if dist.is_initialized():
            dist.barrier()
    finally:
        MMWeightTP.apply = orig_mm
        RMSWeightTP.apply = orig_rms

    wan, _ = ov._bind_tenant(model, tenant)
    num_blocks = len(wan.transformer_weights.blocks)
    agg = _aggregate_segments(prof, num_blocks)
    per_layer = [_layer_totals(prof, b) for b in range(num_blocks)]
    layer0 = [
        _segment_row(0, prof.by_key[(0, seg)])
        for _, seg in SEGMENT_ORDER
        if (0, seg) in prof.by_key
    ]

    block_totals = [p["block_total_ms"] for p in per_layer]
    self_totals = [p["self_attn_ms"] for p in per_layer]
    cross_totals = [p["cross_ffn_ms"] for p in per_layer]
    comm_totals = [p["comm_ms"] for p in per_layer]
    self_comm = []
    cross_comm = []
    for b in range(num_blocks):
        sc = sum(
            prof.by_key[(b, s)].comm_s
            for sec, s in SEGMENT_ORDER
            if sec == "self_attn" and (b, s) in prof.by_key
        )
        cc = sum(
            prof.by_key[(b, s)].comm_s
            for sec, s in SEGMENT_ORDER
            if sec == "cross_ffn" and (b, s) in prof.by_key
        )
        self_comm.append(sc * 1000)
        cross_comm.append(cc * 1000)

    layer_summary = {
        "block_total_ms_mean": round(statistics.mean(block_totals), 3),
        "self_attn_ms_mean": round(statistics.mean(self_totals), 3),
        "cross_ffn_ms_mean": round(statistics.mean(cross_totals), 3),
        "comm_ms_mean": round(statistics.mean(comm_totals), 3),
        "self_attn_comm_ms_mean": round(statistics.mean(self_comm), 3),
        "cross_ffn_comm_ms_mean": round(statistics.mean(cross_comm), 3),
        "comm_pct_mean": round(100 * statistics.mean(comm_totals) / statistics.mean(block_totals), 1),
    }

    result = {
        "config_json": args.config_json,
        "tensor_p_size": args.tensor_p_size,
        "inputs_cache": args.inputs_cache,
        "step_index": args.step_index,
        "one_step_wall_s": step_wall,
        "num_layers": num_blocks,
        "layer_summary": layer_summary,
        "segment_summary_mean_per_layer": agg,
        "layer0_segments": layer0,
        "per_layer_totals": per_layer,
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        md = out.with_suffix(".md")
        _write_md(md, result)
        print(json.dumps({"layer_summary": layer_summary, "segments": agg}, indent=2))
        print(f"wrote {out}")
        print(f"wrote {md}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
