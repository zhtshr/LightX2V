#!/usr/bin/env python3
"""Parse nsys sqlite: kernel breakdown per NVTX range + overlap concurrency."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path


SQLITE = Path(os.environ.get("NSYS_SQLITE", "save_results/nsys/tp2_overlap.sqlite"))
OUT_JSON = Path(os.environ.get("NSYS_ANALYSIS_JSON", "save_results/nsys/tp2_overlap_analysis.json"))
OUT_MD = os.environ.get("NSYS_ANALYSIS_MD")  # 默认不写 md；见 tp2_overlap_root_cause.md
WALL_JSON = Path(os.environ.get("NSYS_WALL_JSON", "save_results/nsys/tp2_overlap_wall.json"))

RANGES = ("COMM_alone_p2", "COMP_alone_p1", "OVERLAP_comm_p2_comp_p1")


def _name(conn: sqlite3.Connection, sid: int) -> str:
    row = conn.execute("SELECT value FROM StringIds WHERE id=?", (sid,)).fetchone()
    return row[0] if row else f"id{sid}"


def _classify(name: str) -> str:
    n = name.lower()
    if "nccl" in n or "allreduce" in n:
        return "nccl"
    if "_attn_fwd" in n or "flashinfer" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n:
        return "gemm"
    if "norm" in n or "layer_norm" in n:
        return "norm"
    if "quantize" in n or "quant" in n:
        return "quant"
    if "memcpy" in n or "memset" in n:
        return "mem"
    return "other"


def _range_events(conn: sqlite3.Connection) -> dict[str, list[tuple[int, int, int]]]:
    """label -> [(start, end, deviceId)]; use min/max per label across threads."""
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS WHERE text IS NOT NULL",
    ):
        if text in RANGES:
            raw[text].append((start, end))
    out: dict[str, list[tuple[int, int, int]]] = {}
    for label, spans in raw.items():
        # merge to envelope per capture (2 ranks → 2 spans; keep both)
        out[label] = [(s, e, -1) for s, e in spans]
    return out


def _kernels_in_range(
    conn: sqlite3.Connection,
    t0: int,
    t1: int,
    device_id: int | None = 0,
) -> list[dict]:
    q = """
        SELECT k.start, k.end, k.deviceId, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE k.end > ? AND k.start < ?
    """
    params: list = [t0, t1]
    if device_id is not None:
        q += " AND k.deviceId = ?"
        params.append(device_id)
    q += " ORDER BY k.start"
    rows = conn.execute(q, params).fetchall()
    out = []
    for start, end, dev, stream, name in rows:
        out.append({
            "start": start,
            "end": end,
            "dur_ns": end - start,
            "deviceId": dev,
            "streamId": stream,
            "name": name,
            "cls": _classify(name),
        })
    return out


def _summarize(kernels: list[dict]) -> dict:
    by_cls: dict[str, int] = defaultdict(int)
    by_name: dict[str, int] = defaultdict(int)
    for k in kernels:
        by_cls[k["cls"]] += k["dur_ns"]
        short = k["name"].split("(")[0].split("<")[0]
        if len(short) > 80:
            short = short[:77] + "..."
        by_name[short] += k["dur_ns"]
    wall_ns = max((k["end"] for k in kernels), default=0) - min((k["start"] for k in kernels), default=0)
    if not kernels:
        wall_ns = 0
    return {
        "kernel_count": len(kernels),
        "gpu_busy_sum_ns": sum(k["dur_ns"] for k in kernels),
        "span_ns": wall_ns,
        "by_class_ns": dict(sorted(by_cls.items(), key=lambda x: -x[1])),
        "top_kernels_ns": dict(sorted(by_name.items(), key=lambda x: -x[1])[:12]),
    }


def _concurrency(kernels: list[dict]) -> dict:
    """Fraction of overlap window where NCCL and compute kernels co-run."""
    nccl = [k for k in kernels if k["cls"] == "nccl"]
    comp = [k for k in kernels if k["cls"] in ("attention", "gemm", "norm", "quant", "other")]
    if not nccl or not comp:
        return {"nccl_comp_overlap_ns": 0, "overlap_frac_of_span": 0.0}

    def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    nccl_m = _merge([(k["start"], k["end"]) for k in nccl])
    comp_m = _merge([(k["start"], k["end"]) for k in comp])
    overlap_ns = 0
    for ns, ne in nccl_m:
        for cs, ce in comp_m:
            s = max(ns, cs)
            e = min(ne, ce)
            if e > s:
                overlap_ns += e - s
    span = max(k["end"] for k in kernels) - min(k["start"] for k in kernels)
    return {
        "nccl_comp_overlap_ns": overlap_ns,
        "overlap_frac_of_span": overlap_ns / span if span else 0.0,
        "nccl_span_ns": sum(e - s for s, e in nccl_m),
        "comp_span_ns": sum(e - s for s, e in comp_m),
    }


def main() -> None:
    conn = sqlite3.connect(SQLITE)
    ranges = _range_events(conn)

    per_range: dict = {}
    for label in RANGES:
        spans = ranges.get(label, [])
        if not spans:
            continue
        # analyze GPU0 using envelope of first span (ranks symmetric)
        t0 = min(s for s, _, _ in spans)
        t1 = max(e for _, e, _ in spans)
        kernels = _kernels_in_range(conn, t0, t1, device_id=0)
        per_range[label] = {
            "nvtx_span_ms": (t1 - t0) / 1e6,
            "summary": _summarize(kernels),
            "concurrency": _concurrency(kernels),
        }
        per_range[label]["summary_ms"] = {
            k: v / 1e6 for k, v in per_range[label]["summary"]["by_class_ns"].items()
        }

    comp = per_range["COMP_alone_p1"]["summary"]["by_class_ns"]
    ov = per_range["OVERLAP_comm_p2_comp_p1"]["summary"]["by_class_ns"]
    slowdown = {}
    for cls in set(comp) | set(ov):
        if cls == "nccl":
            continue
        c_alone = comp.get(cls, 0) / 1e6
        c_ov = ov.get(cls, 0) / 1e6
        if c_alone > 0.1:
            slowdown[cls] = {"alone_ms": c_alone, "overlap_ms": c_ov, "ratio": c_ov / c_alone}

    wall = json.loads(WALL_JSON.read_text())
    analysis = {
        "wall_times_rank0": wall,
        "tp_norm_p2p": wall.get("tp_norm_p2p", False),
        "per_nvtx_range_gpu0": per_range,
        "compute_class_slowdown_in_overlap": slowdown,
    }
    OUT_JSON.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    conc = per_range["OVERLAP_comm_p2_comp_p1"]["concurrency"]
    comp_sum = per_range["COMP_alone_p1"]["summary"]
    ov_sum = per_range["OVERLAP_comm_p2_comp_p1"]["summary"]
    lines = [
        "# Nsight: COMP 变慢原因分析 (A2B1, layer 10, GPU0)",
        "",
        f"tp_norm_p2p: **{wall.get('tp_norm_p2p', False)}**",
        "",
        "## Wall 时间 (rank0)",
        "",
        f"| 场景 | wall ms |",
        "|---|---:|",
        f"| COMM alone | {wall['comm_alone_ms']:.2f} |",
        f"| COMP alone | {wall['comp_alone_ms']:.2f} |",
        f"| serial sum | {wall['serial_sum_ms']:.2f} |",
        f"| ideal max | {wall['ideal_max_ms']:.2f} |",
        f"| OVERLAP | {wall['overlap_ms']:.2f} |",
        "",
        "## GPU kernel 时间按类别 (GPU0, kernel duration 累加)",
        "",
        "| 类别 | COMP alone (ms) | OVERLAP (ms) | 倍率 |",
        "|---|---:|---:|---:|",
    ]
    for cls, d in sorted(slowdown.items(), key=lambda x: -x[1]["alone_ms"]):
        lines.append(f"| {cls} | {d['alone_ms']:.2f} | {d['overlap_ms']:.2f} | {d['ratio']:.2f}x |")
    lines.extend([
        "",
        f"- COMP alone: kernel 数 {comp_sum['kernel_count']}, GPU busy 累加 {comp_sum['gpu_busy_sum_ns']/1e6:.1f} ms",
        f"- OVERLAP: kernel 数 {ov_sum['kernel_count']}, GPU busy 累加 {ov_sum['gpu_busy_sum_ns']/1e6:.1f} ms",
        f"- OVERLAP 中 NCCL kernel 累加 {ov_sum['by_class_ns'].get('nccl',0)/1e6:.1f} ms",
        "",
        "## NCCL 与 COMP kernel 时间重叠 (GPU0)",
        "",
        f"- overlap 窗口 span: {per_range['OVERLAP_comm_p2_comp_p1']['nvtx_span_ms']:.1f} ms",
        f"- NCCL 与 compute kernel **同时运行**的时间: {conc['nccl_comp_overlap_ns']/1e6:.1f} ms",
        f"- 占窗口比例: **{100*conc['overlap_frac_of_span']:.1f}%**",
        f"- NCCL kernel 累加 span: {conc['nccl_span_ns']/1e6:.1f} ms",
        f"- compute kernel 累加 span: {conc['comp_span_ns']/1e6:.1f} ms",
        "",
        "## 结论",
        "",
    ])

    ov_frac = conc["overlap_frac_of_span"]
    attn_ratio = slowdown.get("attention", {}).get("ratio", 0)
    gemm_ratio = slowdown.get("gemm", {}).get("ratio", 0)
    if ov_frac > 0.5:
        lines.append(
            f"1. **NCCL 与 COMP 在 GPU 时间线上大量重叠** ({100*ov_frac:.0f}%)，说明 schedule 层面已在并行。"
        )
    else:
        lines.append(
            f"1. **NCCL 与 COMP 在 GPU 时间线上重叠很少** ({100*ov_frac:.0f}%)，存在明显串行。"
        )
    if attn_ratio > 1.3 or gemm_ratio > 1.3:
        lines.append(
            f"2. **并行时 attention {attn_ratio:.2f}x / gemm {gemm_ratio:.2f}x 变慢**：同卡 NCCL 与计算 kernel 争抢 SM/DRAM 带宽，"
            "单 kernel 变长；累加 busy 时间接近 serial sum，墙钟省不下来。"
        )
    lines.append(
        f"3. OVERLAP wall ({wall['overlap_ms']:.1f} ms) ≈ max(COMM, COMP_slowed) 而非 ideal ({wall['ideal_max_ms']:.1f} ms)。"
    )
    print(json.dumps(analysis, indent=2)[:4000])
    print(f"wrote {OUT_JSON}")
    if OUT_MD:
        Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {OUT_MD}")
    print("结论文档: save_results/nsys/tp2_overlap_root_cause.md")


if __name__ == "__main__":
    main()
