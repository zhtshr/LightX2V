#!/usr/bin/env python3
"""Deep Nsight sqlite analysis: kernel overlap, sync hotspots, timeline, backtraces."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

SQLITE = Path(os.environ.get("NSYS_SQLITE", "save_results/nsys/tp2_serial_fix_a.sqlite"))
WALL_JSON = Path(os.environ.get("NSYS_WALL_JSON", "save_results/nsys/tp2_serial_fix_a_wall.json"))
OUT_JSON = Path(os.environ.get("OUT_JSON", "save_results/nsys/tp2_serial_fix_a_v2_analysis.json"))
OUT_MD = os.environ.get("OUT_MD")  # 默认不写 md；结论见 save_results/nsys/tp2_overlap_root_cause.md

RANGES = ("COMM_alone_p2", "COMP_alone_p1", "OVERLAP_comm_p2_comp_p1")
COMP_CLS = frozenset({"attention", "gemm", "norm", "quant", "other"})
SYNC_RT = (
    "cudaDeviceSynchronize_v3020",
    "cudaStreamSynchronize_v3020",
    "cudaStreamWaitEvent_v3020",
    "cudaEventSynchronize_v3020",
)
SYNC_TYPE = {0: "unknown", 1: "stream_wait", 2: "event_wait", 3: "stream_block"}


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


def _nvtx_span(conn: sqlite3.Connection, label: str) -> tuple[int, int]:
    rows = conn.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text=? ORDER BY start", (label,),
    ).fetchall()
    if not rows:
        return 0, 0
    return min(r[0] for r in rows), max(r[1] for r in rows)


def _kernels(conn: sqlite3.Connection, t0: int, t1: int, device: int = 0) -> list[dict]:
    rows = conn.execute(
        """
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE k.deviceId=? AND k.end>? AND k.start<?
        ORDER BY k.start
        """,
        (device, t0, t1),
    ).fetchall()
    return [
        {
            "start": start, "end": end, "dur_ns": end - start,
            "streamId": stream, "name": name, "cls": _classify(name),
            "short": name.split("(")[0][:70],
        }
        for start, end, stream, name in rows
    ]


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for s, e in sorted(intervals[1:]):
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _overlap_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    for as_, ae in a:
        for bs, be in b:
            s, e = max(as_, bs), min(ae, be)
            if e > s:
                total += e - s
    return total


def _analyze_kernels(kernels: list[dict]) -> dict:
    if not kernels:
        return {}
    t0, t1 = min(k["start"] for k in kernels), max(k["end"] for k in kernels)
    span_ns = t1 - t0
    nccl = [k for k in kernels if k["cls"] == "nccl"]
    comp = [k for k in kernels if k["cls"] in COMP_CLS]
    nccl_iv = _merge([(k["start"], k["end"]) for k in nccl])
    comp_iv = _merge([(k["start"], k["end"]) for k in comp])
    ov_ns = _overlap_ns(nccl_iv, comp_iv)

    by_stream: dict[int, dict] = defaultdict(lambda: {"dur_ns": 0, "nccl_ns": 0, "comp_ns": 0, "count": 0})
    for k in kernels:
        st = by_stream[k["streamId"]]
        st["dur_ns"] += k["dur_ns"]
        st["count"] += 1
        if k["cls"] == "nccl":
            st["nccl_ns"] += k["dur_ns"]
        elif k["cls"] in COMP_CLS:
            st["comp_ns"] += k["dur_ns"]

    first_nccl = min((k["start"] for k in nccl), default=None)
    last_nccl = max((k["end"] for k in nccl), default=None)
    first_comp = min((k["start"] for k in comp), default=None)
    last_comp = max((k["end"] for k in comp), default=None)

    return {
        "kernel_span_ms": span_ns / 1e6,
        "nccl_comp_concurrent_ms": ov_ns / 1e6,
        "nccl_comp_concurrent_pct": 100.0 * ov_ns / span_ns if span_ns else 0.0,
        "nccl_kernel_sum_ms": sum(k["dur_ns"] for k in nccl) / 1e6,
        "comp_kernel_sum_ms": sum(k["dur_ns"] for k in comp) / 1e6,
        "first_nccl_ms": (first_nccl - t0) / 1e6 if first_nccl else None,
        "last_nccl_ms": (last_nccl - t0) / 1e6 if last_nccl else None,
        "first_comp_ms": (first_comp - t0) / 1e6 if first_comp else None,
        "last_comp_ms": (last_comp - t0) / 1e6 if last_comp else None,
        "streams": {
            str(sid): {
                "busy_ms": v["dur_ns"] / 1e6,
                "nccl_ms": v["nccl_ns"] / 1e6,
                "comp_ms": v["comp_ns"] / 1e6,
                "kernels": v["count"],
            }
            for sid, v in sorted(by_stream.items())
        },
        "timeline_ms": [
            {
                "t_ms": (k["start"] - t0) / 1e6,
                "dur_ms": k["dur_ns"] / 1e6,
                "stream": k["streamId"],
                "cls": k["cls"],
                "name": k["short"],
            }
            for k in kernels[:100]
        ],
    }


def _runtime_sync(conn: sqlite3.Connection, t0: int, t1: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT r.start, r.end, s.value, r.callchainId, r.globalTid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId=s.id
        WHERE r.end>? AND r.start<? AND (
            s.value LIKE '%Synchron%' OR s.value LIKE '%WaitEvent%'
        )
        ORDER BY r.start
        """,
        (t0, t1),
    ).fetchall()
    return [
        {
            "start": st, "end": en, "dur_ms": (en - st) / 1e6,
            "api": api, "callchainId": cc, "globalTid": tid,
            "t_ms": (st - t0) / 1e6,
        }
        for st, en, api, cc, tid in rows
    ]


def _gpu_sync(conn: sqlite3.Connection, t0: int, t1: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT syncType, start, end, streamId, correlationId
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        WHERE end>? AND start<?
        ORDER BY start
        """,
        (t0, t1),
    ).fetchall()
    by_type: dict[int, dict] = defaultdict(lambda: {"count": 0, "dur_ms": 0.0})
    for sync_type, start, end, _stream, _corr in rows:
        by_type[sync_type]["count"] += 1
        by_type[sync_type]["dur_ms"] += (end - start) / 1e6
    return [
        {
            "syncType": st,
            "label": SYNC_TYPE.get(st, f"type_{st}"),
            "count": v["count"],
            "dur_ms": v["dur_ms"],
        }
        for st, v in sorted(by_type.items())
    ]


def _resolve_callchain(conn: sqlite3.Connection, callchain_id: int, depth: int = 8) -> list[str]:
    rows = conn.execute(
        """
        SELECT cc.stackDepth, sym.value, mod.value
        FROM OSRT_CALLCHAINS cc
        LEFT JOIN StringIds sym ON cc.symbol = sym.id
        LEFT JOIN StringIds mod ON cc.module = mod.id
        WHERE cc.id=? AND cc.stackDepth < ?
        ORDER BY cc.stackDepth
        """,
        (callchain_id, depth),
    ).fetchall()
    out = []
    for _depth, symbol, module in rows:
        if symbol:
            out.append(symbol)
        elif module:
            out.append(f"[{module}]")
    return out


def _sync_backtraces(conn: sqlite3.Connection, t0: int, t1: int) -> list[dict]:
    """Top callstacks for cudaDeviceSynchronize (requires --cudabacktrace=sync)."""
    rows = conn.execute(
        """
        SELECT r.callchainId, SUM(r.end-r.start) as dur, COUNT(*) as cnt
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId=s.id
        WHERE r.end>? AND r.start<? AND s.value='cudaDeviceSynchronize_v3020'
          AND r.callchainId IS NOT NULL AND r.callchainId != 0
        GROUP BY r.callchainId
        ORDER BY dur DESC
        LIMIT 10
        """,
        (t0, t1),
    ).fetchall()
    result = []
    for cc_id, dur, cnt in rows:
        stack = _resolve_callchain(conn, cc_id)
        result.append({
            "callchainId": cc_id,
            "count": cnt,
            "total_ms": dur / 1e6,
            "stack_top8": stack,
        })
    return result


def _memcpy_overlap(conn: sqlite3.Connection, t0: int, t1: int, device: int = 0) -> dict:
    memcpy_rows = conn.execute(
        """
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE deviceId=? AND end>? AND start<?
        """,
        (device, t0, t1),
    ).fetchall()
    comp_iv = _merge([
        (k["start"], k["end"])
        for k in _kernels(conn, t0, t1, device)
        if k["cls"] in COMP_CLS
    ])
    memcpy_iv = _merge(list(memcpy_rows))
    span = t1 - t0
    ov = _overlap_ns(memcpy_iv, comp_iv)
    return {
        "memcpy_envelope_ms": sum(e - s for s, e in memcpy_iv) / 1e6,
        "memcpy_comp_concurrent_ms": ov / 1e6,
        "memcpy_comp_concurrent_pct": 100.0 * ov / span if span else 0.0,
    }


def main() -> None:
    if not SQLITE.is_file():
        raise SystemExit(f"missing sqlite: {SQLITE}")

    conn = sqlite3.connect(SQLITE)
    wall = json.loads(WALL_JSON.read_text()) if WALL_JSON.is_file() else {}

    per_range: dict[str, dict] = {}
    for label in RANGES:
        t0, t1 = _nvtx_span(conn, label)
        if t0 >= t1:
            continue
        kernels = _kernels(conn, t0, t1)
        per_range[label] = {
            "nvtx_wall_ms": (t1 - t0) / 1e6,
            **_analyze_kernels(kernels),
            "runtime_sync": _runtime_sync(conn, t0, t1),
            "gpu_sync_by_type": _gpu_sync(conn, t0, t1),
            "memcpy_overlap": _memcpy_overlap(conn, t0, t1),
            "device_sync_backtraces": _sync_backtraces(conn, t0, t1),
        }

    # Aggregate runtime sync counts for OVERLAP
    ov = per_range.get("OVERLAP_comm_p2_comp_p1", {})
    rt_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "dur_ms": 0.0})
    for ev in ov.get("runtime_sync", []):
        api = ev["api"]
        if api.endswith("_v3020"):
            continue  # duplicate of unsuffixed entry
        rt_agg[api]["count"] += 1
        rt_agg[api]["dur_ms"] += ev["dur_ms"]

    out = {
        "sqlite": str(SQLITE),
        "wall_rank0": wall,
        "per_range": per_range,
        "overlap_runtime_sync_summary": dict(rt_agg),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# TP2 overlap Nsight 深度分析（Fix A: stream_ar_wait, norm P2P, NCCL row AR）",
        "",
        f"数据源: `{SQLITE.name}`",
        "",
        "## 1. 墙钟（rank0）",
        "",
        "| 场景 | ms |",
        "|---|---:|",
    ]
    for k, label in [
        ("comm_alone_ms", "COMM alone"),
        ("comp_alone_ms", "COMP alone"),
        ("ideal_max_ms", "ideal max"),
        ("overlap_ms", "OVERLAP"),
        ("serial_sum_ms", "serial sum"),
    ]:
        v = wall.get(k)
        lines.append(f"| {label} | {v:.2f} |" if v is not None else f"| {label} | — |")

    lines.extend([
        "",
        "## 2. Kernel 级 NCCL ∥ compute 并发",
        "",
        "| 场景 | span ms | NCCL∥compute 同时运行 | 占比 |",
        "|---|---:|---:|---:|",
    ])
    for label, short in [
        ("COMM_alone_p2", "COMM"),
        ("COMP_alone_p1", "COMP"),
        ("OVERLAP_comm_p2_comp_p1", "OVERLAP"),
    ]:
        r = per_range.get(label, {})
        lines.append(
            f"| {short} | {r.get('kernel_span_ms', 0):.2f} | "
            f"**{r.get('nccl_comp_concurrent_ms', 0):.3f}** | "
            f"{r.get('nccl_comp_concurrent_pct', 0):.1f}% |"
        )

    if ov:
        lines.extend([
            "",
            "## 3. OVERLAP 时间线（相对窗口起点）",
            "",
            f"- 第一个 compute kernel: **{ov.get('first_comp_ms', 0):.2f} ms**",
            f"- 第一个 NCCL kernel: **{ov.get('first_nccl_ms', 0):.2f} ms**",
            f"- 最后一个 NCCL 结束: **{ov.get('last_nccl_ms', 0):.2f} ms**",
            f"- 最后一个 compute 结束: **{ov.get('last_comp_ms', 0):.2f} ms**",
            "",
            "**解读**: 若 compute 先跑、NCCL 晚启动且二者 envelope 不交叠 → 软件调度/同步导致串行，而非 SM 竞争。",
            "",
            "### 按 stream（kernel duration 累加）",
            "",
            "| stream | busy ms | NCCL ms | compute ms |",
            "|---|---:|---:|---:|",
        ])
        for sid, st in ov.get("streams", {}).items():
            if st["busy_ms"] < 0.01:
                continue
            lines.append(f"| {sid} | {st['busy_ms']:.2f} | {st['nccl_ms']:.2f} | {st['comp_ms']:.2f} |")

    lines.extend([
        "",
        "## 4. OVERLAP 窗口内 CUDA Runtime 同步（根因候选）",
        "",
        "| API | 次数 | 累计 ms |",
        "|---|---:|---:|",
    ])
    for api, agg in sorted(rt_agg.items(), key=lambda x: -x[1]["dur_ms"]):
        lines.append(f"| `{api}` | {agg['count']} | **{agg['dur_ms']:.2f}** |")

    lines.extend([
        "",
        "### GPU 同步活动 (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)",
        "",
        "| type | 次数 | 累计 ms |",
        "|---|---:|---:|",
    ])
    for row in ov.get("gpu_sync_by_type", []):
        lines.append(f"| {row['label']} ({row['syncType']}) | {row['count']} | {row['dur_ms']:.2f} |")

    bt = ov.get("device_sync_backtraces", [])
    lines.extend([
        "",
        "## 5. cudaDeviceSynchronize 调用栈（需 `--cudabacktrace=sync`）",
        "",
    ])
    if bt:
        for i, b in enumerate(bt[:5], 1):
            lines.append(f"### #{i} — {b['count']} 次, {b['total_ms']:.2f} ms")
            for frame in b["stack_top8"][:6]:
                lines.append(f"- `{frame}`")
            lines.append("")
    else:
        lines.append("（未采集到 callchain；请用 `--cudabacktrace=sync` 重新 profile）")

    mc = ov.get("memcpy_overlap", {})
    lines.extend([
        "",
        "## 6. Memcpy ∥ compute",
        "",
        f"- memcpy envelope: **{mc.get('memcpy_envelope_ms', 0):.2f} ms**",
        f"- memcpy∥compute 同时: **{mc.get('memcpy_comp_concurrent_ms', 0):.3f} ms** "
        f"({mc.get('memcpy_comp_concurrent_pct', 0):.1f}%)",
        "",
        "## 7. 结论",
        "",
    ])

    conc = ov.get("nccl_comp_concurrent_pct", 0)
    dev_sync_ms = rt_agg.get("cudaDeviceSynchronize_v3020", {}).get("dur_ms", 0)
    overlap_ms = wall.get("overlap_ms", 0)
    ideal = wall.get("ideal_max_ms", 0)

    if conc < 5:
        lines.append(
            f"1. **Kernel 级 NCCL∥compute 并发 ≈ 0%**（{conc:.1f}%）— overlap 失败是真实串行，不是测量假象。"
        )
    if dev_sync_ms > overlap_ms * 0.5:
        lines.append(
            f"2. **OVERLAP 窗口内 `cudaDeviceSynchronize` 累计 ~{dev_sync_ms:.1f} ms**，"
            f"与 wall overlap ~{overlap_ms:.1f} ms 同量级 — **全设备 sync 是主因**。"
        )
    first_nccl = ov.get("first_nccl_ms")
    first_comp = ov.get("first_comp_ms")
    if first_nccl is not None and first_comp is not None and first_nccl > first_comp + 1.0:
        lines.append(
            f"3. **NCCL 晚启动**（first NCCL {first_nccl:.1f} ms vs first compute {first_comp:.1f} ms）— "
            "peer COMP 先跑完大量 kernel，NCCL 才开始，窗口被拉长。"
        )
    if ideal and overlap_ms:
        lines.append(
            f"4. ideal max **{ideal:.1f} ms** vs overlap **{overlap_ms:.1f} ms** "
            f"（{overlap_ms / ideal:.2f}×）— Fix A 未能消除 device sync 链。"
        )

    print(f"wrote {OUT_JSON}")
    if OUT_MD:
        Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {OUT_MD}")
    print("结论文档: save_results/nsys/tp2_overlap_root_cause.md")


if __name__ == "__main__":
    main()
