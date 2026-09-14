#!/usr/bin/env python3
"""Summarize per-stage / per-step throughput from disagg controller metrics JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _request_tags(req: dict[str, Any]) -> tuple[str, str]:
    metrics = req.get("request_metrics") if isinstance(req.get("request_metrics"), dict) else {}
    stage = str(metrics.get("stage_name") or metrics.get("load_stage") or "unknown")
    steps = metrics.get("infer_steps")
    if steps is None:
        steps = req.get("infer_steps")
    return stage, str(steps if steps is not None else "default")


def _block(items: list[float], window_s: float | None) -> dict[str, Any]:
    n = len(items)
    out: dict[str, Any] = {
        "count": n,
        "e2e_mean_s": round(sum(items) / n, 3) if n else None,
        "e2e_p50_s": round(sorted(items)[n // 2], 3) if n else None,
    }
    if window_s and window_s > 0 and n:
        out["throughput_rps"] = round(n / window_s, 4)
    return out


def summarize(metrics_path: Path) -> dict[str, Any]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    requests = data.get("requests") if isinstance(data.get("requests"), list) else []
    batch_total_time_s = _as_float(data.get("batch_total_time_s"))

    by_stage: dict[str, list[float]] = defaultdict(list)
    by_steps: dict[str, list[float]] = defaultdict(list)
    by_stage_steps: dict[str, list[float]] = defaultdict(list)
    e2e_all: list[float] = []
    first_send: float | None = None
    last_done: float | None = None
    stage_span: dict[str, list[float]] = defaultdict(lambda: [float("inf"), float("-inf")])

    for req in requests:
        if not isinstance(req, dict):
            continue
        metrics = req.get("request_metrics") if isinstance(req.get("request_metrics"), dict) else {}
        summary = req.get("latency_summary") if isinstance(req.get("latency_summary"), dict) else {}
        e2e = _as_float(summary.get("end_to_end_delay_s"))
        send_ts = _as_float(metrics.get("controller_send_ts"))
        done_ts = _as_float(req.get("controller_recv_ts"))
        if done_ts is None and send_ts is not None and e2e is not None:
            done_ts = send_ts + e2e
        if send_ts is not None:
            first_send = send_ts if first_send is None else min(first_send, send_ts)
        if done_ts is not None:
            last_done = done_ts if last_done is None else max(last_done, done_ts)
        if e2e is None:
            continue
        stage, steps = _request_tags(req)
        by_stage[stage].append(e2e)
        by_steps[steps].append(e2e)
        by_stage_steps[f"{stage}|s{steps}"].append(e2e)
        e2e_all.append(e2e)
        if send_ts is not None and done_ts is not None:
            stage_span[stage][0] = min(stage_span[stage][0], send_ts)
            stage_span[stage][1] = max(stage_span[stage][1], done_ts)

    wall_s = None
    if first_send is not None and last_done is not None and last_done > first_send:
        wall_s = last_done - first_send
    if wall_s is None:
        wall_s = batch_total_time_s

    stage_windows: dict[str, float] = {}
    for stage, (lo, hi) in stage_span.items():
        if hi > lo and lo != float("inf"):
            stage_windows[stage] = hi - lo

    monitor = data.get("monitor_samples") if isinstance(data.get("monitor_samples"), list) else []
    service_addrs: dict[str, set[str]] = defaultdict(set)
    for sample in monitor:
        if not isinstance(sample, dict) or sample.get("status") != "ok":
            continue
        st = str(sample.get("service_type", ""))
        addr = str(sample.get("address", ""))
        if st and addr:
            service_addrs[st].add(addr)

    return {
        "metrics_path": str(metrics_path),
        "completed": len(e2e_all),
        "wall_time_s": round(wall_s, 3) if wall_s else None,
        "batch_total_time_s": batch_total_time_s,
        "overall": _block(e2e_all, wall_s),
        "by_stage": {k: _block(v, stage_windows.get(k, wall_s)) for k, v in sorted(by_stage.items())},
        "by_infer_steps": {k: _block(v, wall_s) for k, v in sorted(by_steps.items())},
        "by_stage_steps": {k: _block(v, wall_s) for k, v in sorted(by_stage_steps.items())},
        "unique_instances_seen": {k: len(v) for k, v in sorted(service_addrs.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(args.metrics_json)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
