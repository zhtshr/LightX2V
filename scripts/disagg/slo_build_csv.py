#!/usr/bin/env python3
"""Build SLO CSV from arrival jsonl + DisagFusion controller metrics JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arrivals", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scheme", type=str, required=True)
    p.add_argument("--arrival_rate", type=float, required=True)
    p.add_argument("--timeout_s", type=float, default=600.0)
    p.add_argument("--now_ts", type=float, default=0.0, help="wall clock when merging; 0=use max known")
    p.add_argument(
        "--finish_from",
        choices=("controller_recv", "decoder_done"),
        default="controller_recv",
        help="controller_recv=when controller got result; decoder_done=decoder output_enqueued "
        "(use to salvage runs where controller only drained on next ingress)",
    )
    args = p.parse_args()

    arrivals: dict[int, dict[str, Any]] = {}
    for line in args.arrivals.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        arrivals[int(rec["req_id"])] = rec

    finishes: dict[int, float] = {}
    errors: set[int] = set()
    if args.metrics.exists():
        data = json.loads(args.metrics.read_text(encoding="utf-8"))
        for req in data.get("requests") or []:
            if not isinstance(req, dict):
                continue
            metrics = req.get("request_metrics") if isinstance(req.get("request_metrics"), dict) else {}
            rid = metrics.get("request_id")
            if rid is None:
                rid = req.get("data_bootstrap_room")
            if rid is None:
                continue
            rid = int(rid)
            done = None
            if args.finish_from == "decoder_done":
                stages = metrics.get("stages") if isinstance(metrics.get("stages"), dict) else {}
                dec = stages.get("decoder") if isinstance(stages.get("decoder"), dict) else {}
                done = _f(dec.get("output_enqueued_ts")) or _f(dec.get("compute_end_ts"))
            if done is None:
                done = _f(req.get("controller_recv_ts"))
            if done is None:
                summary = req.get("latency_summary") if isinstance(req.get("latency_summary"), dict) else {}
                e2e = _f(summary.get("end_to_end_delay_s"))
                send = _f(metrics.get("client_send_ts")) or _f(metrics.get("controller_send_ts"))
                if e2e is not None and send is not None:
                    done = send + e2e
            if req.get("error") or req.get("status") == "error":
                errors.add(rid)
            if done is not None:
                finishes[rid] = done

    now = args.now_ts
    if now <= 0:
        now = max([float(r["arrival_ts"]) for r in arrivals.values()] + list(finishes.values()) + [0.0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scheme", "arrival_rate", "req_id", "arrival_ts", "finish_ts", "is_warmup", "status"],
        )
        w.writeheader()
        for rid in sorted(arrivals):
            rec = arrivals[rid]
            arrival = float(rec["arrival_ts"])
            is_warmup = int(rec.get("is_warmup", 0))
            if rid in errors and rid not in finishes:
                status, finish = "error", ""
            elif rid in finishes:
                status, finish = "ok", f"{finishes[rid]:.6f}"
            elif now - arrival >= args.timeout_s:
                status, finish = "timeout", ""
            else:
                # Still pending at merge time — treat as timeout for SLO delivery.
                status, finish = "timeout", ""
            w.writerow(
                {
                    "scheme": args.scheme,
                    "arrival_rate": args.arrival_rate,
                    "req_id": rid,
                    "arrival_ts": f"{arrival:.6f}",
                    "finish_ts": finish,
                    "is_warmup": is_warmup,
                    "status": status,
                }
            )
    print(f"wrote {args.out} ({len(arrivals)} rows)")


if __name__ == "__main__":
    main()
