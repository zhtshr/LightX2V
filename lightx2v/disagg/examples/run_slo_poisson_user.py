#!/usr/bin/env python3
"""Open-loop Poisson arrival user for DisagFusion SLO benches.

Env:
  SLO_ARRIVAL_RATE     target λ (req/s), required
  SLO_NUM_REQUESTS     total requests to send (default 65)
  SLO_WARMUP_REQUESTS  first K marked warmup (default 5)
  SLO_SEED             RNG seed (default 0)
  SLO_ARRIVAL_LOG      jsonl path for (req_id, arrival_ts, is_warmup)
  DISAGG_CONTROLLER_HOST / DISAGG_CONTROLLER_REQUEST_PORT  (same as run_user)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, ReqManager
from lightx2v.disagg.workload import (
    build_payload,
    current_stage,
    load_base_config,
    load_stage_specs,
    send_workload_end_signal,
    start_workload_clock,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poisson open-loop DisagFusion SLO user")
    parser.add_argument("--controller_host", type=str, default=os.getenv("DISAGG_CONTROLLER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--controller_request_port",
        type=int,
        default=int(os.getenv("DISAGG_CONTROLLER_REQUEST_PORT", str(REQUEST_POLLING_PORT - 2))),
    )
    parser.add_argument("--max_requests", type=int, default=0, help="ignored; use SLO_NUM_REQUESTS")
    args = parser.parse_args()

    rate = float(os.getenv("SLO_ARRIVAL_RATE", "0"))
    if rate <= 0:
        raise SystemExit("SLO_ARRIVAL_RATE must be > 0")
    n = int(os.getenv("SLO_NUM_REQUESTS", "65"))
    warmup_n = int(os.getenv("SLO_WARMUP_REQUESTS", "5"))
    seed = int(os.getenv("SLO_SEED", "0"))
    arrival_log = Path(
        os.getenv(
            "SLO_ARRIVAL_LOG",
            os.getenv("DISAGG_CONTROLLER_METRICS_OUTPUT_JSON", "/tmp/slo_arrivals.jsonl").replace(
                "_metrics.json", "_arrivals.jsonl"
            ),
        )
    )
    arrival_log.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    req_mgr = ReqManager()
    stages = load_stage_specs()
    base_config = load_base_config()
    start_workload_clock()

    print(f"slo poisson user: rate={rate} n={n} warmup={warmup_n} seed={seed} log={arrival_log}")
    with arrival_log.open("w", encoding="utf-8") as logf:
        for i in range(n):
            stage = current_stage(stages)
            payload = build_payload(base_config, stage, i)
            metrics = payload.setdefault("request_metrics", {})
            metrics["request_id"] = i
            metrics["slo_arrival_rate"] = rate
            arrival_ts = time.time()
            metrics["client_send_ts"] = arrival_ts
            is_warmup = 1 if i < warmup_n else 0
            metrics["slo_is_warmup"] = bool(is_warmup)

            req_mgr.send(args.controller_host, args.controller_request_port, payload)
            rec = {
                "req_id": i,
                "arrival_ts": arrival_ts,
                "is_warmup": is_warmup,
                "arrival_rate": rate,
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"sent req_id={i} arrival_ts={arrival_ts:.3f} warmup={is_warmup}")

            if i + 1 < n:
                time.sleep(rng.expovariate(rate))

    send_workload_end_signal()
    print(f"workload finished: sent={n}, end signal sent")


if __name__ == "__main__":
    main()
