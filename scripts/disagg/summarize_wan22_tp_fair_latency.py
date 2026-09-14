#!/usr/bin/env python3
"""Summarize fair Wan2.2 TP=2/4/8 latency rerun with mean/std."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"mean": None, "std": None, "min": None, "max": None, "cv_pct": None}
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    cv = 100.0 * std / mean if mean else None
    return {
        "mean": mean,
        "std": std,
        "min": min(samples),
        "max": max(samples),
        "cv_pct": cv,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", default="/root/zht/LightX2V/save_results/optimization_study")
    parser.add_argument(
        "--output",
        default="/root/zht/LightX2V/save_results/optimization_study/wan22_distill_tp_fair_comparison.md",
    )
    parser.add_argument("--prefix", default="wan22_distill_tp")
    parser.add_argument("--suffix", default="_fair")
    args = parser.parse_args()

    study = Path(args.study_dir)
    rows = []
    for tp in [2, 4, 8]:
        path = study / f"{args.prefix}{tp}{args.suffix}.json"
        row: dict = {"tp": tp, "ok": False}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            samples = data.get("transformer_compute_samples_s") or []
            st = _stats(samples)
            row.update(
                {
                    "ok": st["mean"] is not None,
                    "samples": samples,
                    "st": st,
                    "world": data.get("world_size"),
                    "warmup": data.get("warmup_iters"),
                    "measure": data.get("measure_iters"),
                    "config": data.get("config_json"),
                    "model_load_s": data.get("model_load_s_excluded"),
                }
            )
        rows.append(row)

    # Also load prior single-sample runs for side-by-side if present.
    prior: dict[int, float | None] = {}
    for tp in [2, 4, 8]:
        p = study / f"{args.prefix}{tp}_real.json"
        if p.is_file():
            prior[tp] = json.loads(p.read_text(encoding="utf-8")).get("transformer_compute_s")
        else:
            prior[tp] = None

    base_tp2 = next((r["st"]["mean"] for r in rows if r["tp"] == 2 and r.get("ok")), None)

    lines = [
        "# Wan2.2-Distill — Fair TP Latency Comparison (TP=2/4/8)",
        "",
        "Unified settings for all runs:",
        "- `cpu_offload=false`, real weight sharding",
        "- `unload_modules=false` (both experts resident; no lazy reload in timed path)",
        "- Same 4-step denoise, 480×832×81, int8-q8f",
        "- `warmup=2`, `measure_iters=5`",
        "- Run order: **2 → 8 → 4** (interleaved to reduce thermal bias)",
        "",
        "Metric: `transformer_compute_s` = scheduler.prepare + denoise loop (excludes model load).",
        "",
        "## Fair rerun results",
        "",
        "| TP | mean (s) | std | min | max | CV% | speedup vs TP=2 | prior 1-sample (s) | samples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        if not row.get("ok"):
            lines.append(f"| {row['tp']} | FAILED | — | — | — | — | — | {prior.get(row['tp'], '—')} | — |")
            continue
        st = row["st"]
        speedup = f"{base_tp2 / st['mean']:.3f}×" if base_tp2 and st["mean"] else "—"
        samp = ", ".join(f"{s:.2f}" for s in row["samples"])
        prior_s = prior.get(row["tp"])
        prior_str = f"{prior_s:.2f}" if prior_s is not None else "—"
        lines.append(
            f"| {row['tp']} | {st['mean']:.2f} | {st['std']:.2f} | {st['min']:.2f} | "
            f"{st['max']:.2f} | {st['cv_pct']:.1f}% | {speedup} | {prior_str} | {samp} |"
        )

    lines.extend(["", "## Interpretation", ""])
    ok = [r for r in rows if r.get("ok")]
    if len(ok) >= 2 and base_tp2:
        ordered = sorted(ok, key=lambda r: r["st"]["mean"])
        best = ordered[0]
        worst = ordered[-1]
        lines.append(
            f"- Best: **TP={best['tp']}** at {best['st']['mean']:.2f}s "
            f"(±{best['st']['std']:.2f}s)"
        )
        lines.append(
            f"- Worst: **TP={worst['tp']}** at {worst['st']['mean']:.2f}s "
            f"(±{worst['st']['std']:.2f}s)"
        )
        if worst["st"]["mean"] and best["st"]["mean"]:
            gap = 100.0 * (worst["st"]["mean"] - best["st"]["mean"]) / best["st"]["mean"]
            lines.append(f"- Spread best→worst: **{gap:.1f}%**")
        for row in ok:
            if row["tp"] == 1:
                continue
            eff = (base_tp2 / row["st"]["mean"]) / row["tp"]
            lines.append(
                f"- TP={row['tp']}: scaling efficiency vs TP=2 baseline = **{100 * eff:.1f}%**"
            )
    else:
        lines.append("- Incomplete runs; check logs in `save_results/optimization_study/`.")

    lines.extend(
        [
            "",
            "## vs prior sweep",
            "",
            "Prior `wan22_distill_tp*_real.json` used `unload_modules=true` and `measure_iters=1`,",
            "so expert `tp_load` + NCCL distribute was inside the timed window — especially noisy at TP=4.",
            "",
            "Re-run: `bash scripts/disagg/run_wan22_tp_fair_rerun.sh`",
        ]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
