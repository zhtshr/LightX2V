#!/usr/bin/env python3
"""Summarize Wan2.2-Distill-Models TP latency sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", default="/root/zht/LightX2V/save_results/optimization_study")
    parser.add_argument("--output", default="/root/zht/LightX2V/save_results/optimization_study/wan22_distill_tp_scaling_real.md")
    parser.add_argument("--prefix", default="wan22_distill_tp")
    args = parser.parse_args()

    study = Path(args.study_dir)
    rows = []
    for tp in [1, 2, 4, 8]:
        path = study / f"{args.prefix}{tp}_real.json"
        row = {"tp": tp, "ok": False}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            t = data.get("transformer_compute_s")
            samples = data.get("transformer_compute_samples_s") or []
            row.update(
                {
                    "ok": t is not None,
                    "t": t,
                    "samples": samples,
                    "world": data.get("world_size"),
                    "parallel_mode": data.get("parallel_mode"),
                    "model_load_s": data.get("model_load_s_excluded"),
                }
            )
        rows.append(row)

    base_t = next((r["t"] for r in rows if r["tp"] == 1 and r.get("t")), None)

    lines = [
        "# Wan2.2-Distill-Models — Tensor Parallel Latency (real TP)",
        "",
        "Model: `Wan2.2-Distill-Models` / `wan2.2_moe` I2V int8-q8f",
        "Settings: TP=1 uses `cpu_offload=block` (single-GPU baseline, no weight shard);",
        "TP≥2 uses real TP + `cpu_offload=false`. 4 denoise steps, 480×832×81 int8-q8f.",
        "Metric: `transformer_compute_s` = scheduler.prepare + denoise loop (excludes load/encoder/decoder)",
        "",
        "## Results",
        "",
        "| TP | world | transformer_s | speedup vs TP=1 | scaling efficiency | samples (s) |",
        "|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        t = row.get("t")
        if t is None:
            lines.append(f"| {row['tp']} | — | FAILED | — | — | — |")
            continue
        speedup = f"{base_t / t:.3f}×" if base_t else "—"
        eff = f"{100 * (base_t / t) / row['tp']:.1f}%" if base_t else "—"
        samp = ", ".join(f"{s:.3f}" for s in row.get("samples", [])) or "—"
        lines.append(
            f"| {row['tp']} | {row.get('world', '—')} | {t:.3f} | {speedup} | {eff} | {samp} |"
        )

    lines.extend(["", "## Scaling summary", ""])
    if base_t:
        ok_rows = [r for r in rows if r.get("t")]
        if len(ok_rows) >= 2:
            best = min(ok_rows, key=lambda r: r["t"])
            lines.append(f"- Baseline (TP=1): **{base_t:.3f}s**")
            lines.append(f"- Best latency: TP={best['tp']} at **{best['t']:.3f}s** ({base_t / best['t']:.3f}× vs TP=1)")
            for row in ok_rows:
                if row["tp"] == 1:
                    continue
                eff = (base_t / row["t"]) / row["tp"]
                lines.append(
                    f"- TP={row['tp']}: {row['t']:.3f}s, speedup {base_t / row['t']:.3f}×, "
                    f"scaling efficiency {100 * eff:.1f}%"
                )
            avg_eff = sum((base_t / r["t"]) / r["tp"] for r in ok_rows if r["tp"] > 1) / max(
                1, sum(1 for r in ok_rows if r["tp"] > 1)
            )
            lines.append(f"- Average scaling efficiency (TP>1): **{100 * avg_eff:.1f}%**")
        else:
            lines.append("- Only TP=1 completed; multi-GPU runs may have failed (OOM or env).")
    else:
        lines.append("- No successful runs.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Real TP: weights sharded via `MMWeightTP` + rank0 distribute; requires `cpu_offload=false`.",
            "- MoE loads high+low noise experts; memory scales ~1/tp per rank for linear weights.",
            "- TP adds per-layer all-reduce; speedup depends on compute/comm balance.",
            "",
            f"Run: `FORCE_RERUN=1 bash scripts/disagg/run_wan22_tp_latency_sweep.sh`",
        ]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
