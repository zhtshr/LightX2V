#!/usr/bin/env python3
"""Summarize T2V-1.3B dual overlap results across resolutions."""

from __future__ import annotations

import json
import re
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
PAT = re.compile(r"p3_dual_overlap_t2v_1\.3b_(\d+)x\1_seqp(\d+)\.json")


def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def main() -> None:
    rows: list[dict] = []
    for path in sorted(STUDY.glob("p3_dual_overlap_t2v_1.3b_*x*_seqp*.json")):
        m = PAT.match(path.name)
        if not m:
            continue
        res, p = int(m.group(1)), int(m.group(2))
        d = json.loads(path.read_text())
        if d.get("error"):
            continue
        tp = d.get("throughput", {})
        sp = d.get("speedup", {})
        single_s = d.get("single_decomposed_s") or d.get("single_transformer_s")
        overlap_s = d.get("dual_a2a_overlap_s")
        serial_s = d.get("dual_a2a_serial_s")
        fair_b2b = d.get("fair_dual_baseline_s") or d.get("dual_decomposed_back_to_back_s")

        p1_path = STUDY / f"p1_t2v_1.3b_{res}x{res}_seqp1.json"
        t1 = None
        if p1_path.is_file():
            t1 = json.loads(p1_path.read_text()).get("transformer_compute_s")

        ideal_rps = (p / t1) if t1 and t1 > 0 else None
        overlap_rps = tp.get("dual_a2a_overlap_rps")
        vs_ideal = (overlap_rps / ideal_rps) if ideal_rps and overlap_rps else None

        rows.append({
            "resolution": res,
            "seq_p": p,
            "single_sp_s": single_s,
            "dual_overlap_s": overlap_s,
            "dual_serial_s": serial_s,
            "fair_b2b_s": fair_b2b,
            "overlap_rps": overlap_rps,
            "a2a_vs_serial": sp.get("a2a_overlap_vs_a2a_serial"),
            "a2a_vs_fair": sp.get("a2a_overlap_vs_fair_dual_baseline"),
            "vs_ideal": vs_ideal,
            "t1": t1,
        })

    rows.sort(key=lambda r: (r["resolution"], r["seq_p"]))

    lines = [
        "# Wan2.1-T2V-1.3B — Dual a2a-overlap vs 分辨率 × SP",
        "",
        "口径：`cpu_offload=false`，4 steps，`enable_cfg=false`；2 req in-flight wall。",
        "「vs N/T₁」= overlap 吞吐 / 同卡数独立单卡理想 **N/T₁**（T₁ 为同分辨率 P=1 `transformer_compute_s`）。",
        "",
        "## 汇总表",
        "",
        "| 分辨率 | P | 单请求 SP (s) | dual overlap wall (2 req) | overlap 吞吐 | a2a-overlap vs serial | vs N/T₁ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['resolution']}×{r['resolution']} | {r['seq_p']} | {f(r['single_sp_s'])} | "
            f"{f(r['dual_overlap_s'])} | {f(r['overlap_rps'])} | "
            f"{f(r['a2a_vs_serial'], 2)}× | {pct(r['vs_ideal'])} |"
        )

    out_md = STUDY / "phase3_dual_overlap_t2v_resolution.md"
    out_json = STUDY / "phase3_dual_overlap_t2v_resolution.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()
