#!/usr/bin/env python3
"""Summarize MoE I2V dual overlap results across resolutions."""

from __future__ import annotations

import json
import re
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
PAT_SQ = re.compile(r"p3_dual_overlap_moe_i2v_(\d+)x\1_seqp(\d+)\.json")
PAT_480 = re.compile(r"p3_dual_overlap_moe_i2v_480x832_seqp(\d+)\.json")


def f(v, n=1):
    return "—" if v is None else f"{v:.{n}f}"


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def main() -> None:
    rows: list[dict] = []
    for path in sorted(STUDY.glob("p3_dual_overlap_moe_i2v_*_seqp*.json")):
        m = PAT_SQ.match(path.name) or PAT_480.match(path.name)
        if not m:
            continue
        if path.name.startswith("p3_dual_overlap_moe_i2v_480"):
            res, p = 480, int(m.group(1))
            label = "480×832"
            p1_path = STUDY / "p3_sp_seqp1.json"
            sp_path = STUDY / f"p3_sp_seqp{p}.json"
        else:
            res, p = int(m.group(1)), int(m.group(2))
            label = f"{res}×{res}"
            p1_path = STUDY / f"p1_moe_i2v_{res}x{res}_seqp1.json"
            sp_path = STUDY / f"p1_moe_i2v_{res}x{res}_seqp{p}.json"

        d = json.loads(path.read_text())
        if d.get("error") or not d.get("dual_a2a_overlap_s"):
            continue
        t1 = json.loads(p1_path.read_text()).get("transformer_compute_s") if p1_path.is_file() else None
        sp_s = json.loads(sp_path.read_text()).get("transformer_compute_s") if sp_path.is_file() else None
        ov = d["dual_a2a_overlap_s"]
        rows.append({
            "label": label,
            "resolution": res,
            "seq_p": p,
            "t1": t1,
            "sp_s": sp_s,
            "sp_eff": (t1 / sp_s / p) if t1 and sp_s else None,
            "ov_s": ov,
            "ov_eff": (2 * t1 / (ov * p)) if t1 and ov else None,
        })

    rows.sort(key=lambda r: (r["resolution"], r["seq_p"]))
    out_md = STUDY / "phase3_dual_overlap_moe_resolution.md"
    out_json = STUDY / "phase3_dual_overlap_moe_resolution.json"
    lines = [
        "# Wan2.2-MoE I2V — Dual a2a-overlap vs 分辨率 × SP",
        "",
        "| 分辨率 | P | 单卡去噪 (s) | 单请求 (s) | SP 扩展效率 | overlap (s) | overlap 扩展效率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['seq_p']} | {f(r['t1'])} | {f(r['sp_s'])} | {pct(r['sp_eff'])} | "
            f"{f(r['ov_s'])} | {pct(r['ov_eff'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
