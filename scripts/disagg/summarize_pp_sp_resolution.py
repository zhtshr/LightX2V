#!/usr/bin/env python3
"""Summarize PP×SP hybrid + quad results across resolutions."""

from __future__ import annotations

import json
import re
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
HYBRID_PAT = re.compile(r"wan22_pp2_sp(\d+)_hybrid_(\d+)x\2_lps(\d+)_m2\.json")
QUAD_PAT = re.compile(r"wan22_pp2_sp(\d+)_quad_(\d+)x\2_lps(\d+)_(oct|dual)\.json")


def f(v, n=1):
    return "—" if v is None else f"{v:.{n}f}"


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def t1_block_offload_p1(res: int) -> float | None:
    """MoE I2V block-offload P=1 baseline (§2.3.0); only runnable single-GPU config on A10."""
    t1_block = {
        256: 6.681,
        512: 37.831,
        1024: 360.566,
        480: 72.645,
    }
    return t1_block.get(res)


def main() -> None:
    rows: list[dict] = []

    for path in sorted(STUDY.glob("wan22_pp2_sp*_hybrid_*x*_lps*_m2.json")):
        m = HYBRID_PAT.match(path.name)
        if not m:
            continue
        sp, res, lps = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d = json.loads(path.read_text())
        dual = next((r for r in d.get("gpipe_results", []) if r.get("num_microbatches") == 2 and r.get("feasible")), None)
        n_gpu = 2 * sp
        t1 = t1_block_offload_p1(res)
        single_s = d.get("single_transformer_s")
        ov_s = dual.get("gpipe_wall_s") if dual else None
        rows.append({
            "mode": f"PP×SP P=2×{sp} hybrid",
            "resolution": res,
            "lps": lps,
            "n_gpu": n_gpu,
            "t1": t1,
            "single_s": single_s,
            "single_eff": (t1 / single_s / n_gpu) if single_s else None,
            "overlap_s": ov_s,
            "overlap_eff": (2 * t1 / (ov_s * n_gpu)) if ov_s else None,
            "n_req": 2,
        })

    for path in sorted(STUDY.glob("wan22_pp2_sp*_quad_*x*_lps*_*.json")):
        m = QUAD_PAT.match(path.name)
        if not m:
            continue
        sp, res, lps, _layout = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        d = json.loads(path.read_text())
        quad = d.get("modes", {}).get("quad_pp_sp_overlap", {})
        n_req = d.get("num_quad_requests", 4)
        n_gpu = 2 * sp
        t1 = t1_block_offload_p1(res)
        single_s = d.get("single_transformer_s")
        ov_s = quad.get("wall_s")
        rows.append({
            "mode": f"PP×SP P=2×{sp} quad",
            "resolution": res,
            "lps": lps,
            "n_gpu": n_gpu,
            "t1": t1,
            "single_s": single_s,
            "single_eff": (t1 / single_s / n_gpu) if single_s else None,
            "overlap_s": ov_s,
            "overlap_eff": (n_req * t1 / (ov_s * n_gpu)) if ov_s else None,
            "n_req": n_req,
        })

    rows.sort(key=lambda r: (r["resolution"], r["mode"]))

    lines = [
        "# Wan2.2-MoE I2V — PP×SP hybrid / quad vs 分辨率",
        "",
        "口径：PP×SP 去噪 `cpu_offload=false`（多卡）；T₁ = **block-offload P=1 实测**（§2.3.0 同口径，单卡唯一可跑基线）。",
        "表中「多卡单请求」= N 卡 PP×SP 单路 wall；单请求效率 = `T₁/(T_multi×N)`；overlap 效率 vs `N/T₁` 理想吞吐。",
        "",
        "| 配置 | 分辨率 | T₁ block P=1 (s) | 多卡单请求 (s) | 单请求效率 | overlap (s) | overlap 效率 | 并发 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['resolution']}×{r['resolution']} | {f(r['t1'])} | {f(r['single_s'])} | {pct(r['single_eff'])} | "
            f"{f(r['overlap_s'])} | {pct(r['overlap_eff'])} | {r['n_req']} req |"
        )

    out_md = STUDY / "phase1_moe_pp_sp_resolution.md"
    out_json = STUDY / "phase1_moe_pp_sp_resolution.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
