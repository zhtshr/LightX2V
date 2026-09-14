#!/usr/bin/env python3
"""Summarize T2V-1.3B SP scaling across resolutions."""

from __future__ import annotations

import json
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
RESOLUTIONS = [256, 512, 1024, 2048]
SEQ_PS = [1, 2, 3, 4, 6]
VAE_STRIDE = (4, 8, 8)
PATCH = (1, 2, 2)


def seq_tokens(res: int) -> int:
    t = (81 - 1) // VAE_STRIDE[0] + 1
    lh, lw = res // VAE_STRIDE[1], res // VAE_STRIDE[2]
    return (t // PATCH[0]) * (lh // PATCH[1]) * (lw // PATCH[2])


def load(res: int, p: int) -> dict | None:
    path = STUDY / f"p1_t2v_1.3b_{res}x{res}_seqp{p}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    lines = [
        "# Wan2.1-T2V-1.3B — SP 扩展效率 vs 分辨率",
        "",
        "口径：`cpu_offload=false`，Ulysses SP，4 denoise steps，`enable_cfg=false`。",
        "模型：`num_heads=12` → 有效 P ∈ {1,2,3,4,6,12}。",
        "",
        "## 序列长度（latent tokens）",
        "",
        "| 分辨率 | seq tokens | P=6 local |",
        "|---|---:|---:|",
    ]
    for res in RESOLUTIONS:
        s = seq_tokens(res)
        lines.append(f"| {res}×{res} | {s} | {s // 6} |")

    lines.extend(["", "## 各分辨率 SP 扩展", ""])

    summary_rows: list[dict] = []

    for res in RESOLUTIONS:
        base_t = None
        rows = []
        for p in SEQ_PS:
            d = load(res, p)
            t = d.get("transformer_compute_s") if d else None
            err = d.get("error") if d else None
            if p == 1:
                base_t = t
            speedup = (base_t / t) if (base_t and t) else None
            eff = (speedup / p) if speedup else None
            rows.append({"p": p, "t": t, "speedup": speedup, "eff": eff, "err": err})

        lines.append(f"### {res}×{res} (seq={seq_tokens(res)})")
        lines.append("")
        lines.append("| seq_p | transformer_s | speedup | scaling eff |")
        lines.append("|---:|---:|---:|---:|")
        for r in rows:
            if r["err"]:
                lines.append(f"| {r['p']} | FAILED | — | — |")
            elif r["t"] is not None and r["speedup"] is not None:
                lines.append(
                    f"| {r['p']} | {r['t']:.3f}s | {r['speedup']:.2f}× | {r['eff'] * 100:.1f}% |"
                )
            else:
                lines.append(f"| {r['p']} | — | — | — |")
        lines.append("")

        # P=4 and P=6 efficiency for cross-res comparison
        eff_p4 = next((r["eff"] for r in rows if r["p"] == 4 and r["eff"]), None)
        eff_p6 = next((r["eff"] for r in rows if r["p"] == 6 and r["eff"]), None)
        summary_rows.append({
            "res": res,
            "seq": seq_tokens(res),
            "t1": base_t,
            "eff_p4": eff_p4,
            "eff_p6": eff_p6,
        })

    lines.extend([
        "## 跨分辨率对比（扩展效率）",
        "",
        "| 分辨率 | seq | P=1 wall | P=4 eff | P=6 eff | 趋势 |",
        "|---|---:|---:|---:|---:|---|",
    ])

    ref_eff4 = summary_rows[0].get("eff_p4") if summary_rows else None
    for row in summary_rows:
        t1 = f"{row['t1']:.2f}s" if row["t1"] else "—"
        e4 = f"{row['eff_p4'] * 100:.1f}%" if row["eff_p4"] else "—"
        e6 = f"{row['eff_p6'] * 100:.1f}%" if row["eff_p6"] else "—"
        trend = ""
        if row["eff_p4"] and ref_eff4 and row["res"] != RESOLUTIONS[0]:
            if row["eff_p4"] > ref_eff4 + 0.02:
                trend = "eff ↑ vs 256"
            elif row["eff_p4"] < ref_eff4 - 0.02:
                trend = "eff ↓ vs 256"
            else:
                trend = "≈ flat"
        lines.append(
            f"| {row['res']}×{row['res']} | {row['seq']} | {t1} | {e4} | {e6} | {trend} |"
        )

    lines.extend([
        "",
        "## 解读提示",
        "",
        "- **小分辨率**：compute 窗口短，comm 占比高 → 扩展效率通常更低。",
        "- **大分辨率**：compute 更重，comm 相对易隐藏 → 扩展效率通常更高。",
        "- 若某分辨率 OOM/失败，检查 `${STUDY}/p1_t2v_1.3b_{res}x{res}_seqp{p}.log`。",
        "",
        f"Run: `FORCE_RERUN=1 bash scripts/disagg/run_phase1_t2v_resolution_sp_sweep.sh`",
    ])

    out = STUDY / "phase1_t2v_1.3b_resolution_sp_scaling.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)

    # JSON for programmatic use
    jout = STUDY / "phase1_t2v_1.3b_resolution_sp_scaling.json"
    payload = {"resolutions": RESOLUTIONS, "seq_ps": SEQ_PS, "summary": summary_rows}
    for res in RESOLUTIONS:
        payload[f"{res}x{res}"] = {
            str(p): load(res, p) for p in SEQ_PS
        }
    jout.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
