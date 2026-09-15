#!/usr/bin/env python3
"""Summarize Wan2.2-MoE I2V SP scaling across resolutions."""

from __future__ import annotations

import json
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
RESOLUTIONS = [256, 512, 1024, 2048]
SEQ_PS = [1, 2, 4, 8]
VAE_STRIDE = (4, 8, 8)
PATCH = (1, 2, 2)


def seq_tokens(res: int) -> int:
    t = (81 - 1) // VAE_STRIDE[0] + 1
    lh, lw = res // VAE_STRIDE[1], res // VAE_STRIDE[2]
    return (t // PATCH[0]) * (lh // PATCH[1]) * (lw // PATCH[2])


def load(res: int, p: int) -> dict | None:
    path = STUDY / f"p1_moe_i2v_{res}x{res}_seqp{p}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    lines = [
        "# Wan2.2-MoE I2V — SP 扩展效率 vs 分辨率",
        "",
        "口径：`cpu_offload=true`（block），Ulysses SP，4 denoise steps，`enable_cfg=false`。",
        "模型：`num_heads=40` → 有效 P ∈ {1,2,4,5,8,10,20,40}。",
        "",
        "## 序列长度（latent tokens）",
        "",
        "| 分辨率 | seq tokens |",
        "|---|---:|",
    ]
    for res in RESOLUTIONS:
        lines.append(f"| {res}×{res} | {seq_tokens(res)} |")

    lines.extend(["", "## 各分辨率 SP 扩展", ""])
    summary_rows: list[dict] = []
    full_matrix: dict[str, dict] = {}

    for res in RESOLUTIONS:
        base_t = None
        rows = []
        res_key = f"{res}x{res}"
        full_matrix[res_key] = {}
        for p in SEQ_PS:
            d = load(res, p)
            full_matrix[res_key][str(p)] = d
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
            elif r["t"] is not None and r["speedup"] is not None and r["eff"] is not None:
                lines.append(
                    f"| {r['p']} | {r['t']:.2f}s | {r['speedup']:.2f}× | {r['eff'] * 100:.1f}% |"
                )
            elif r["t"] is not None:
                lines.append(f"| {r['p']} | {r['t']:.2f}s | — | — |")
            else:
                lines.append(f"| {r['p']} | — | — | — |")
        lines.append("")

        summary_rows.append({
            "res": res,
            "seq": seq_tokens(res),
            "t1": base_t,
            "eff": {r["p"]: r["eff"] for r in rows if r["eff"] is not None},
        })

    lines.extend([
        "## 跨分辨率扩展效率（P=1/2/4/8）",
        "",
        "| 分辨率 | seq | P=1 wall | P=2 eff | P=4 eff | P=8 eff |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        t1 = f"{row['t1']:.2f}s" if row["t1"] else "—"
        eff = row["eff"]
        cols = [f"{eff[p] * 100:.1f}%" if p in eff else "—" for p in SEQ_PS[1:]]
        lines.append(f"| {row['res']}×{row['res']} | {row['seq']} | {t1} | {cols[0]} | {cols[1]} | {cols[2]} |")

    lines.extend([
        "",
        "## 解读提示",
        "",
        "- **小分辨率**：compute 窗口短，offload + comm 占比高 → 扩展效率通常更低。",
        "- **大分辨率**：attention compute 更重 → 扩展效率通常更高。",
        "- 失败/OOM 见 `${STUDY}/p1_moe_i2v_{res}x{res}_seqp{p}.log`。",
        "",
        "Run: `FORCE_RERUN=1 bash scripts/disagg/run_phase1_moe_resolution_sp_sweep.sh`",
    ])

    out = STUDY / "phase1_moe_i2v_resolution_sp_scaling.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)

    jout = STUDY / "phase1_moe_i2v_resolution_sp_scaling.json"
    payload = {
        "resolutions": RESOLUTIONS,
        "seq_ps": SEQ_PS,
        "summary": summary_rows,
        **full_matrix,
    }
    jout.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(jout)


if __name__ == "__main__":
    main()
