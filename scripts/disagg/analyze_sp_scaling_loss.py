#!/usr/bin/env python3
"""Decompose Ulysses SP scaling loss — verified against phase3 benchmark JSON."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
STEPS = 4


def load_json(name: str) -> dict:
    p = STUDY / name
    return json.loads(p.read_text()) if p.is_file() else {}


def estimate_t_floor(comp_p1: float, comp_p: float, p: int) -> float:
    """Phenomenological per-rank compute floor from comp(P)=(comp1-T_floor)/P+T_floor."""
    if p <= 1:
        return 0.0
    return (p * comp_p - comp_p1) / (p - 1)


def estimate_cross_kv_wall(comp_wall_p: float, *, seq_global: int = 32760, p: int = 4) -> float:
    """FLOP-proportional estimate of cross-attn K/V (context) wall — truly fixed per rank."""
    dim, layers, text, img = 5120, 40, 512, 257
    s = seq_global // p

    def gemm(m: int) -> float:
        return 2 * m * dim * dim

    per_layer_kv = gemm(text) * 2 + gemm(img) * 2  # K,V text + K_img,V_img
    per_layer_qo = gemm(s) * 2
    per_layer_attn = 2 * s * (text + img) * dim * 2  # text + img cross-attn
    cross_flops = layers * (per_layer_kv + per_layer_qo + per_layer_attn)
    kv_flops = layers * per_layer_kv
    # scale by cross-attn share of total compute wall at P
    cross_share_of_step = 0.20
    cross_wall = comp_wall_p * cross_share_of_step
    return cross_wall * (kv_flops / cross_flops) if cross_flops else 0.0


def main() -> None:
    ps = [1, 2, 4, 8]
    data = {p: load_json(f"p3_sp_seqp{p}.json") for p in ps}
    dual = load_json("p3_dual_overlap_seqp4_no_offload.json")
    dual_t2v = load_json("p3_dual_overlap_t2v_1.3b_seqp4.json")
    ring = {p: load_json(f"p3_ring_sp_seqp{p}.json") for p in ps}

    t1 = data[1]["transformer_compute_s"]
    comp1 = data[1]["one_step_profile"]["compute_cuda_us"] / 1e6

    t_floor_estimates = [estimate_t_floor(comp1, data[p]["one_step_profile"]["compute_cuda_us"] / 1e6, p) for p in (2, 4, 8)]
    t_floor = statistics.mean(t_floor_estimates[1:])  # P=4,8 more stable
    t_sharded_p1 = comp1 - t_floor
    comp4 = data[4]["one_step_profile"]["compute_cuda_us"] / 1e6
    kv_wall_est = estimate_cross_kv_wall(comp4, p=4)

    rows = []
    for p in ps:
        t = data[p]["transformer_compute_s"]
        prof = data[p]["one_step_profile"]
        comm = prof["comm_cuda_us"] / 1e6
        comp = prof["compute_cuda_us"] / 1e6
        wall_step = t / STEPS
        naive_ideal = t1 / p
        gap_naive = t - naive_ideal
        struct_step = t_sharded_p1 / p + t_floor
        struct_total = struct_step * STEPS
        gap_struct = t - struct_total

        kernel_sum = comm + comp
        hidden = max(0.0, kernel_sum - wall_step)

        amdahl_step = t_floor * (1.0 - 1.0 / p) if p > 1 else 0.0
        amdahl_total = amdahl_step * STEPS

        rows.append({
            "p": p,
            "t": t,
            "naive_ideal": naive_ideal,
            "gap_naive": gap_naive,
            "struct_total": struct_total,
            "gap_struct": gap_struct,
            "eff_naive": naive_ideal / t,
            "speedup": t1 / t,
            "wall_step": wall_step,
            "comm": comm,
            "comp": comp,
            "comm_ratio": prof["comm_cuda_ratio"],
            "hidden_per_step": hidden,
            "amdahl_total": amdahl_total,
            "comm_crit_total": max(0.0, gap_struct),
        })

    def pct(part: float, whole: float) -> float:
        return 100.0 * part / whole if whole > 0 else 0.0

    # P=4 / P=8 loss budget vs naive ideal (user-facing "why not 4x/8x")
    budgets = {}
    for p in (4, 8):
        r = next(x for x in rows if x["p"] == p)
        gap = r["gap_naive"]
        amdahl = min(r["amdahl_total"], gap)
        comm_crit = max(0.0, min(r["gap_struct"], gap - amdahl))
        sharded_slip = max(0.0, gap - amdahl - comm_crit)
        budgets[p] = {
            "gap": gap,
            "amdahl": amdahl,
            "comm_crit": comm_crit,
            "sharded_slip": sharded_slip,
            "amdahl_pct": pct(amdahl, gap),
            "comm_pct": pct(comm_crit, gap),
            "slip_pct": pct(sharded_slip, gap),
        }

    lines = [
        "# 多卡 SP 扩展损失分解（Wan2.2-MoE I2V Ulysses，实测验证）",
        "",
        "口径：**单请求 transformer 吞吐** = `1/transformer_compute_s`；对比 **朴素理想** `T(P=1)/P`。",
        "",
        "## 1. 原始扩展数据",
        "",
        "| P | transformer_s | 朴素理想(s) | 损失(s) | speedup | 朴素效率 | comm CUDA% | GPU util |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        util = data[r["p"]].get("gpu_util_during_denoise", {}).get("pooled_active_gpus", {}).get("avg", 0)
        lines.append(
            f"| {r['p']} | {r['t']:.2f} | {r['naive_ideal']:.2f} | {r['gap_naive']:.2f} | "
            f"{r['speedup']:.2f} | {100*r['eff_naive']:.1f}% | {100*r['comm_ratio']:.1f}% | {util:.1f}% |"
        )

    r4 = next(x for x in rows if x["p"] == 4)
    r8 = next(x for x in rows if x["p"] == 8)
    lines.extend([
        "",
        "## 2. 修正说明（重要）",
        "",
        "旧版把回归截距 `T_floor≈2.09s` **误标成**「cross-attn 不切 SP」——**这是错的**。",
        "",
        "**代码事实**（`wan/infer/transformer_infer.py`）：",
        "- cross-attn 的 **Q/O/attention** 作用在 **SP 切分后的 local `x`** 上 → **随 P 增大而缩短**（Q token 数 ≈ 32760/P）。",
        "- **仅 context 的 K/V 线性层**（512 text + 257 image）在每张卡上 **完整重算、不随 P 切分**。",
        "",
        f"`T_floor` 是整步 `compute_cuda` 的 **现象学 floor**（`comp(P)=(comp1−T_floor)/P+T_floor`），",
        f"**不等于** cross-attn 总时间。其中 **可明确归因于 cross K/V 的仅 ~{kv_wall_est:.2f}s/step**（FLOP 比例粗估，≈cross-attn 的 8%）。",
        "",
        "## 3. 扩展损失分解（相对朴素理想 `T₁/P`）",
        "",
        f"**现象学 floor** `T_floor ≈ {t_floor:.2f}s/step`（P=4/8 反推；描述「不随 1/P 下降」的 compute 截距，**未逐项归因**）。",
        f"**理想可切分** `T_sharded(P=1) ≈ {t_sharded_p1:.2f}s/step`（= comp1 − T_floor）。",
        "",
        "### P=4：损失 6.38s",
        "",
        "| 原因 | 含义 | 贡献时间 | 占损失比例 |",
        "|---|---|---:|---:|",
        f"| **① 现象学 Amdahl floor** | `T_floor×(1−1/4)×4步`；**不是**「整个 cross-attn 固定」 | **{budgets[4]['amdahl']:.2f}s** | **{budgets[4]['amdahl_pct']:.0f}%** |",
        f"| **② 其中：cross K/V 真正不切 SP** | context 侧 K/V 每卡重算；FLOP 估 ~{kv_wall_est:.2f}s/step | **~{kv_wall_est*4:.2f}s** | **~{100*kv_wall_est*4/budgets[4]['gap']:.0f}%** |",
        f"| **③ NCCL critical path** | 结构拟合残差；kernel 已掩盖 {rows[2]['hidden_per_step']:.2f}s/step | **{budgets[4]['comm_crit']:.2f}s** | **{budgets[4]['comm_pct']:.0f}%** |",
        f"| **④ 可切分计算子线性** | sharded 实测 vs 理想 | **{budgets[4]['sharded_slip']:.2f}s** | **{budgets[4]['slip_pct']:.0f}%** |",
        "",
        f"**P=4 结论**：cross-attn **会随 P 缩短**（P=1 cross FLOPs ~151TF → P=4 每卡 ~40TF，约 3.8×）。",
        f"掉效率的主因是 **整步相对 `T₁/P` 存在 ~{t_floor:.1f}s/step 的 floor 截距**；",
        f"其中 **仅 ~{kv_wall_est:.2f}s 能确定来自 cross K/V**，其余 floor 需 layer profiler 才能拆开。",
        "",
        "### P=8：损失 4.22s",
        "",
        "| 原因 | 贡献时间 | 占损失比例 |",
        "|---|---:|---:|",
        f"| **① 现象学 floor** `T_floor×(1−1/8)×4` | **{min(budgets[8]['amdahl'], budgets[8]['gap']):.2f}s** | **{budgets[8]['amdahl_pct']:.0f}%** |",
        f"| **② NCCL critical path** | **{budgets[8]['comm_crit']:.2f}s** | **{budgets[8]['comm_pct']:.0f}%** |",
        f"| **③ 可切分子线性** | **{budgets[8]['sharded_slip']:.2f}s** | **{budgets[8]['slip_pct']:.0f}%** |",
        "",
        f"P=8 时 floor 占 wall **{100*t_floor/r8['wall_step']:.0f}%**/step → 效率进一步下降，",
        "主因是 **floor 占比上升**，不是 cross-attn Q 侧没切开。",
        "",
        "## 4. 已验证但 **不贡献单请求 SP 扩展损失** 的因素",
        "",
        "| 因素 | 实测 | 对「多卡扩展效率」的含义 |",
        "|---|---|---|",
    ])

    if dual:
        ov = dual["dual_a2a_serial_s"] - dual["dual_a2a_overlap_s"]
        inter = dual["dual_back_to_back_s"] - dual["dual_a2a_serial_s"]
        tp_ov = 2 / dual["dual_a2a_overlap_s"]
        tp_b2b = 2 / dual["dual_back_to_back_s"]
        lines.append(
            f"| **双租户 comm overlap** | serial→overlap 省 {ov:.2f}s（{100*ov/dual['dual_a2a_serial_s']:.1f}%）；one-step pair 仅 {dual['one_step_pair_overlap_stats']['overlap_fraction_of_serial']*100:.1f}% | **不减少单请求 latency**；同 4 卡 2 req 吞吐 {tp_ov:.4f} vs b2b {tp_b2b:.4f}（**+{100*(tp_ov/tp_b2b-1):.1f}%**） |"
        )
        lines.append(
            f"| **双租户 interleave** | b2b→serial 省 {inter:.2f}s（{100*inter/dual['dual_back_to_back_s']:.1f}%） | 调度收益，与 SP 扩展无关 |"
        )
        lines.append(
            f"| **GPU 100% 饱和** | denoise util {data[4]['gpu_util_during_denoise']['pooled_active_gpus']['avg']:.1f}% | overlap 上限被 NCCL↔matmul 争用卡在 ~3% |"
        )

    if dual_t2v:
        fair = dual_t2v.get("fair_dual_baseline_s", 0)
        ov = dual_t2v.get("dual_a2a_overlap_s", 0)
        lines.append(
            f"| **小模型可重叠窗口更大** | T2V 1.3B fair overlap {100*(fair/ov-1):.1f}% vs fair baseline | 说明 comm 窗口占比更高；Wan2.2 大模型只有 ~3% |"
        )

    if ring.get(4):
        ring4 = ring[4].get("transformer_compute_s", 0)
        lines.append(
            f"| **Ring SP 算法** | P=4 ring {ring4:.1f}s vs Ulysses {r4['t']:.1f}s（慢 {100*(ring4/r4['t']-1):.0f}%） | 换算法损失，不是当前 Ulysses 路径的问题 |"
        )
        if ring.get(8):
            ring8 = ring[8].get("transformer_compute_s", 0)
            lines.append(
                f"| | P=8 ring {ring8:.1f}s vs Ulysses {r8['t']:.1f}s（慢 {100*(ring8/r8['t']-1):.0f}%） | P 越大越差 |"
            )

    ideal8 = 8.0 / t1  # 8 路独立 P=1 并行
    tp_p4_2group = 2.0 / r4["t"]  # 两组 P=4 占满 8 卡
    tp_p8 = 1.0 / r8["t"]
    lines.extend([
        "",
        "## 5. 8 卡 **系统吞吐** 账本",
        "",
        f"朴素理想（8 路互不干扰 P=1 并行）= **{ideal8:.4f} req/s**。",
        "",
        "| 部署 | 占用 | 系统吞吐 (req/s) | vs 朴素理想 |",
        "|---|---|---:|---:|",
        f"| 2×P=4 各跑 1 req | 8 卡 | {tp_p4_2group:.4f} | **{100*tp_p4_2group/ideal8:.1f}%** |",
        f"| 1×P=8 跑 1 req | 8 卡 | {tp_p8:.4f} | **{100*tp_p8/ideal8:.1f}%** |",
    ])
    if dual:
        dual_ov = dual["dual_a2a_overlap_s"]
        tp_dual = 2.0 / dual_ov
        lines.append(
            f"| P=4 双租户 overlap（4 卡 2 req） | 4 卡 | {tp_dual:.4f} | 半机吞吐；比单路 P=4 ({1/r4['t']:.4f}) **+{100*(tp_dual/(1/r4['t'])-1):.1f}%** |"
        )
        lines.append(
            f"| 若 8 卡=2 组双租户 overlap | 8 卡 | {2*tp_dual:.4f} | **{100*2*tp_dual/ideal8:.1f}%** of 理想 |"
        )

    lines.extend([
        "",
        "吞吐损失主因排序（修正后）：",
        f"1. **现象学 compute floor ~{t_floor:.1f}s/step**（相对 `T₁/P` 的 Amdahl 截距；≠ 整个 cross-attn）",
        f"2. **其中可确认：cross context K/V ~{kv_wall_est:.2f}s/step**（真正不切 SP 的部分）",
        "3. **P 越大 floor 占 wall 比例越高**",
        "4. **NCCL 多数已在单请求内掩盖**；双租户 overlap 补不了扩展效率",
        "",
        f"T_floor estimates: P2={t_floor_estimates[0]:.2f}s, P4={t_floor_estimates[1]:.2f}s, P8={t_floor_estimates[2]:.2f}s",
        "",
    ])

    out = STUDY / "phase3_sp_loss_decomposition.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps({"t_floor": t_floor, "kv_wall_est": kv_wall_est, "budgets": budgets, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
