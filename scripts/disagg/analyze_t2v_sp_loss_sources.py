#!/usr/bin/env python3
"""Wan2.1-T2V-1.3B P=4 SP loss source breakdown."""

from __future__ import annotations

import json
from pathlib import Path

STUDY = Path("/root/zht/LightX2V/save_results/optimization_study")
STEPS = 4
P4 = 4
SEQ_GLOBAL = 32760
DIM, FFN, LAYERS, TEXT = 1536, 8960, 30, 512
PREFIX = "p3_t2v_1.3b_sp"
P1_PREFIX = "p1_t2v_1.3b_transformer"


def load_p3(p: int) -> dict:
    return json.loads((STUDY / f"{PREFIX}_seqp{p}.json").read_text())


def load_p1(p: int) -> dict:
    return json.loads((STUDY / f"{P1_PREFIX}_seqp{p}.json").read_text())


def gemm(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def component_flops(seq_local: int, *, t2v: bool = True) -> dict[str, float]:
    """Per-rank FLOPs per step."""
    cross_kv = LAYERS * gemm(TEXT, DIM, DIM) * 2  # K,V text only (no I2V image path)
    cross_qos = LAYERS * (
        gemm(seq_local, DIM, DIM) * 2  # Q + O
        + 2 * seq_local * TEXT * DIM  # one text cross-attn
    )
    self_attn = LAYERS * (gemm(seq_local, DIM, DIM) * 4 + 2 * seq_local * seq_local * DIM)
    ffn = LAYERS * (gemm(seq_local, FFN, DIM) + gemm(seq_local, DIM, FFN))
    head = gemm(seq_local, DIM, DIM)
    return {
        "cross_kv": cross_kv,
        "cross_qos": cross_qos,
        "self_attn": self_attn,
        "ffn": ffn,
        "head": head,
    }


def flops_to_time(flops: dict[str, float], total_comp_s: float) -> dict[str, float]:
    s = sum(flops.values())
    return {k: total_comp_s * v / s for k, v in flops.items()}


def main() -> None:
    ps = [1, 2, 4]
    data = {p: load_p3(p) for p in ps if (STUDY / f"{PREFIX}_seqp{p}.json").is_file()}
    wall = {p: data[p]["transformer_compute_s"] / STEPS for p in data}
    comp = {p: data[p]["one_step_profile"]["compute_cuda_us"] / 1e6 for p in data}
    comm = {p: data[p]["one_step_profile"]["comm_cuda_us"] / 1e6 for p in data}

    gap_step = wall[P4] - wall[1] / P4
    gap_total = gap_step * STEPS

    f1 = component_flops(SEQ_GLOBAL)
    f4 = component_flops(SEQ_GLOBAL // P4)
    t1_parts = flops_to_time(f1, comp[1])
    t4_parts = flops_to_time(f4, comp[P4])

    labels = {
        "self_attn": "self-attn（Ulysses）",
        "cross_qos": "cross Q/O/attn（text）",
        "cross_kv": "cross K/V（text context，固定）",
        "ffn": "FFN",
        "head": "head/norm",
    }
    module_excess: list[tuple[str, float, float, float]] = []
    for k in ("self_attn", "cross_qos", "cross_kv", "ffn", "head"):
        ideal = t1_parts[k] / P4
        actual = t4_parts[k]
        module_excess.append((labels[k], ideal, actual, actual - ideal))

    named_loss = sum(max(0.0, e) for _, _, _, e in module_excess)
    named_gain = sum(max(0.0, -e) for _, _, _, e in module_excess)
    residual = gap_step - named_loss + named_gain

    hidden_p4 = max(0.0, comp[P4] + comm[P4] - wall[P4])
    cross_kv_excess = next(e for n, _, _, e in module_excess if "K/V" in n)

    ideal_comp_p4 = comp[1] / P4
    comm_critical = wall[P4] - comp[P4]  # comm not fully hidden on critical path
    compute_delta = comp[P4] - ideal_comp_p4

    lines = [
        "# Wan2.1-T2V-1.3B — P=4 性能损失来源",
        "",
        "口径：`cpu_offload=false`，Ulysses SP，4 denoise steps，480×832×81。",
        "数据：`p3_t2v_1.3b_sp_seqp{1,4}.json` + FLOP 校准到 `compute_cuda`。",
        "",
        "## 1. 扩展结果（对比）",
        "",
        "| 模型 | P=1 | P=4 | speedup | 效率 | comm% @P=4 | offload |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| **T2V-1.3B** | {wall[1]*STEPS:.2f}s | {wall[P4]*STEPS:.2f}s | {wall[1]/wall[P4]:.2f}× | {100*(wall[1]/wall[P4])/P4:.1f}% | {100*data[P4]['one_step_profile']['comm_cuda_ratio']:.1f}% | 无 |",
        "| Wan2.2-MoE I2V | 72.65s | 24.54s | 2.96× | 74.0% | 35.4% | block |",
        "",
        f"T2V P=4 每步 wall：**{wall[P4]:.3f}s**；朴素理想 **{wall[1]/P4:.3f}s**；**损失 {gap_step:.3f}s/step**（4 步 **{gap_total:.2f}s**）。",
        "",
        "### P=1/2/4 扩展曲线",
        "",
        "| P | wall/step | 理想 T₁/P | 损失 | compute | comm | comm% | overlap 估 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(data):
        w = wall[p]
        c = comp[p]
        m = comm[p]
        ov = max(0.0, c + m - w)
        ratio = 100 * data[p]["one_step_profile"]["comm_cuda_ratio"] if p > 1 else 0.0
        loss = w - wall[1] / p
        lines.append(
            f"| {p} | {w:.3f}s | {wall[1]/p:.3f}s | {loss:+.3f}s | {c:.3f}s | {m:.3f}s | {ratio:.1f}% | {ov:.3f}s |"
        )
    lines.extend([
        "",
        "## 2. Wall 时间分解（P=4 主因）",
        "",
        "朴素理想 `T₁/4` 假设 **零 comm + 线性 compute**。T2V 小模型 compute 其实接近 4×，**损失主要来自 NCCL 无法完全 overlap 的关键路径**。",
        "",
        "| 来源 | P=4 值 | vs 理想 T₁/4 | 占 gap |",
        "|---|---:|---:|---:|",
        f"| compute_cuda | {comp[P4]:.3f}s | {compute_delta:+.3f}s（{'优于' if compute_delta < 0 else '劣于'}线性） | {100*compute_delta/gap_step:+.0f}% |",
        f"| comm 关键路径（wall − compute） | {comm_critical:.3f}s | **+{comm_critical:.3f}s** | **{100*comm_critical/gap_step:.0f}%** |",
        f"| overlap 节省（comp+comm−wall） | {hidden_p4:.3f}s | — | — |",
        f"| **净损失** | {wall[P4]:.3f}s | **+{gap_step:.3f}s** | 100% |",
        "",
        f"校验：`{compute_delta:+.3f}s + {comm_critical:.3f}s = {compute_delta + comm_critical:.3f}s ≈ {gap_step:.3f}s`。",
        "",
        "## 3. 固定项 vs 可切分项（T2V 特有）",
        "",
        "T2V **没有** I2V 的 image cross-attn（257 token）；固定项只有 **text context K/V（512 token）**。",
        "",
        "| 类型 | 内容 | P=4 估时 |",
        "|---|---|---:|",
        f"| **固定** | `cross_attn_k/v(context)` 每卡全算，30 层 | **~{t4_parts['cross_kv']:.3f}s/step** |",
        f"| **可切分** | cross Q/O/attn、self-attn、FFN（local S={SEQ_GLOBAL//P4}） | 其余 ~{sum(t4_parts[k] for k in ('cross_qos','self_attn','ffn','head')):.2f}s/step |",
        "",
        f"cross-attn 整体：P=1 **{t1_parts['cross_kv']+t1_parts['cross_qos']:.2f}s** → P=4 **{t4_parts['cross_kv']+t4_parts['cross_qos']:.2f}s**（约 **{(t1_parts['cross_kv']+t1_parts['cross_qos'])/(t4_parts['cross_kv']+t4_parts['cross_qos']):.1f}×** 更快）。",
        "",
        "## 4. Compute 预算内模块分解（FLOP 校准，理想 = P=1 compute÷4）",
        "",
        "说明：这是在 **{comp:.3f}s compute** 内部分配，不等于 wall gap；self-attn 因 Ulysses 通信特性，FLOP 模型会低估其加速。".format(
            comp=comp[P4]
        ),
        "",
        "| 模块 | P=1 compute | 理想÷4 | P=4 compute | 超额 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, ideal, actual, excess in module_excess:
        ex_s = f"**{excess:+.3f}s**" if abs(excess) > 0.01 else f"{excess:+.3f}s"
        lines.append(
            f"| {name} | {ideal*P4:.3f}s | {ideal:.3f}s | {actual:.3f}s | {ex_s} |"
        )

    lines.extend([
        "",
        f"compute 内：正超额 **{named_loss:.3f}s**，抵消 **−{named_gain:.3f}s**（self-attn Ulysses 超线性）；与 wall gap 残差 **{residual:.3f}s** 对应 §2 的 comm 关键路径。",
        "",
        "## 5. 主因解释",
        "",
        f"1. **NCCL 关键路径（+{comm_critical:.3f}s，~{100*comm_critical/gap_step:.0f}% gap）**：P=4 comm **{comm[P4]:.3f}s/step（{100*data[P4]['one_step_profile']['comm_cuda_ratio']:.1f}% CUDA）**，overlap 仅 ~**{hidden_p4:.3f}s**，仍有 **{comm_critical:.3f}s** 落在 wall 上。小模型 compute 变短后 comm 占比从 P=2 的 32% 升到 **44%**，是 T2V SP 与 MoE 的最大差异。",
        f"2. **Compute 接近线性（{compute_delta:+.3f}s）**：`compute_cuda` **{comp[P4]:.3f}s** 略优于理想 **{ideal_comp_p4:.3f}s**；FFN/cross 在 compute 预算内未满 4×，但被 self-attn 超线性抵消。",
        f"3. **固定 K/V（+{max(0,cross_kv_excess):.3f}s）**：text 512 token 每卡重复，T2V 无 image path，可忽略（<2% gap）。",
        "",
        "## 6. 与 Wan2.2-MoE I2V 对比（P=4）",
        "",
        "| 维度 | T2V-1.3B | MoE I2V |",
        "|---|---|---|",
        f"| 每步损失 vs T₁/4 | **{gap_step:.3f}s** | **1.59s** |",
        f"| 扩展效率 | **{100*(wall[1]/wall[P4])/P4:.1f}%** | 74.0% |",
        f"| comm CUDA% | **{100*data[P4]['one_step_profile']['comm_cuda_ratio']:.1f}%** | 35.4% |",
        f"| cross K/V 固定超额 | **~{max(0,cross_kv_excess):.3f}s** | ~0.07s |",
        "| 损失主因 | **NCCL 关键路径（44% comm）** | FFN + cross Q/O sub-linear |",
        "| P=8 | **不可用**（num_heads=12） | 5.46× |",
        "",
        "## 7. 结论",
        "",
        "1. **P=4 损失 ~0.31s/step 的主因是 NCCL**（~{comm_pct:.0f}% gap）：compute 已接近 4×，但 comm 无法完全藏进 overlap。".format(
            comm_pct=100 * comm_critical / gap_step,
        ),
        "2. **真固定项极小**：text K/V ~{kv:.3f}s（无 I2V image 257 token）。".format(kv=max(0, cross_kv_excess)),
        "3. **效率 ~75% 与 MoE ~74% 相近**，但机理不同：MoE 是大 compute 子线性；T2V 是 **小 compute + 高 comm 占比** 拖慢 wall。",
        "4. **P=8 不可行**（`num_heads=12`，需 `seq_p | 12`）；有效 P ∈ {{1,2,3,4,6,12}}。",
        "",
    ])

    out = STUDY / "phase3_t2v_1.3b_sp_loss_sources.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps({
        "gap_step": gap_step,
        "module_excess": {n: e for n, _, _, e in module_excess},
        "t1_parts": t1_parts,
        "t4_parts": t4_parts,
    }, indent=2))


if __name__ == "__main__":
    main()
