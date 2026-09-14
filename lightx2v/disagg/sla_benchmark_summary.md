# SLA 稀疏注意力 & 并行配置性能总览

> 与 [`motivation.md`](motivation.md) 同口径；本文在 **480×832** 在线分辨率下汇总 **稠密 sage** 与 **SLA triton**（`sparsity_ratio=0.8`），以及大模型 **PP / PP×SP** 实测。  
> 稠密多分辨率完整数据见 motivation §1。

**口径**：去噪 = `transformer_compute_s`（`scheduler.prepare` + denoise loop，不含 load/encoder/decoder）。**T₁** = 该配置族 **P=1 单卡**实测（小模型无 offload；大模型 MoE **block-offload SP P=1**）。**单请求扩展效率** = `T₁/(T×N)`（**N** = GPU 数）。**overlap 扩展效率** = `n·T₁/(T_overlap×N)`（`n` = in-flight 请求数）。**—** = 未测。

**× 判定**（同 motivation §1）：有 overlap 时 **overlap wall > T₁** 或 **overlap 效率 < 70%**；无 overlap 时 **单请求效率 < 70%**。

---

## 小模型 Wan2.1-T2V-1.3B（480×832）

无 offload，Ulysses SP，4 steps。稠密 T₁ = **14.9 s**；SLA T₁ = **10.5 s**。

| × | 配置 | attn | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 稠密 | 1 | 14.9 | 100% | — | — | 1 |
| | SP P=1 | SLA | 1 | 10.5 | 100% | — | — | 1 |
| | SP P=2 | 稠密 | 2 | 9.0 | 82.9% | 14.7 | 101.4% | 2 |
| × | SP P=2 | SLA | 2 | 6.8 | 77.6% | 11.9 | 88.7% | 2 |
| | SP P=3 | 稠密 | 3 | 6.5 | 76.3% | 9.9 | 100.3% | 2 |
| × | SP P=3 | SLA | 3 | 5.1 | 69.3% | 7.4 | 95.2% | 2 |
| | SP P=4 | 稠密 | 4 | 5.0 | 74.9% | 7.4 | 100.9% | 2 |
| × | SP P=4 | SLA | 4 | 3.9 | 67.3% | 5.7 | 93.1% | 2 |
| | SP P=6 | 稠密 | 6 | 3.5 | 71.7% | 5.0 | 99.5% | 2 |
| × | SP P=6 | SLA | 6 | 2.7 | 65.2% | 4.0 | 86.7% | 2 |

SLA P=2 overlap wall（11.9 s）> SLA T₁（10.5 s），标 ×。SLA P=3/4/6 单请求效率 < 70% 标 ×。

数据来源：`p1_t2v_1.3b_transformer_seqp*.json`、`p1_t2v_1.3b_sla_transformer_seqp*.json`、`p3_dual_overlap_t2v_1.3b_seqp*.json`、`p3_dual_overlap_t2v_1.3b_sla_seqp*.json`。

---

## 大模型 Wan2.2-MoE I2V（480×832）

4 steps，`enable_cfg=false`，int8-q8f。稠密 T₁ = **73.6 s**（block offload）；SLA T₁ = **54.4 s**（block offload）。**PP / PP×SP** 行 DiT offload = **否**（PP 不支持 block offload）；`pp_layers_per_stage=2`。

| × | 配置 | attn | DiT offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 稠密 | block | 1 | 73.6 | 100% | — | — | 1 |
| | SP P=1 | SLA | block | 1 | 54.4 | 100% | — | — | 1 |
| | SP P=2 | 稠密 | block | 2 | 44.1 | 83.5% | 86.3 | 85.3% | 2 |
| | SP P=2 | SLA | block | 2 | 33.9 | 80.4% | 68.0 | 80.0% | 2 |
| | SP P=4 | 稠密 | block | 4 | 25.3 | 72.6% | 50.5 | 72.9% | 2 |
| × | SP P=4 | SLA | block | 4 | 19.8 | 68.7% | 38.3 | 71.1% | 2 |
| × | SP P=8 | 稠密 | block | 8 | 14.4 | 63.8% | 29.1 | 63.2% | 2 |
| × | SP P=8 | SLA | block | 8 | 11.0 | 61.6% | 24.6 | 55.3% | 2 |
| × | PP P=2 GPipe | 稠密 | 否 | 2 | 70.3 | 52.3% | 77.0 | 95.5% | 2 |
| × | PP P=2 GPipe | SLA | 否 | 2 | 51.1 | 53.2% | 58.3 | 93.4% | 2 |
| × | PP×SP P=2×2 hybrid | 稠密 | 否 | 4 | 42.8 | 43.0% | 46.0 | 79.9% | 2 |
| × | PP×SP P=2×2 hybrid | SLA | 否 | 4 | 33.5 | 40.6% | 36.3 | 75.0% | 2 |
| × | PP×SP P=2×2 quad | 稠密 | 否 | 4 | 42.9 | 42.8% | 84.1 | 87.5% | 4 |
| × | PP×SP P=2×2 quad | SLA | 否 | 4 | 33.5 | 40.6% | 65.0 | 83.7% | 4 |
| × | PP×SP P=2×4 hybrid | 稠密 | 否 | 8 | 24.2 | 38.0% | 24.8 | 74.1% | 2 |
| × | PP×SP P=2×4 hybrid | SLA | 否 | 8 | 19.6 | 34.7% | 20.1 | 67.7% | 2 |
| × | PP×SP P=2×4 quad | 稠密 | 否 | 8 | 24.2 | 38.0% | 42.2 | 87.2% | 4 |
| × | PP×SP P=2×4 quad | SLA | 否 | 8 | 19.6 | 34.7% | 33.1 | 82.3% | 4 |

**读表要点**

| 对比 | 稠密 | SLA | 说明 |
| --- | --- | --- | --- |
| 单卡 T₁ | 73.6 s | 54.4 s（**1.35×**） | SLA 仅稀疏 self-attn |
| SP P=8 单请求 | 14.4 s / 63.8% | 11.0 s / 61.6% | SLA 绝对更快，扩展效率略降 |
| SP dual overlap P=8 | 29.1 s / 63.2% | 24.6 s / **55.3%** | SLA 绝对更快，overlap 效率更差 |
| PP GPipe 2-req | 77.0 s / 95.5% | 58.3 s / **93.4%** | SLA wall 更短，效率略降 2pp |
| PP×SP quad 4-req | 84.1 s / 87.5% | 65.0 s / **83.7%** | SLA 绝对更快，quad 效率降 ~4pp |
| PP×SP quad 8-req | 42.2 s / 87.2% | 33.1 s / **82.3%** | 同上，单请求效率仍 ~35–41% |

PP / PP×SP 单请求效率均 < 70%（标 ×），适合 **多请求吞吐** 而非单请求降延迟；2-req hybrid / 4-req quad 的 overlap 效率 **74–88%**，优于同 GPU 数 SP dual overlap（P=8 仅 63%）。

数据来源：`p1_transformer_seqp*.json`、`p1_moe_i2v_480_sla_triton_seqp*.json`、`p3_dual_overlap_seqp*.json`、`p3_dual_overlap_moe_i2v_480x832_sla_seqp*.json`、`p1_moe_480_pp2_*.json`、`p1_moe_480_sla_pp2_*.json`、`p1_moe_480_pp2_sp{2,4}_*.json`；PP/PP×SP SLA 详见 [`p1_moe_480_pp_sp_sla_summary.md`](../../save_results/optimization_study/p1_moe_480_pp_sp_sla_summary.md)。

### SP 稀疏通信实验（480p MoE SLA P=4，负结果）

在 SLA 已稀疏 **计算** 的前提下，尝试稀疏 **通信**（只传 top-20% K/V block）：

| 路径 | transformer_s | vs 稠密 Ulysses |
| --- | ---: | ---: |
| Ulysses 稠密 a2a | 19.9 s | — |
| Ulysses 稀疏 K/V（orig） | 21.9 s | +10% 慢 |
| Ulysses 稀疏 L1（skip fill + reuse map） | 22.1 s | +11% 慢 |
| Ulysses 稀疏 L2（compact K/V kernel） | **21.4 s** | +7.5% 慢 |
| Ring 稀疏 KV | 32.2 s | +63% 慢 |

L1/L2 优化后 L2 略好，但仍慢于稠密 a2a；详见 [`p1_moe_sla_ulysses_sparse_summary.md`](../../save_results/optimization_study/p1_moe_sla_ulysses_sparse_summary.md)。

---

## 多分辨率 SLA 稀疏（256² / 512² / 1024²）

完整表格见 [`save_results/optimization_study/p1_sla_resolution_summary.md`](../../save_results/optimization_study/p1_sla_resolution_summary.md)。复现：`scripts/disagg/run_phase1_sla_resolution_bench.sh`。

### T₁ 对比（SLA vs 稠密）

| 分辨率 | 小模型稠密 | 小模型 SLA | 加速 | 大模型稠密 | 大模型 SLA | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256² | 1.34 s | 1.23 s | 1.09× | 6.68 s | 6.39 s | 1.05× |
| 512² | 8.00 s | 6.00 s | 1.33× | 37.8 s | 30.5 s | 1.24× |
| 1024² | 86.2 s | 44.5 s | **1.94×** | 360.6 s | 214.6 s | **1.68×** |

SLA 收益随分辨率升高而放大（1024² 小模型近 2×）；256² 小模型差距仅 ~8%。

### 读表要点

| 分辨率 | 小模型 SLA SP | 小模型 SLA overlap | 大模型 SLA SP | 大模型 SLA overlap |
| --- | --- | --- | --- | --- |
| 256² | P=2/6 单请求 ×；P=3–4 可用 | P=6 ×（59%） | 全 P 单请求 ×（offload 瓶颈） | 与稠密同量级，均 × |
| 512² | P=4–6 可用（55–65%） | 85–95%，均可用 | P=4/8 单请求 × | P=4/8 均 ×（42–59%） |
| 1024² | P=2–6 均 >75% | 未测 | P=2–8 均 >75% | P=4 **87%**、P=8 **79%** |

1024² 是大模型 SLA + SP overlap 的甜点：单卡 SLA 省 40%+，P=4 dual overlap 效率仍近 87%。256² 受 block offload 限制，大模型 SP 扩展极差。

---

## SLA 算子变体（P=1，大模型）

| 变体 | 延迟 (s) | vs 稠密 | 状态 |
| --- | ---: | ---: | --- |
| 稠密 sage | 73.6 | 1.00× | ✅ |
| **sla_triton** | **54.4** | **1.35×** | ✅ |
| sla_flex_block / flashinfer / fa4 / nbhd | — | — | ❌ 依赖缺失或环境限制 |

---

## 结论摘要

| 场景 | 推荐 |
| --- | --- |
| 单卡 / 低并行延迟 | ✅ **SLA**（T₁ −35%） |
| 小模型 SP + dual-overlap 吞吐 | 稠密 overlap ~100%；SLA 绝对更快但效率降 5–13pp |
| 大模型 SP P=2–4 + overlap | SLA 可行（绝对 wall 更短，效率差距 2–9pp） |
| 大模型 SP P=8 + overlap | 稠密已 ×；SLA 进一步降至 55% |
| 大模型多租户吞吐（无 offload） | **PP×SP quad**（~87%）> hybrid（~74–80%）> SP dual（63%） |
| 大模型单请求多卡 | **纯 SP**；PP/PP×SP 单请求效率 38–52% |

---

## 复现

| 脚本 | 内容 |
| --- | --- |
| `scripts/disagg/run_phase1_sla_t2v_1.3b_bench.sh` | 小模型 SLA SP + overlap |
| `scripts/disagg/run_phase1_sla_moe_sp_bench.sh` | 大模型 SLA SP |
| `scripts/disagg/run_phase1_sla_sp_overlap_bench.sh` | 大模型 SLA overlap |
| `scripts/disagg/run_phase1_moe_pp_sp_480p_bench.sh` | 大模型 PP / PP×SP |
| `scripts/disagg/run_phase1_moe_pp_sp_sla_480p_bench.sh` | 大模型 SLA PP / PP×SP |
| `scripts/disagg/run_phase1_sla_resolution_bench.sh` | 256/512/1024 SLA SP + overlap |

汇总 JSON：`save_results/optimization_study/p1_t2v_1.3b_sla_summary.json`、`p1_moe_i2v_480_sla_sp_summary.json`、`p1_sla_sp_overlap_summary.json`、`p1_moe_480_pp_sp_summary.json`、`p1_moe_480_pp_sp_sla_summary.json`、`p1_sla_resolution_summary.json`。

---

*生成日期：2026-07-01。数值四舍五入至一位小数。*
