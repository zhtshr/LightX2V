# Self-Forcing / AR 视频生成优化总结

> 范围：Wan Self-Forcing（AR chunkwise denoise + KV cache）推理路径  
> 主要模型：Wan2.1-SF **14B INT8**（KIVI KV）、对照 **1.3B**  
> 硬件：NVIDIA A10；多卡默认 `CUDA_VISIBLE_DEVICES=0,2,4,5`（避开故障卡 1/3）  
> 更新：2026-07-22（Ulysses 通信公平口径：含当前 chunk K/V a2a）

---

## 0. 结论速览

| # | 优化点 | 适用场景 | 实测收益 | 状态 |
|---|--------|----------|----------|------|
| 1 | **kv_offload + 只在 rerun 写回 KV** | 长视频（显存不够放全量 KV） | 10s：88.8s → **79.4s**（相对「每步写回」）；相对无 offload 仍慢约 **11%**（换显存） | **已定稿** |
| 2 | **Stripe / Hybrid SP** | 长 KV + 小 P；或 P≥4 用 Hybrid | 14B P2 30s Stripe ~6%；**14B P4 30s Stripe/Hybrid ~577s < Uly 585s** | **场景启用** |
| 3 | **KIVI KV 量化** | 长视频显存 | 峰值约少 ~1–3.5 GB；速度有税（见 §3） | **已启用（配合 offload）** |
| 4 | **Ulysses SP 扩展** | 短/中视频、PCIe 多卡默认 | 1.3B P=1→4：17.1s → **6.3s** | **默认并行** |
| 5 | **双请求 a2a overlap** | 多租户吞吐 | P=4：b2b 12.6s → overlap **10.2s**（~1.2×） | **吞吐场景可用** |

以下未写入生产默认：Sliding window / SLA / FlowCache（有加速但质量或复用率不过关）。

---

## 1. kv_offload 下跳过中间步骤的 KV 写回

### 1.1 动机

Self-Forcing 每个 chunk 有多步 denoise + 1 次 **rerun**：

- 中间 denoise：改的是 **latent**；当前 chunk 的 `k_cur/v_cur` 只服务本步 attention，下一步会因 latent 变化重新算。
- **rerun**：用最终 latent 再跑一遍，才需要把当前 chunk KV **持久化**，供后续 chunk 当历史。

开启 `kv_offload` 后，默认路径会在**每步** `store_kv` 时把 dirty 段 D2H 回 CPU。中间步的写回对正确性无贡献，却带来 PCIe + HBM 争用。

### 1.2 做法

配置项（已写入 5s/10s/30s KIVI offload 配置）：

```json
"ar_config": {
  "kv_offload": true,
  "store_kv_only_on_rerun": true
}
```

实现：`transformer_infer._should_persist_self_attn_kv()` —— 仅当 `scheduler.is_rerun` 时调用 `store_kv`。

定稿 **单缓冲** offload（HEAD）：一层一块 GPU staging；`store_kv` 内联 dirty 段 D2H。  
双缓冲 / dirty-range 异步写回在 10s 上反而更慢（DMA 与 Flash 抢带宽），已回退。

### 1.3 测试结果

**环境**：14B INT8 + KIVI，Ulysses P=4，`CVD=0,2,4,5`，同环境 A/B。

#### 10s（161 帧，13 chunks）— offload 内部策略对比

| 策略 | Wall (s) | 相对「单缓冲每步写」 |
|------|----------|----------------------|
| 单缓冲，每步写回 | 88.8 | 1.00× |
| 双缓冲，整层写回 | 95.8 | 1.08× 更慢 |
| 双缓冲，dirty 写回 | 102.6 | 1.16× 更慢 |
| 双缓冲 + **只在 rerun 写** | 82.1 | 0.92× |
| **单缓冲 + 只在 rerun 写（定稿）** | **79.4** | **0.89×（约快 10.6%）** |

数据：`sf_14b_kv_offload_final_policy.json`、`sf_14b_int8_ulysses_p4_10s_kivi_offload_*sameenv.json`。

#### 5s（81 帧）— 定稿策略 vs 无 offload（两边都开 `store_kv_only_on_rerun`）

| | Wall (s) | vs no-offload |
|--|----------|---------------|
| 无 offload | **29.2** | — |
| 单缓冲 offload + rerun-only | **32.4** | **+3.2s（+10.9%）** |

对照旧策略（每步写回）：无 offload 34.2s → offload 43.7s（**+27.7%**）。  
**只在 rerun 写把 offload 税从约 28% 压到约 11%。**

数据：`sf_14b_int8_ulysses_p4_5s_offload_vs_no_offload_rerun_store_sameenv.json`。

### 1.4 取舍

- **正确性**：中间步不落盘；attention 用 history（CPU/staging）+ 临时 `k_cur`。
- **代价**：相对无 offload 仍慢约一成（H2D 预取仍按层拉历史）；换来长视频可跑（10s+）。
- **不推荐**：双缓冲异步写回（在本机 PCIe A10 上墙钟更差）。

---

## 2. 长视频 / 好互联下用 Stripe 换通信与计算

### 2.1 动机

- **Ulysses**：按 head 切分，每卡 Flash 看全长 K；每步 a2a **Q + 当前 chunk K/V + out**（不搬历史 KV）。短视频、PCIe、大 P 上通常更稳。
- **Stripe**（`stripe_partial` / `stripe_pe` formc）：按 **序列** 切 KV，每卡 Flash 只看 `K/P`，再 gather/a2a `out+LSE` 合并。  
  → Flash 算力随 P 下降；通信主要跟 Nq，**不随历史 Nk** 线性变重。

因此：

- **短 KV / 大 P / PCIe**：Ulysses 更好（Stripe gather 更贵）。
- **长 KV + 小 P（如 P=2 30s）或 NVLink**：Stripe 更值得——用更好的 Flash/访存形态换掉过重的全长扫 K。

### 2.2 测试结果

#### 14B INT8+KIVI，P=4，Ulysses vs Stripe formc（公平 Ulysses）

| 时长 | Ulysses (s) | Stripe formc (s) | 结论 |
|------|-------------|------------------|------|
| 5s（无 offload） | **34.1** | 46.1 | Ulysses 快 **1.35×** |
| 10s（+offload） | **82.7** | 115.0 | Ulysses 快 **1.39×** |
| **30s（+offload）** | 584.8 | **576.7** | **Stripe 略快 ~1.4%**（fair 重跑；旧 600.6 为 7/17 不可比） |

数据：`sf_14b_int8_stripe_formc_p4_30s_commfix_per_chunk.json`；Ulysses `sf_14b_int8_ulysses_p4_30s_commfix_per_chunk.json`。

#### 14B P=2 对照（同会话 fair）

| 时长 | Ulysses | Stripe | 结论 |
|------|--------:|-------:|------|
| 5s | 57.9 | **57.8** | 近似持平 |
| 10s | 141.3 | **139.0** | Stripe 略快 |
| 30s | 987.2 | **930.9** | **Stripe ~6%** |

#### 1.3B 480p 5s（公平口径）

| SP | Ulysses | Stripe | flash U/S | comm U/S | 结论 |
|----|--------:|-------:|----------:|---------:|------|
| 2 | 12.11 | **11.75** | 4.25 / **3.82** | 2.05 / **1.57** | Stripe ~3% |
| 4 | **7.38** | 12.69 | — | — | Ulysses 更好 |

数据：`sf_1p3b_p2_5s_ulysses_vs_stripe_comm_compute.json`、`sf_ar_ulysses_fair_comm_compute_summary.json`。

### 2.3 使用建议

| 条件 | 建议 |
|------|------|
| 5–10s、PCIe A10、P=4 | **Ulysses**（默认） |
| **≥30s 长 KV + P=2** | 优先 **Stripe formc**（本机约 6% 端到端） |
| **≥30s、P=4** | 本机公平口径下 **Ulysses 仍略优**；Stripe 在 Flash 侧仍更省，换更好互联再评估 |
| P=2、每卡 KV 压力大 | 优先试 **Stripe** |
| **NVLink / 高带宽互联** | Stripe 通信税更低，长视频更倾向 Stripe（本机无 NVLink，属外推；机制上成立） |

配置示例：`--seq_p_attn_type stripe_pe --stripe_partial_out`（或 config `parallel.seq_p_attn_type`）。

---

## 3. 其它已验证、可写入方案的优化点

### 3.1 KIVI KV 量化（显存使能项）

- **作用**：self-attn KV INT8（group_size=64）压缩，配合 offload 支撑 10s/30s。
- **代价**：短视频有速度税（例：14B 约 2s 短跑 12.9s → 14.6s）；长视频是「能跑」的前提。
- **配置**：`ar_config.kv_quant.quant_scheme = "kivi"`。

### 3.2 Ulysses 序列并行扩展（短/中视频默认）

1.3B 480p transformer（7 chunks × 4+rerun）：

| P | Wall (s) | vs P=1 |
|---|----------|--------|
| 1 | 17.1 | 1.0× |
| 2 | 11.2 | 1.5× |
| 4 | **6.3** | **2.7×** |
| 6 | 5.4 | 3.2×（效率下降） |

数据：`sf_transformer_sp_scaling.md`。

### 3.3 双请求 a2a overlap（吞吐）

同 SP 上两路请求重叠 Ulysses a2a 与计算：

| P | Dual b2b (s) | Dual overlap (s) | vs b2b |
|---|--------------|------------------|--------|
| 2 | 22.5 | **19.6** | ~1.15× |
| 4 | 12.6 | **10.2** | ~1.24× |

数据：`sf_dual_overlap_scaling.json`。适合多租户；不降低单请求延迟。

### 3.4 尝试过但未采纳（避免重复踩坑）

| 方向 | 结果 | 原因 |
|------|------|------|
| 双缓冲 / dirty-range 异步 D2H | 10s 更慢 | DMA 与 Flash 抢 PCIe/HBM |
| Sliding window（`local_attn_size`） | la=9 约 1.19×，SSIM ~0.73 | 质量不够 |
| SLA sparse attn（SF AR） | ~1.29×，SSIM ~0.32 | 质量崩 |
| FlowCache 跳步 | 复用率 0，反而更慢 | SF 步间变化大，阈值未命中 |
| 完全跳过 rerun | ~1.24× | latent 漂移大，不可作无损优化 |

---

## 4. 推荐默认栈（14B 长视频）

```text
INT8 DiT
+ KIVI KV quant
+ kv_offload=true
+ store_kv_only_on_rerun=true   # 单缓冲写回，只在 rerun
+ Ulysses SP=4                  # 5–30s / PCIe 默认（公平口径）
# 或 Stripe formc SP=2          # P=2 长视频本机已验证约 6%
# 或 Stripe formc SP=4          # 更好互联时再评估；本机 P=4 30s 未赢 Ulysses
```

对应配置：

- `configs/self_forcing/wan_t2v_sf_14b_int8_sp_5s_kivi_offload.json`
- `configs/self_forcing/wan_t2v_sf_14b_int8_sp_10s_kivi_offload.json`
- `configs/self_forcing/wan_t2v_sf_14b_int8_sp_30s_kivi_offload.json`

---

## 5. 数据索引

结果均在 `save_results/optimization_study/`（路径相对仓库根目录）：

| 主题 | 文件 |
|------|------|
| Offload 定稿策略 | `save_results/optimization_study/sf_14b_kv_offload_final_policy.json` |
| 10s 单/双缓冲 / rerun-only | `save_results/optimization_study/sf_14b_int8_ulysses_p4_10s_kivi_offload_*sameenv.json` |
| 5s offload vs no-offload（当前策略） | `save_results/optimization_study/sf_14b_int8_ulysses_p4_5s_offload_vs_no_offload_rerun_store_sameenv.json` |
| 5/10/30s Ulysses vs Stripe（公平通信） | `save_results/optimization_study/sf_ar_ulysses_fair_comm_compute_summary.json` |
| 旧 per-chunk（历史） | `save_results/optimization_study/sf_14b_int8_ulysses_vs_stripe_formc_{5,10,30}s_per_chunk.json` |
| 1.3B Stripe vs Ulysses | `save_results/optimization_study/sf_stripe_vs_ulysses.json` |
| SP / dual overlap | `save_results/optimization_study/sf_transformer_sp_scaling.md`、`sf_dual_overlap_scaling.json` |
| 1.3B 早期 tracker | `save_results/sf_optimization_tracker.md` |
