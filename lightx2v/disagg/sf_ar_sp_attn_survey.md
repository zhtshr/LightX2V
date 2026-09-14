# AR 视频生成中的序列并行注意力调研：从 Ulysses 利用率到 Stripe / Hier

> **叙事线**：Self-Forcing（AR chunkwise + KV cache）上发现 Ulysses 少 head 拉低 Flash 利用率 → 提出 **Stripe** 用满 head、切序列换算力 → 大 P 上 Stripe 通信偏贵 → 提出 **Hier** 降通信 → **部分配置下端到端已快于 Ulysses**。  
> **硬件**：NVIDIA A10（PCIe）；多卡避开故障 GPU 1/3。  
> **细节文档**：[ulysses_head_utilization.md](./ulysses_head_utilization.md)、[stripe_hier_algorithm.md](./stripe_hier_algorithm.md)、[sf_ar_optimization_summary.md](./sf_ar_optimization_summary.md)。

---

## 0. 结论一览


| 阶段  | 问题 / 手段                     | 实验上站得住的结论                                                              |
| --- | --------------------------- | ---------------------------------------------------------------------- |
| 1   | AR + Ulysses：每卡 `H/P`、短 `Q` | Flash grid 变窄，达峰利用率低于满 head；非 AR 全长 `Q` 时问题轻得多                         |
| 2   | **Stripe**（序列切 KV，满 `H`）    | 同 FLOPs 下 Flash 更快；**P=2 长视频**及 **14B P4 30s** E2E 可超过 Ulysses         |
| 3   | Stripe 通信税                  | 大 `P` 时 gather/a2a 吃掉算力收益（短中视频仍明显）                                     |
| 4   | **Hier / Hybrid**           | Hier 压通信；短视频 Hybrid 优于纯 Stripe；14B P4 30s 上 Hybrid≈Stripe，均略快于 Ulysses |


---



## 1. 发现：AR 下 Ulysses 利用率下降



### 1.1 场景差异


|            | 非 AR 整段去噪                      | AR / Self-Forcing      |
| ---------- | ------------------------------ | ---------------------- |
| 每步 Q       | 全视频 token（480p 5s ≈ **32760**） | 固定 chunk（实测常 **4680**） |
| K          | 与 Q 同长                         | cache **随 chunk 增长**   |
| Ulysses 每卡 | `H/P`，但 Q-tile 很多              | `H/P`，且 Q-tile 少       |


FlashAttention 并行度大致：

```text
CTA ∝ B × H_local × ceil(Q / 64)
```

AR 把 Q 锁在 chunk 上，Ulysses 再把 H 切成 `H/P`，CTA 很容易偏少。

**数量级（A10，72 SM，**`B=1`**）**


| Workload              | Q     | H_local | CTA  | waves   |
| --------------------- | ----- | ------- | ---- | ------- |
| 非 AR 480p 5s，1.3B P=4 | 32760 | 3       | 1536 | **~21** |
| SF chunk，1.3B P=4     | 4680  | 3       | 222  | **~3**  |
| SF chunk，1.3B P=1     | 4680  | 12      | 888  | ~12     |




### 1.2 机制（摘要）

- **主因**：CTA 少 → 占不满 SM，和/或同一 SM 上可切换的活少、**藏不住扫 K 的访存延迟**。  
- **不是**「`waves` 刚好比整数多一点就多付一整波墙钟」——该说法已被 CTA=216 vs 222 实测否定（详见利用率文档）。  
- **验证**：同 FLOPs 下 `B=1,H=12` 与 `B=12,H=1` 达峰 util 几乎一样（~55%）→ 缺的是 `B×H×Q-tile`，不是 head 语义。



### 1.3 微基准支撑（同 FLOPs 形状对比）

Stripe 形状 `(Q, H, K/P)` vs Ulysses `(Q, H/P, K)`：


| 设定                              | Stripe 相对 Ulysses Flash                       |
| ------------------------------- | --------------------------------------------- |
| 14B/P=4 量级（H=40 vs 10，Q=4680）   | 约快 **8–11%**                                  |
| 1.3B/P=2（H=12 vs 6，同 FLOPs 扫 K） | 均值约快 **~9%**（`mean_stripe_over_ulysses≈0.91`） |


数据：`sf_flash_stripe_vs_ulysses_p2.json` 等。  
→ **满 head + 短本地 K** 的 Flash 效率更高（约一成）。

### 1.4 Ulysses 在 SF+KV cache 下搬什么

历史 KV 已按 head 存在本卡（`[Nk, H/P, D]`）。每步 self-attn 的通信是：

```text
Q a2a + 当前 chunk 的 K/V a2a + out a2a   （均 ~Nq）
历史 KV：本卡直接读，不再 a2a
```

公平口径：`ulysses_comm_ms`（Q+out）+ `sp_kv_a2a_ms`（当前 chunk K/V）。  
Stripe：gather Q + 换 partial；无当前 chunk K/V a2a；Flash 扫本地 `Nk/P`、满 `H`。

---



## 2. 对策一：Stripe —— 先把计算利用率拉回来



### 2.1 思路

按 **序列 stripe 切 KV**：每卡 Flash 看 **全** `H`**、本地** `K/P`，再 merge partial。  
典型路径：gather `Q` → 本地 Flash → 交换 partial out/LSE。

### 2.2 端到端：Ulysses vs Stripe（wall / flash / comm，秒）

口径：Ulysses comm = Q+out + 当前 chunk K/V a2a；Stripe comm = `stripe_comm`。

#### 1.3B SF


|        | Ulysses                | Stripe                   |
| ------ | ---------------------- | ------------------------ |
| P2 5s  | 12.1 / 4.2 / 2.1       | **11.7 / 3.8 / 1.6**     |
| P2 10s | 27.9 / 13.9 / 4.1      | **26.5 / 12.5 / 3.1**    |
| P2 30s | 172.3 / 129.0 / 14.8   | **158.5 / 115.5 / 11.3** |
| P4 5s  | **7.4 / 2.1 / 1.8**    | 12.7 / 1.8 / 6.8         |
| P4 10s | **15.8 / 6.9 / 3.2**   | 24.3 / 5.8 / 11.7        |
| P4 30s | **89.7 / 63.5 / 11.0** | 106.9 / 53.8 / 34.2      |
| P6 5s  | **6.8 / 1.9 / 1.9**    | 11.1 / 1.2 / 6.1         |
| P6 10s | **14.0 / 6.3 / 3.2**   | 21.0 / 3.9 / 11.1        |
| P6 30s | **80.4 / 57.2 / 11.3** | 84.4 / 35.3 / 32.6       |




#### 14B INT8+KIVI


|        | Ulysses                | Stripe                    |
| ------ | ---------------------- | ------------------------- |
| P2 5s  | 57.9 / 15.9 / 9.4      | **57.8 / 15.4 / 8.3**     |
| P2 10s | 141.3 / 52.1 / 18.4    | **139.0 / 50.3 / 16.6**   |
| P2 30s | 987.2 / 489.0 / 73.0   | **930.9 / 467.0 / 63.6**  |
| P4 5s  | **34.1 / 8.4 / 7.4**   | 46.1 / 7.4 / 17.5         |
| P4 10s | **82.7 / 27.4 / 19.6** | 115.0 / 23.8 / 38.2       |
| P4 30s | 584.8 / 255.5 / 120.5  | **576.7 / 220.7 / 169.0** |
| P6 5s  | N/A†                   | 46.0 / 4.9 / 23.5         |
| P6 10s | N/A†                   | 96.6 / 15.9 / 49.6        |


† 14B 有 40 head，不能整除 P=6，Ulysses 不可用。P4 30s Stripe 为同环境 fair 重跑（`*_commfix_per_chunk.json`）。

读法：P=2 越长越偏 Stripe；P≥4 短中视频 Stripe flash 更省但 **comm 更贵**、wall 仍输 Ulysses；**14B P4 30s** 长 KV 下 Stripe 已略快于 Ulysses（~1.4%）。

### 2.3 阶段小结

Stripe 收回算力；**小 P / 长 KV**（1.3B P=2、14B P=2，以及 14B P4 30s）可赢 Ulysses。短中视频大 P 上通信税仍主导 → 需要降通信的拓扑（下一节）。

---



## 3. 对策二：Hier / Hybrid —— 在 Stripe 路线上砍通信



### 3.1 动机（量级）

早期 `Nk ≈ Nq` 时，粗算每卡通信量（单位 ×HDb，忽略 LSE）：


|       | Stripe（gather Q） | Hier g=2            | Ulysses（SF+KV）                          |
| ----- | ---------------- | ------------------- | --------------------------------------- |
| 随 P   | → 4·Nq           | → 2·Nq（固定 g=2）      | O(Nq/P)（Q+out+当前 chunk K/V；**不含历史 Nk**） |
| P=4 早 | 3·Nq             | Nk+Nq               | 更小                                      |
| P=6 早 | ≈3.3·Nq          | (2/3)·Nk + (4/3)·Nq | 更小                                      |


Hier：组内先换一点 KV，再只向其它组代表传 `Q`/partial。算力仍走满 head、局部 K。

### 3.2 Hier

相对纯 Stripe：comm 下降、短视频墙钟更好（尤其更大 P）。长 KV 时组内换 KV 变贵，后期可能反不如 FormC。

### 3.3 Hybrid：前段 Hier + 后段 Stripe

按 chunk 墙钟选切换点（Hier 更快则继续，否则切 FormC）：`--formc_hier_until N`。  
取两者之长：前段省通信，后段避免 Hier 在长 KV 上换 KV 过重。

### 3.4 全表：Ulysses / Stripe / Hier / Hybrid（wall / flash / comm）

口径同 §2.2；完整 JSON：`sf_ar_full_matrix_wall_flash_comm.json`。P=2 无 Hier/Hybrid。

#### 1.3B SF


|        | Ulysses                | Stripe                   | Hier                | Hybrid                        |
| ------ | ---------------------- | ------------------------ | ------------------- | ----------------------------- |
| P2 5s  | 12.1 / 4.2 / 2.1       | **11.7 / 3.8 / 1.6**     | —                   | —                             |
| P2 10s | 27.9 / 13.9 / 4.1      | **26.5 / 12.5 / 3.1**    | —                   | —                             |
| P2 30s | 172.3 / 129.0 / 14.8   | **158.5 / 115.5 / 11.3** | —                   | —                             |
| P4 5s  | **7.4 / 2.1 / 1.8**    | 12.7 / 1.8 / 6.8         | 12.1 / 2.1 / 5.7    | 10.5 / 2.0 / 4.1 (until=4)    |
| P4 10s | **15.8 / 6.9 / 3.2**   | 24.3 / 5.8 / 11.7        | 26.1 / 6.7 / 11.4   | 22.9 / 6.1 / 9.8 (until=5)    |
| P4 30s | **89.7 / 63.5 / 11.0** | 106.9 / 53.8 / 34.2      | 185.5 / 59.5 / 96.5 | 108.9 / 53.8 / 35.8 (until=5) |
| P6 5s  | **6.8 / 1.9 / 1.9**    | 11.1 / 1.2 / 6.1         | 8.9 / 1.8 / 3.0     | 8.9 / 1.8 / 3.0 (until=6)     |
| P6 10s | **14.0 / 6.3 / 3.2**   | 21.0 / 3.9 / 11.1        | 21.0 / 5.6 / 8.6    | 19.0 / 4.5 / 8.1 (until=6)    |
| P6 30s | **80.4 / 57.2 / 11.3** | 84.4 / 35.3 / 32.6       | 141.3 / 48.9 / 69.0 | 83.5 / 35.7 / 30.9 (until=6)  |




#### 14B INT8+KIVI


|        | Ulysses                | Stripe                    | Hier                   | Hybrid                              |
| ------ | ---------------------- | ------------------------- | ---------------------- | ----------------------------------- |
| P2 5s  | 57.9 / 15.9 / 9.4      | **57.8 / 15.4 / 8.3**     | —                      | —                                   |
| P2 10s | 141.3 / 52.1 / 18.4    | **139.0 / 50.3 / 16.6**   | —                      | —                                   |
| P2 30s | 987.2 / 489.0 / 73.0   | **930.9 / 467.0 / 63.6**  | —                      | —                                   |
| P4 5s  | **34.1 / 8.4 / 7.4**   | 46.1 / 7.4 / 17.5         | 48.2 / 8.6 / 18.2      | 42.3 / 7.7 / 14.2 (until=2)         |
| P4 10s | **82.7 / 27.4 / 19.6** | 115.0 / 23.8 / 38.2       | 137.0 / 27.3 / 67.3    | 99.2 / 24.4 / 36.2 (until=3)        |
| P4 30s | 584.8 / 255.5 / 120.5  | **576.7 / 220.7 / 169.0** | 1069.5 / 245.4 / 594.1 | **576.4 / 222.1 / 166.7** (until=2) |
| P6 5s  | N/A†                   | 46.0 / 4.9 / 23.5         | 37.7 / 5.8 / 14.9      | **37.5 / 5.6 / 15.1** (until=5)     |
| P6 10s | N/A†                   | 96.6 / 15.9 / 49.6        | 101.7 / 18.4 / 51.1    | **87.2 / 16.5 / 40.5** (until=5)    |
| P6 30s | N/A†                   | —                         | —                      | —                                   |


† 同 §2.2。P6 30s 未测。

vs Ulysses：1.3B / 14B 短视频仍慢；**14B P4 30s 上 Stripe 与 Hybrid 几乎持平（576s），均略快于 Ulysses（585s，约 1.4%）**；短视频大 P 上 Hybrid 相对纯 Stripe 更明显。

### 3.5 阶段小结

Hier 压通信；**短视频 Hybrid 优于纯 Stripe**；长 KV（14B P4 30s）上 Hybrid≈Stripe。短视频大 P 仍难全面超过 Ulysses。

---



## 4. 整条调研链怎么串起来

```text
AR + Ulysses
  │  短 Q、H/P → Flash CTA 少 → 利用率↓
  │  热路径 a2a = Q+out + 当前 chunk K/V；不搬全长历史 KV
  │  但 Flash 仍在本卡扫全长 Nk（少 head）
  ▼
Stripe（满 H，切 K；主要搬 Q + partial）
  │  Flash↑；小 P / 长 KV → 可赢 Ulysses
  │  短中视频大 P → 通信税过大
  ▼
Hier / Hybrid（前 Hier 后 Stripe）
  │  短视频 vs 纯 Stripe：comm↓、wall↓
  │  14B P4 30s：Stripe≈Hybrid，均略快于 Ulysses
  ▼
场景策略
  · 短中视频、大 P → Ulysses
  · 长 KV + 小 P → Stripe
  · Stripe 路线默认 → Hybrid（formc_hier_until；长 KV 上与纯 Stripe 接近）
```



### 4.1 「延迟低于 Ulysses」落点


| 配置                | 更快                  | 幅度    |
| ----------------- | ------------------- | ----- |
| 1.3B P=2，5/10/30s | Stripe              | ~3–8% |
| 14B P=2，10/30s    | Stripe              | ~2–6% |
| **14B P=4，30s**   | **Stripe / Hybrid** | ~1.4% |
| 其余已测 P≥4          | Ulysses             | —     |




### 4.2 和「非 AR」的边界

非 AR 480p 5s 的 Q ≈ 3.3e4，即便 `H/P=3` 也有 ~21 wave，**少 head 利用率问题轻得多**。  
本调研的核心矛盾是 **AR 短 chunk Q** 放大了 Ulysses 的算力形态劣势；这也是为何 Stripe/Hier 首先在 SF 路径上值得做。

---



## 5. Hybrid 更有优势的场景

本调研主要在 **A10 + PCIe** 上测；Hybrid / Stripe 路线在下列条件下相对 Ulysses 更值得押：

1. **NVLink、SM 数更大的硬件**  
   Stripe/Hybrid 的算力形态（满 `H`、短本地 K）更能吃满更强 SM；NVLink 压低 gather / a2a 通信税后，Flash 侧收益更容易落到端到端。

2. **长视频（长 KV）**  
   AR 下 `Nk` 随 chunk 增长，Ulysses 少 head 扫全长 K 的劣势放大；14B P4 30s 上 Stripe/Hybrid 已略快于 Ulysses，更长视频预期差距继续拉开。

3. **P=2**  
   通信规模小、满 head 收益够直接兑现；实测 1.3B / 14B 的 P=2 上 Stripe 已稳定优于 Ulysses（P=2 无 Hier，等同 Stripe 路线）。

短中视频、大 P（尤其 PCIe）仍倾向 Ulysses。

---

## 6. 一句话

**AR 短 chunk + Ulysses 少 head → Flash 利用率下降；Stripe 收回算力，小 P/长 KV（含 14B P4 30s）可赢；Hybrid 在短视频上进一步压通信，长 KV 上与纯 Stripe 接近。**
