# Ulysses 少 Head 与 Flash 利用率下降

> 范围：Self-Forcing / Wan 在 **Ulysses SP** 下每卡 `H_local = H/P` 时，FlashAttention 达峰利用率为何下降。  
> 硬件口径：NVIDIA **A10**（**72 SM**），`flash_attn` 2.x，`head_dim=128`，BF16；峰值参照 **125 TFLOPS**（与既有 microbench 一致）。  
> 相关代码/数据：`scripts/disagg/bench_flash_stripe_vs_ulysses.py`；`save_results/optimization_study/flash_cta216_vs_222_h3.json`。

---

## 1. 现象

Ulysses 把注意力从「全 head、序列切分」换成「**头切分、序列汇合**」：

| 形态 | 每卡形状（示意） |
|------|------------------|
| Stripe | `Q=[B,Q,H,D]`，`K=[B,K/P,H,D]` |
| Ulysses | `Q=[B,Q,H/P,D]`，`K=[B,K,H/P,D]` |

同 FLOPs 下，**少 head 的 Ulysses 形状 Flash 达峰利用率通常更低**；`P` 越大、`H/P` 越小、越短 `Q`，掉得越狠。  
SF 里每 chunk 的 `Q` 往往固定（如 14B 实测 `tokens_per_chunk=4680`），**加长视频不会自动加长 Q**，只加长 `K`。

---

## 2. Flash 的并行粒度（CTA）

FlashAttention-2 大致按：

```text
CTA ≈ B × H_local × ceil(Q / BLOCK_M)
```

启动 thread block。`head_dim=128` 时 `BLOCK_M` 多为 **64**（**编译期常量**，`flash_attn_func` **不能**运行时改）。

常用代理指标：

```text
waves ≈ CTA / #SM
（A10：#SM = 72）
```

注意：这只是「工作量相对 SM 宽度」的尺子，**不等于**「墙钟 = ceil(waves) × T」。

---

## 3. 机制：两种「填不满」（纠正后的说法）

### 3.1 空间饥饿（CTA ≪ 并行槽）——主因之一

并行槽 ≈ `#SM × 每 SM 驻留 CTA 数`（occupancy，Flash 常为 2–3，槽位可到一百多）。

当 `CTA` 明显小于槽位时，**有的 SM 整段 kernel 分不到活**。  
短 `Q` + 很小的 `H/P` 时，达峰利用率可以掉到个位数～二十。

这不是「CTA 之间快慢不齐」，而是 **根本没有足够多的 block 可派**。

### 3.2 访存延迟藏不住（CTA 刚够或略多）——另一主因

Flash 扫 `K` 时经常等 HBM。SM 靠 **多个 CTA/warp 轮流跑** 盖住等待。

- `H` 少 → 同时在飞的 CTA 少 → 等内存时 SM 更容易空转 → 利用率缓降。  
- `H` 多 → 同 SM 上总有别的活可切 → 更容易接近平台上限（本机 Flash 约 **55–62%** of 125T）。

### 3.3 收尾尾巴（次要）

最后结束的 CTA 决定 makespan；CTA 很少时尾巴占比更大。  
Flash 各 Q-tile 工作量接近，**通常不是大头**。

### 3.4 已证伪：`waves=3.08` ⇒ 必多付一整波墙钟

曾用

```text
waves / ceil(waves) = 3.083 / 4 ≈ 77%
```

推断「第 4 波只跑 6 个 CTA、66 个 SM 空 → 墙钟多 ~23%」。该推法假设：

1. 每 SM 同时只跑 1 个 CTA；  
2. 波与波全局齐步。

两者都不成立：

- occupancy ≥ 2 时，`CTA=216` 与 `222` 的「轮数」可以相同；  
- 调度是贪心的，尾 CTA 叠在上一轮收尾上跑。

**实测（H=3，Q=4608→CTA=216 刚好 3×72 vs Q=4680→CTA=222）：**

| 条件 | CTA=216 | CTA=222 | 延迟比 |
|------|--------:|--------:|-------:|
| `K=Q` | 0.515ms / 63.3T | 0.541ms / 62.1T | ~1.05（含 Q 变长） |
| `K=32760` | 3.658ms / 63.4T | 3.715ms / 63.5T | **1.016 ≈ Q 比** |

长 `K` 下延迟几乎纯跟工作量走，**没有 4/3 悬崖**。  
瞬时占用图上仍可能看到「末波很空」，但 **不能**据此估 23% 墙钟损失。

---

## 4. 关键对照实验（摘要）

### 4.1 同 FLOPs、同 CTA 数 → 利用率恢复（一票否决「head 语义有毒」）

| 配置 | CTA | 约达峰 util |
|------|----:|------------:|
| `B=1, H=12, Q=4680` | 888 | ~55% |
| `B=12, H=1, Q=4680` | 888 | ~56% |
| `B=4, H=3, Q=4680` | 888 | ~56% |

缺的是 **`B × H × ceil(Q/64)`**，不是 head 维本身。

### 4.2 固定 `Q=K=4680`，扫 `H`（效率）

| H | CTA | waves | 约 TFLOPS | util@125T |
|--:|----:|------:|----------:|----------:|
| 12 | 888 | 12.3 | ~71 | ~57% |
| 6 | 444 | 6.2 | ~61 | ~49% |
| 3 | 222 | 3.1 | ~62 | ~50% |
| 2 | 148 | 2.1 | ~43 | ~34% |
| 1 | 74 | 1.0 | ~53 | ~43% |

### 4.3 `H=1` 拉长 `Q`：从饥饿到饱和

| Q | CTA | util@125T |
|--:|----:|----------:|
| 512 | 8 | ~2%（大量 SM 无活） |
| 4096 | 64 | ~45% |
| ≥16384 | ≥256 | ~58–59%（接近上限） |

### 4.4 14B / P=4 形态（`H_local=10`）拉长 Q

| Q | waves | util@125T |
|--:|------:|----------:|
| 512 | ~1.1 | ~14% |
| 4680（真实 chunk） | ~10.3 | **~56%** |
| ≥8192 | ≥17 | **~62%**（平台附近封顶） |

真实 SF 14B P=4 的 prefill **已经不是 CTA 饥饿**；再加长 Q 也只能到 ~62%，解释不了 30s 被 Stripe 反超（那是长 `K` 下 Stripe 算力形态 + Form C 通信结构问题，见 `sf_ar_optimization_summary.md`）。

### 4.5 BLOCK_M / 对齐 72

- `flash_attn_func` **无**运行时 `BLOCK_M`；改它需换 Triton/重编，且合法值粗。  
- 把 `Q` pad 到 CTA 对齐 72：**实测更慢**（白算 padding）。  
- decode/`flash_attn_with_kvcache` 的 **`num_splits`** 可造 CTA，只在真正饥饿时有用；已喂满时会变慢。

---

## 5. 和 Stripe / 长视频的关系（避免混淆）

| 问题 | 结论 |
|------|------|
| 少 head 为何 Flash util 低？ | CTA 并行度下降 → 占不满 SM 和/或藏不住访存（本文） |
| 30s Stripe 为何能在 P=2 反超 Ulysses？ | **Q 不变、K 变长**；Stripe `(全 H, K/P)` Flash 持续略快，Form C 通信近似跟 Nq、与历史 Nk 无关；Ulysses **a2a 也不随历史 Nk 涨**（只搬当前 chunk），但本卡仍扫全长 K → Flash/kv_read 更吃亏。累计反超主要来自算力/访存，不是「Ulysses 通信随 K 爆掉」。P=4 30s 公平重跑后墙钟仍常是 Ulysses 略快。 |

等 FLOPs 微基准（Q=4680，P=4）：Stripe `H=40,K/P` 相对 Ulysses `H=10,K` 大约 **快 8–11%**（~72–76T vs ~67–69T），与 profile 里 `stripe_flash < ulysses_flash` 一致。

---

## 6. 实用含义

1. **短 Q / 大 P / 小 `H/P`**：优先怀疑 CTA 饥饿；加 **batch**、减 P、或换 Stripe（满 H）比改 BLOCK_M 有用。  
2. **中等以上 Q（如 4680）+ `H_local≳3`**：更多是 latency hiding / 形状效率，不是「对齐 72」。  
3. **不要用** `waves / ceil(waves)` **估墙钟损失**；用总 CTA 工作量 + 是否 `CTA ≪` 槽位来判断。  
4. 若要抬利用率：`B↑` 最干净；decode 饥饿用 `num_splits`；prefill 已满则别硬造 CTA。

---

## 7. 一句话

**Ulysses 把 head 切少之后，Flash grid（≈ `B×H×Q-tile`）变窄：要么占不满 SM，要么同一 SM 上可切换的活变少、访存气泡变大，所以达峰利用率下降。这是并行度问题，不是「多半个 wave 就要多付一整波时间」。**
