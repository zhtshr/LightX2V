# Hierarchical Stripe Attention（stripe_hier）

本文说明 **Hier（分层 Stripe）** 的算法逻辑，以及在不同并行度 P、组大小 g、序列长度比 `r = Nk/Nq` 下与 **Form C**、**Ulysses** 的通信量对比。实现见 `lightx2v/common/kvcache/base.py`（`_stripe_attn_form_hier` / `_hier_pair_and_crosses`）。

---

## 1. 背景：三种序列并行注意力

Self-Forcing 等场景下，每步本地 Q 长度 Nq（通常为一个 chunk），KV cache 长度 Nk 随时间增长。P 张卡做序列并行时，常见三条路径：

| 方案 | 数据布局 | 通信直觉 |
|------|----------|----------|
| **Ulysses** | 序列切分 → all-to-all 换成头切分 | 搬 Q,K,V,O，体积 ∝ (Nq+Nk)/P |
| **Stripe Form C** | KV 按序列 stripe 留在本地；gather 全 Q，对本地 K 做 Flash，再 a2a 回 partial | **不搬 KV**；体积 ∝ Nq |
| **Stripe Hier** | 先在小组内交换 KV stripe，再把 Q（及 partial）发给其它组 | 在「搬一点 KV」和「少搬 Q」之间折中 |

Form C 在 P 大时：全量 Q gather + 满 head 的 Flash 分片回收，通信因子接近 4·Nq，且集体通信易受 skew。Hier 用 **两级拓扑** 降低有效通信因子，并便于 P2P + 计算重叠。

---

## 2. 算法逻辑

### 2.1 记号

- P：`seq_p` world size  
- g：组大小（当前实现固定 **g=2**，即 pair；下文给出一般 g）  
- 组数 `G = P/g`（要求 g 整除 P）  
- 每卡持有一条 KV stripe，长度 `Uk = Nk/P`；本地 Q 长度 `Uq = Nq/P`  
- 单位：下文通信量均为 **每卡网络 send+recv**，量纲 ×HDb（忽略 LSE；H 头数、D head dim、b 字节/元素）

### 2.2 一般形式（组大小 g）

把 P 个 rank 划成 G 个互斥组，每组 g 人。

**Round 1 — 组内 KV 聚合**

- 组内交换 stripe，使组内每张卡都持有本组全部 KV：

```text
|K_group| = g · Uk = (g/P) · Nk
```

- g=2 时即 **pair 互换**（一次双向 P2P，可 pack `K||V`）。

**Round 2 — 跨组 Q 与 partial**

- 每张卡把本地 Q（长度 Uq）发给其余 `G-1 = P/g - 1` 个组各 **一个代表 peer**（同组内 offset 对齐，保证一一对应）。
- 对端用自己的 K_group 对收到的 Q 做 Flash，把 **partial out + LSE** 传回。
- 本地：对本卡 Q 在 K_group 上 Flash（可与 Q 飞行重叠），再与收回的各跨组 partial 做 online softmax merge。

**正确性**

- 本地 Q 先看本组 `(g/P)·Nk` 的 KV，再通过 G−1 路 remote partial 覆盖其余组 → 合起来覆盖全长 Nk。
- 每组对「外来 Q」只算本组 KV 上的贡献并归还 → 全局等价于对完整 K,V 的注意力（与 Form C / B2 相同的 partial-merge 语义）。

### 2.3 当前实现：g=2（P=4 / P=6）

拓扑由 `_hier_pair_and_crosses` 决定：

| P | Pair（KV） | Cross peers（Q / partial） |
|--:|------------|----------------------------|
| 4 | `(0,1)(2,3)`，partner = `rank^1` | 1 个：`rank^2` |
| 6 | `(0,1)(2,3)(4,5)`，partner = `rank^1` | 2 个：其它两组中 **同 offset** 的 rank |

**时间线（实现中的 overlap）**

```text
  compute stream                    side stream
  ─────────────────                 ───────────
  post pair-KV ───────────────────► isend/irecv K||V
  Flash(Q, K_local)   [可选重叠]
  wait KV
  build K_group = cat(local, peer)
  post cross-Q ───────────────────► isend/irecv Q × (G-1)
  Flash(Q, K_peer) + merge        或 一次性 Flash(Q, K_group)
  wait Q
  Flash(cat(remote Qs), K_group)    # fused
  return pack(out||lse) to peers
  merge remote partials → 本地 out
```

环境变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `LIGHTX2V_STRIPE_HIER_PACK` | `1` | pack `K||V`、`out||lse`，减少 P2P 次数 |
| `LIGHTX2V_STRIPE_HIER_OVERLAP_LOCAL` | `1` | KV 飞行时做 `Flash(Q, K_local)` |

入口：`seq_p_attn_type=stripe_hier` / `hier_exchange=True`。Form C 路径可用 `stripe_formc_hier_until_chunk`（或 env `LIGHTX2V_STRIPE_FORMC_HIER_UNTIL`）在前几个 chunk 自动切到 hier。

### 2.4 与 Form C / B2 的关系（直觉）

```text
Form C:   每卡留 1/P 的 K  ── gather 全 Q ── Flash ── a2a partial
B2:       Q 依次（或网状）访问每个 K owner
Hier:     先把 K 聚成 g/P，再让 Q 只访问 G-1 个「组代表」
```

`g → 1`：退化成「不聚 KV、Q 访遍其它卡」（接近 B2 字节量）。  
`g → P`：组内 allgather 全 KV，无跨组 Q（接近「每卡复制全 KV」）。

---

## 3. 通信量公式

### 3.1 Hier（一般 g）

组内 KV（K+V，每 peer send+recv）+ 跨组 Q + 跨组 out（与 Q 同量级，忽略 LSE）：

```text
V_hier(g) = (4/P) · [ (g-1)·Nk + (P/g - 1)·Nq ]
```

固定 **g=2**：

```text
V_hier(2) = (4/P)·Nk + (2 - 4/P)·Nq
```

连续最优组大小（再投影到 P 的因子，并夹紧到 [2, P/2]）：

```text
g* ≈ sqrt(P · Nq/Nk) = sqrt(P/r),   r = Nk/Nq

V* ≈ 8·sqrt(Nk·Nq)/sqrt(P) - 4·(Nk+Nq)/P
   = O(P^(-1/2))
```

### 3.2 Form C

`all_gather(Q)` + `all_to_all(partial out)`（LSE 忽略）：

```text
V_FormC = 4·(1 - 1/P)·Nq
（与 Nk 无关）
```

### 3.3 Ulysses（本仓库 SF + head-shard KV cache）

经典无 cache Ulysses 会对 **全长** Q/K/V/O 做 a2a，体积含 `Nk`。  
**SF 路径不是这样**：历史 KV 已按 head 落盘，每步只 a2a **Q、当前 chunk K/V、out**（均 ~Nq）：

```text
V_Uly_SF ≈ 4 · (P-1)/P² · Nq
         = O(Nq / P)     # 不随历史 Nk 增长
```

（对照：无 cache 经典式 `4·(P-1)/P²·(Nq+Nk)` 仅适用于非 AR / 无 head-shard cache；**勿用于 SF 长视频外推**。）

### 3.4 渐近（P → ∞）

| 方案 | 渐近 |
|------|------|
| Form C | → 4·Nq |
| Hier g=2 固定 | → 2·Nq |
| Hier **最优 g** | → 0（O(P^(-1/2))） |
| Ulysses | → 0（O(Nq/P)，**不随 Nk**） |

---

## 4. 数值对比表

以下均以 **每卡 V / (HDb)** 计。令 Nq = N。

### 4.1 固定 g=2：早期 Nk = Nq = N

| P | Hier g=2 | Form C | Ulysses | Hier/FormC | Hier/Uly |
|--:|---------:|-------:|--------:|-----------:|---------:|
| 2 | 2.00N | 2.00N | 1.00N | 1.00 | 2.00 |
| 4 | 2.00N | 3.00N | 0.75N | 0.67 | 2.67 |
| 6 | 2.00N | 3.33N | 0.56N | 0.60 | 3.60 |
| 8 | 2.00N | 3.50N | 0.44N | 0.57 | 4.57 |
| 16 | 2.00N | 3.75N | 0.23N | 0.53 | 8.53 |

### 4.2 固定 g=2：后期 Nk = 10N，Nq = N

| P | Hier g=2 | Form C | Ulysses（SF） | 相对 Form C |
|--:|---------:|-------:|-------------:|:------------|
| 2 | 6.00N | 2.00N | **1.00N** | Hier 更差 |
| 4 | 5.00N | 3.00N | **0.75N** | Hier 更差 |
| 6 | 4.00N | 3.33N | **0.56N** | Hier 更差 |
| 8 | 3.50N | 3.50N | **0.44N** | **持平 FormC** |
| 16 | 2.50N | 3.75N | **0.23N** | **Hier 更好 vs FormC** |

> 旧表曾把 Ulysses 写成随 `Nk` 涨（P=4 记 4.13N）；那是**无 cache 经典公式**，对 SF **高估** Ulysses 通信。SF 下 Ulysses 仍远小于 Hier/FormC 字节量，长视频 Stripe 赢面主要来自 **Flash/kv_read**，不是「Ulysses 通信随 Nk 爆掉」。

交叉条件（g=2 vs Form C）：约当

```text
Nk / Nq ≈ P - 2
```

### 4.3 选最优 g：早期 Nk = Nq = N（g* = sqrt(P)）

| P | g* | 离散 g | V* | Form C | Ulysses |
|--:|---:|-------:|---:|-------:|--------:|
| 4 | 2 | 2 | 2.00N | 3.00N | 0.75N |
| 8 | ≈2.8 | 2 | 2.00N | 3.50N | 0.44N |
| 16 | 4 | **4** | **1.50N** | 3.75N | 0.23N |
| 64 | 8 | **8** | **0.875N** | 3.94N | 0.062N |

相对锁死 g=2：P=16 通信量 2N → 1.5N（约 −25%）；P=64 约 −56%。

### 4.4 选最优 g：P=16，扫描 g ∈ {2,4,8}

```text
V_16(g) = (1/4) · [ (g-1)·Nk + (16/g - 1)·Nq ]
```

| g | 表达式 | 早 Nk=Nq=N | 晚 Nk=10N |
|--:|--------|-----------:|----------:|
| 2 | 0.25·Nk + 1.75·Nq | 2.00N | 4.25N |
| **4** | 0.75·Nk + 0.75·Nq | **1.50N** | 8.25N |
| 8 | 1.75·Nk + 0.25·Nq | 2.00N | 17.75N |
| Form C | 3.75·Nq | 3.75N | 3.75N |
| Ulysses（SF） | 4·15/256 · Nq | 0.23N | **0.23N**（不随 Nk） |

判据（仅比 g=2 vs g=4）：Nk < 2·Nq 选 **g=4**；Nk > 2·Nq 选 **g=2**。

### 4.5 P=8：Hier g=2 vs Form C（便于与实现对照）

| | Hier g=2 | Form C |
|--|---------:|-------:|
| 公式 | 0.5·Nk + 1.5·Nq | 3.5·Nq |
| 早 Nk=Nq | 2N | 3.5N |
| 晚 Nk=10·Nq | 6.5N | 3.5N |

（本机无可靠 8 卡；公式外推。实现目前仅挂 P=4/6。）

---

## 5. 怎么选参数（理论建议）

1. **短视频 / 前几 chunk（Nk ~ Nq）**  
   - Hier 相对 Form C：固定 g=2 已稳赢；更大 P 用 `g ~ sqrt(P)` 再降。  
   - 相对 Ulysses：字节量仍大几倍；墙钟要赢需靠 overlap / 算力侧，不能只靠加大 P。

2. **长 KV（Nk ≫ Nq）**  
   - 小 P：Form C 往往更省通信（不搬 KV）。  
   - g* 常被夹到 **2**；只有很大 P（约 `P ≳ 4·Nk/Nq`）才值得升到 g=4。  
   - Ulysses（SF）每步只搬 ~Nq 的 Q/K_cur/V_cur/out，**长视频通信不随历史 Nk 变重**；Stripe 系赢长视频靠算力/访存形态，不是「Ulysses 通信 O(Nk)」。

3. **不均分组（如 P=8 想用「约 g=3」→ 3+3+2）**  
   - 可行但不推荐：大组拖 wall、Q 拓扑变脏；能整除时优先 **g=2 或 g=4**。

4. **实用规则**

```text
g = clamp( nearest_divisor(P, sqrt(P · Nq/Nk)),  2,  P/2 )
```

---

## 6. 实现与开关速查

| 项 | 位置 / 值 |
|----|-----------|
| 核心实现 | `lightx2v/common/kvcache/base.py` → `_stripe_attn_form_hier` |
| 拓扑 | `_hier_pair_and_crosses`（现支持 P=4、P=6，g=2） |
| 配置 | `seq_p_attn_type: stripe_hier` |
| Form C 早期切 hier | `stripe_formc_hier_until_chunk` / `LIGHTX2V_STRIPE_FORMC_HIER_UNTIL` |
| Pack / overlap | `LIGHTX2V_STRIPE_HIER_PACK`、`LIGHTX2V_STRIPE_HIER_OVERLAP_LOCAL` |
| Bench | `scripts/disagg/run_sf_transformer_sp_per_chunk_bench.py` |

---

## 7. 小结

- **Hier** = 组内聚 KV（g/P 长度）+ 跨组传 Q/partial（P/g−1 路），用 partial merge 拼出全注意力。  
- **通信量** `V(g) = (4/P)·[(g-1)·Nk + (P/g-1)·Nq]`；最优 `g* ~ sqrt(P/r)`，最优体积 O(P^(-1/2))。  
- **vs Form C**：短序列 Hier 更省且随 P（尤其调 g）优势扩大；长 KV 小 P 时 Form C 更省。  
- **vs Ulysses（SF）**：字节量全程 O(Nq/P)，通常仍低于 Hier/FormC；调 g 缩小的是 Stripe 系内部差距，短视频要赢 Ulysses 还需 overlap/算力侧。
