# DisagFusion 扩展：Stage-aware 并行与多 Batch 联合优化

> 日期：2026-05-28
> 基于 DisagFusion（encode/decode 阶段解耦 + 异步流水线并行 + 弹性调度）的扩展 idea

---

## 一、动机

DisagFusion 实现了 diffusion serving 的阶段级解耦架构（encode/decode 分离、异步流水线、弹性调度），但当前系统在各阶段内部尚未引入并行策略和多 batch 处理。现有工作（TetriServe、GenServe 等）虽然在单集群内探索了步级序列并行和动态资源分配，但存在两个局限：

1. **并行策略单一**：TetriServe 仅做 SP 维度的并行度调整，GenServe 仅做弹性 SP，都没有考虑 TP、SP、Patch 等多种并行策略的选择问题。
2. **并行与 batch 割裂**：现有工作将并行度调整和 batch 管理视为独立问题，没有显式建模两者之间的协同效应——并行策略降低 per-GPU 计算量后，原本 batch=1 即饱和 GPU 的计算可能获得多 batch 的空间。

本 idea 的核心观察是：**diffusion serving 的不同阶段计算特征差异巨大，应按阶段特征分治——计算密集的阶段用并行策略切分计算量并用异步流水线隐藏通信，计算轻量的阶段用多 batch 饱和 GPU**。

---

## 二、核心策略：按计算特征分治

### 2.1 策略概述

```
┌───────────────────-──────────────────────────────────────────┐
│                     DisagFusion 扩展架构                      │
├──────────-┬──────────────────────┬───────────────────────────┤
│  Encode   │      Denoise         │          Decode           │
│  (轻量)    │      (密集)          │         (中等)             │
├─────────-─┼──────────────────────┼───────────────────────────┤
│  多 Batch │  并行 (SP/TP)         │         多 Batch          │
│  B >> 1   │  + 异步流水线覆盖通信   │         B >> 1            │
│  P = 1    │  P >> 1              │         P = 1             │
│  1 GPU    │  K GPUs per request  │         1 GPU             │
└─────────-─┴──────────────────────┴───────────────────────────┘
```

**判断标准**：根据每个阶段（甚至每个请求在每个阶段内）的计算量和显存需求，动态选择优化策略：

- **计算/显存密集** → 并行策略（SP-Ulysses 或 TP）+ 异步流水线覆盖通信
- **计算/显存轻量** → 多 batch，不使用并行

### 2.2 各阶段的策略映射

**Encode 阶段（Text Encoder）**
- 计算特征：text encoder（T5、CLIP 等）的 forward pass 计算量远小于 denoise，单次推理毫秒级。
- 显存特征：模型参数量中等（T5-XXL 约 11B 但可用量化版本），activation 小。
- 策略：多请求 batch（B=16~32），单 GPU 即可饱和。不使用并行。
- 产出：text embedding + latent 初始化噪声，体积小，传给 denoise 实例的传输开销低。

**Denoise 阶段（DiT Iterative Denoising）**
- 计算特征：每步做一次完整的 DiT forward pass，attention 复杂度 O(N²×d)，N 为 patch 数（高分辨率图像可达 4096+）。加上 CFG 需要 conditional + unconditional 两次 forward。TetriServe profiling 证实此阶段 compute-bound，batch=1 即可饱和单 GPU。
- 显存特征：大模型权重（DiT-XL 约 675M，更大的视频模型可达 14B+）+ 大 activation（高分辨率时序列长）。
- 策略：并行策略切分计算量（SP-Ulysses 或 TP），多请求通过异步流水线覆盖通信。
- 产出：去噪后的 latent，传给 decode 实例。

**Decode 阶段（VAE Decoder）**
- 计算特征：VAE decoder 是卷积网络，计算量比 denoise 小一到两个数量级。
- 显存特征：模型参数小，activation 中等（取决于输出分辨率）。
- 策略：多请求 batch（B=8~16），单 GPU 即可。不使用并行。

---

## 三、关键设计决策

### 3.1 并行策略选择：SP-Ulysses vs TP

两种并行策略的选择应基于模型大小和互联拓扑自动决策：

| 条件 | 选择 | 理由 |
|------|------|------|
| 模型能放进单 GPU 显存 + 高分辨率（长序列） | SP-Ulysses | 切序列维度，通信 O(seq_len × d_model)，可跨节点 |
| 模型太大、单 GPU 放不下 | TP（或 TP+SP 混合） | 切权重矩阵，通信 O(model_size)，必须 NVLink |
| 低分辨率（短序列） | 不并行，用多 batch | 序列短、计算量不足以从并行中获益 |

**SP-Ulysses 的通信机制**：将输入序列沿 head 维度切分到 K 个 GPU，每个 GPU 持有完整的 token 子集但只计算部分 attention head。每层需要两次 all-to-all 通信（QKV 分发前和 attention 计算后），通信量为 O(seq_len × d_model × num_layers)。

**TP 的通信机制**：将 QKV projection 和 FFN 权重矩阵沿列/行切分，每层需要 all-reduce，通信量为 O(model_size × num_layers)。通信在 critical path 上，对互联带宽要求极高。

**联合策略（TP+SP）**：类似 Megatron-LM 的做法，节点内 TP（NVLink）+ 节点间 SP（InfiniBand），适合超大规模模型的多节点部署。

### 3.2 异步流水线覆盖通信

这是本方案的关键使能技术，直接复用 DisagFusion 已有的异步流水线基础设施。

**通信隐藏的原理**：

denoise 阶段同时有多个请求在不同步上推进。当请求 A 在某层做 SP-Ulysses 的 all-to-all 通信时，GPU 的计算单元空闲，此时可以执行请求 B 的同一层矩阵乘法。通信和计算在不同请求之间重叠，有效隐藏通信延迟。

**覆盖条件**：设每层通信时间为 T_comm，计算时间为 T_comp，需要约 ⌈T_comm / T_comp⌉ 个并发请求才能完全覆盖通信。在大分辨率下 T_comp 大（compute-bound），可能 2~3 个请求即可；在小分辨率下 T_comp 小，但此时可能不需要并行策略，直接用多 batch。

**与 TP 的结合**：TP 的 all-reduce 同样可以被覆盖。all-reduce 是通信操作不占计算单元，请求 A 的 all-reduce 可以和请求 B 的矩阵乘法并行。请求越多，TP 的有效通信开销趋近于零。

**与 PP（PipeFusion 风格）的对比**：PipeFusion 通过跨步流水线减少通信量（只传边界信息），但引入 staleness。本方案的异步流水线不减少通信量，而是隐藏通信延迟，保持零 staleness。两者可以结合——用 PipeFusion 的 patch 级流水线降低通信量，再用异步流水线覆盖剩余通信。

### 3.3 CFG（Classifier-Free Guidance）的处理

CFG 要求每个 denoising 步做 conditional 和 unconditional 两次 forward pass，需要特别设计：

- **方案 A（Batch 合并）**：将 conditional 和 unconditional 合并为 batch=2 做一次大 forward。优点：实现简单，GPU 利用率翻倍。缺点：显存加倍，高分辨率时可能 OOM。
- **方案 B（串行）**：先后做两次 forward。优点：显存不增加。缺点：延迟翻倍。
- **方案 C（GPU 子组分配）**：K 张 GPU 分成两组，K/2 做 conditional，K/2 做 unconditional，然后合并结果。优点：延迟不增加、显存不增加。缺点：调度复杂度高，需要跨子组的同步。

**建议**：默认使用方案 A（batch 合并），在显存不足时 fallback 到方案 C。这个决策可以作为调度器的 per-request 参数。

### 3.4 GPU 资源分配

给定 N 张 GPU，需要在 encode、denoise、decode 三种角色之间动态分配：

- **encode 实例**：1 GPU / 实例，大 batch
- **denoise 实例**：K GPUs / 实例（K = 并行度），多实例并行
- **decode 实例**：1 GPU / 实例，大 batch

**动态分配策略**：denoise 阶段是延迟瓶颈，应优先分配 GPU。encode/decode 的 throughput 需要和 denoise 匹配——如果 denoise 处理速度是 10 req/s，encode/decode 也需要至少 10 req/s 的吞吐。由于 encode/decode 单次推理快、batch 大，通常 1 个实例即可匹配多个 denoise 实例。

GPU 分配可以根据实时负载动态调整（复用 DisagFusion 的弹性调度机制）：当 denoise 队列积压时，临时将 encode/decode GPU 转为 denoise 用；当 encode 队列积压时，增加 encode 实例。

### 3.5 权重放置与 GPU 角色切换

不同并行策略对权重的放置方式有本质差异，这直接影响系统的显存预算、GPU 角色切换灵活性和动态并行度调整。

**权重切分 vs 权重复制**：

| 并行策略 | 权重放置 | 每 GPU 权重占比 | 显存影响 |
|----------|----------|----------------|----------|
| TP | 切分（每张 GPU 持有 1/K 的权重矩阵） | 1/K | 权重显存降为 1/K，可部署更大模型 |
| SP-Ulysses | 复制（每张 GPU 持有完整权重） | 1（完整） | 权重显存不减少，仅切分 activation |
| Patch Parallel | 复制（每张 GPU 持有完整权重） | 1（完整） | 同 SP |
| PP (PipeFusion) | 复制（每张 GPU 持有完整权重） | 1（完整） | 同 SP |

**对系统设计的影响**：

1. **GPU 角色切换成本**：SP/Patch/PP 策略下所有 GPU 持有相同权重，因此一张 GPU 可以从 denoise 角色切换到 encode/decode 角色（只需加载对应阶段的权重），无需重新分发并行权重。而 TP 策略下每张 GPU 只持有权重的一部分，角色切换需要先收集/重建完整权重，开销更大。这意味着在需要频繁动态调整 GPU 角色分配的场景下，SP 比 TP 更灵活。

2. **动态并行度调整**：TetriServe 的步级并行度调整（如从 P=4 变为 P=2）在 SP 下相对简单——多余的 GPU 直接释放，留下的 GPU 仍持有完整权重。在 TP 下调整并行度则需要重新切分和分发权重，代价显著更高。

3. **显存预算与模型规模**：TP 的唯一显存优势是权重切分——当模型太大（如 14B 视频 DiT）单 GPU 放不下完整权重时，TP 是必要的。但对于参数量适中的模型（如 DiT-XL 675M），SP 的权重复制不构成显存瓶颈，此时 SP 的灵活性优势更为突出。

**调度建议**：默认优先选择 SP-Ulysses（权重复制、灵活切换、可跨节点），仅在模型权重超出单 GPU 显存时 fallback 到 TP。在 TP 模式下应尽量减少动态并行度调整和角色切换的频率。

---

## 四、(P, B) 联合优化模型

### 4.1 核心洞察

并行度 P 和 batch size B 不是独立变量。对于 denoise 阶段：

- 高分辨率请求，batch=1 时单 GPU 即 compute-bound → 增加 B 不提升吞吐、反增延迟
- 引入并行（P=K）后，per-GPU 计算量降为 C/K → GPU 有 headroom → 可以塞入 B>1 的请求

因此最优策略是 **(P, B) 联合决策**：给定 K 张 GPU 分配给某个请求，找到最大化 throughput/SLO-attainment 的 (P, B) 组合。

### 4.2 不同并行策略创造的 batch 空间

不同并行策略对总计算量的影响不同，因此创造的 batch 空间也不同：

| 并行策略 | 总 FLOPs 变化 | 通信开销 | 创造的 batch 空间 | 质量影响 |
|----------|--------------|----------|-------------------|----------|
| TP | 不变 | O(model_size)/层 | ~K 倍（通信吃掉 15-20%） | 无损 |
| SP-Ulysses (exact) | 不变 | O(seq×d)/层 | ~K 倍（通信吃掉 10-20%） | 无损 |
| Patch (shifted) | 减少（O(N²)→O(N×W)） | O(boundary) | >K 倍（计算量本身减少） | 有损 |
| PP (PipeFusion) | 不变 | O(boundary)/步 | ~K 倍 + pipeline slot 复用 | 有损 (staleness) |

### 4.3 调度决策模型

对于每个到达的请求，调度器需要决策：

```
输入：请求特征（分辨率 R，模型 M，deadline D）
      集群状态（可用 GPU 数，当前负载，互联拓扑）

决策：
  1. 阶段路由：encode → denoise → decode
  2. Denoise 并行策略：P_type ∈ {TP, SP-Ulysses, Patch, None}
  3. Denoise 并行度：P ∈ {1, 2, 4, 8, ...}
  4. Denoise batch 位置：加入哪个正在运行的 batch
  5. Encode/decode batch 位置：加入哪个 batch

约束：
  - 端到端延迟 ≤ D（SLO）
  - GPU 显存不 OOM
  - 通信可以被异步流水线覆盖（并发请求数 ≥ T_comm/T_comp）

目标：
  - 最大化 SLO 达成率，或最大化吞吐
```

这可以建模为一个在线优化问题，用 cost model 或 roofline model 估计不同 (P, B) 组合的延迟和吞吐，选择最优解。

---

## 五、与现有工作的差异

| 工作 | 阶段解耦 | 并行策略 | 多 Batch | (P,B) 联合 | 通信隐藏 |
|------|----------|----------|----------|------------|----------|
| DisagFusion (当前) | ✓ | ✗ | ✗ | ✗ | ✓（跨阶段） |
| TetriServe | ✗ | ✓（仅 SP 度调整） | ✓（步级 packing） | 隐含 | ✗ |
| GenServe | ✗ | ✓（弹性 SP） | ✓（步级适配） | 隐含 | ✗ |
| PipeFusion | ✗ | ✓（patch PP） | ✗ | ✗ | ✓（跨步流水） |
| MixFusion | ✗ | ✓（patch 并行） | ✓（混合分辨率） | 隐含 | ✗ |
| **DisagFusion 扩展** | **✓** | **✓（TP/SP/Patch 可选）** | **✓（per-stage）** | **✓（显式联合）** | **✓（跨阶段+跨请求）** |

核心差异：
1. **阶段解耦 + 并行**：各阶段可独立扩缩容（encode 暴增不影响 denoise）、适配异构硬件（encode 放 L40、denoise 放 H100）、权重常驻避免切换开销、以及故障隔离。
2. **显式 (P,B) 联合优化**：TetriServe 隐含了部分 trade-off（deadline 紧时增大 P），但没有显式建模 batch size 的联合决策。
3. **通信隐藏的双重来源**：DisagFusion 已有的跨阶段异步流水线 + 新增的跨请求通信-计算重叠。

---

## 六、预期收益与验证计划

### 6.1 预期收益

- **延迟**：高分辨率请求通过并行策略（SP/TP）降低单请求延迟，通信被异步流水线覆盖后有效延迟进一步降低。
- **吞吐**：并行创造的 headroom 被多 batch 填充，GPU 利用率从并行后的下降恢复到接近饱和。
- **SLO 达成率**：(P,B) 联合优化可以根据每个请求的 deadline 选择最优配置，避免"一刀切"的并行策略。

### 6.2 验证计划

1. **Micro-benchmark**：测量不同分辨率、不同模型在 (P, B) 组合下的 per-step 延迟和 GPU 利用率，验证"并行创造 batch 空间"的假设。
2. **通信覆盖验证**：测量 SP-Ulysses / TP 在不同并发请求数下的有效通信延迟，验证异步流水线的覆盖效果。
3. **端到端评估**：在混合 workload（不同分辨率、不同 deadline）下，对比以下配置的 SLO 达成率和吞吐：
   - Baseline：DisagFusion（无并行、无 batch）
   - +SP only：denoise 阶段加 SP-Ulysses
   - +Batch only：所有阶段加大 batch
   - +SP+Batch（独立）：SP 和 batch 独立配置
   - +SP+Batch（联合）：(P,B) 联合优化
4. **CFG 方案对比**：对比 batch 合并、串行、GPU 子组三种 CFG 方案在不同分辨率下的延迟和显存。

---

## 七、开放问题

1. **(P,B) cost model 的构建**：如何建立一个轻量但准确的 cost model，输入 (R, M, P, B, K) 就能预测延迟和吞吐？是用 roofline model 还是 learning-based profiler？
2. **在线调度的计算开销**：per-request 的 (P,B) 决策需要在毫秒级完成，cost model 的推理时间和调度算法的复杂度是否可接受？
3. **异构 GPU 下的策略选择**：如果集群中有 A100 和 H100 混用，不同 GPU 的 compute/bandwidth 比值不同，最优 (P,B) 也会不同。如何做异构感知？
4. **与 AR+diffusion 混合生成架构的兼容性**：如果未来 denoise 阶段被替换为 AR 逐帧生成（如 CausVid），本策略框架是否仍然适用？AR 生成的 KV cache 管理如何与并行策略结合？
