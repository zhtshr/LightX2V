# Wan Pipeline Parallel（Layer PP）— 可行性分析与实现计划

> 日期：2026-06-05 | 状态：设计稿  
> **目标**：按 **Transformer 层** 将 Wan DiT 切到多卡（Layer-wise PP），使每卡常驻约 `1/pp_size` 的层权重，**消除 block CPU offload**；通过 **多请求 pipeline 填谷** 提升 GPU 利用率与系统吞吐。  
> **非目标**：PipeFusion 式跨 step/patch PP（权重复制、staleness 有损）— 另案处理。  
> **与 SP 关系**：PP **不替代** Ulysses SP 降延迟；可与 SP 组合 `world = pp_size × seq_p_size`。

---

## 1. 为什么要做 Layer PP

### 1.1 问题（与 TP 计划共享的背景）

| 现象 | 说明 |
|------|------|
| 14B MoE + `cpu_offload=block` | 单卡 24GB OOM（`cpu_offload=false`）；靠 PCIe 流式加载权重 |
| offload 饱和带宽 | comm overlap 仅 **3–4%**（Phase 3 MoE） |
| SP 不减权重 | Ulysses 每卡仍载 **全量** 40 层权重 → 必须 offload |
| 用户直觉「单请求对半砍，变成两个请求」 | PP 将层切到多卡；**单请求** wall 几乎不降；需 **≥2 个 in-flight 请求** 填满 stage 间 bubble |

### 1.2 Layer PP 能带来的收益

| 维度 | 预期 |
|------|------|
| **显存** | 每卡约 **num_layers/pp_size** 层权重；PP=2 时单 expert 约一半 → **有望无 offload** |
| **去 offload** | 同 TP，释放带宽给 overlap |
| **Stage 间通信** | 每 forward 仅 **pp_size−1 次** P2P 传 `x`（`seq×dim`），远少于 SP 每层 all_to_all |
| **实现成本** | **低于 TP**：主要切 `blocks` 列表 + P2P，不必改每个 Linear 的 col/row split |
| **Disagg 角色切换** | 每 rank 持有一段层，比 TP 分片权重 **更易** 与 encode/decode 角色切换共存 |

### 1.3 Layer PP 的代价

| 维度 | 说明 |
|------|------|
| **单请求延迟** | 1 个 microbatch：**不加速**（stage 串行）；wall ≈ Σ stage_time + P2P |
| **GPU bubble** | PP=2 单请求时约 **50%** 卡 idle；必须 **pipeline 调度** 多请求 |
| **调度复杂度** | 需 stage 队列 + 与 Disagg ready_pool / SP 槽拓扑协同 |
| **MoE** | `boundary_step` 切换 expert 时全 PP group 同步换 checkpoint |

### 1.4 与 TP 对比（选型）

| | Layer PP | Tensor Parallel |
|---|----------|-----------------|
| 减权重显存 | ✅ ÷ pp（按层） | ✅ ÷ tp（按矩阵维） |
| 单请求降延迟 | ❌ | 可能 ✅ |
| 实现难度（Wan） | **中** | **高** |
| 通信 | 少次 P2P（activation） | 每层 all-reduce |
| 吞吐 | 靠多请求填 pipeline | 靠 SP 式 overlap 或算力分摊 |
| 代码参考 | **无**（需新建） | LTX2 已有 |

**建议**：PP 作为 **更快落地** 的去 offload 路径；TP 作为需要 **单请求进一步降延迟** 或 PP 仍 OOM 时的增强。

### 1.5 与 PipeFusion 的区分

| | Layer PP（本文） | PipeFusion |
|---|------------------|------------|
| 切分对象 | **层**（blocks 0–19 / 20–39） | **空间 patch / 时间步** |
| 权重 | 每卡 **部分层** 的完整权重 | 每卡 **全量** 权重 |
| 质量 | 无损 | 有损（staleness） |
| 目标 | 显存 / offload | 减跨卡通信量 |

---

## 2. 可行性评估（Wan2.2-MoE I2V @ 8×A10）

### 2.1 显存粗算

- 配置：`num_layers=40`, `dim=5120`, `ffn_dim=13824`, int8 量化
- 单 expert 全量约 **~22GB**（实测 no-offload OOM 经验值）
- **PP=2**：每 stage ~20 层 → **~11GB** 权重 + activation（480×832, seq≈32760, 主激活 ~300MB 级）→ **24GB 内可行**
- **PP=4**：~5.5GB/ stage → 更宽裕，bubble 更大

**MoE 注意**：每个 denoise step 只用一个 expert；PP 组应在 `boundary_step` **barrier 后切换** 对应 expert 权重（或每 stage 只加载当前 expert 的层段）。

### 2.2 延迟

单请求、PP=2、1 个 forward in flight：

```text
GPU0 (stage0): [pre + blocks 0–19] ──P2P(x)──►
GPU1 (stage1): [blocks 20–39 + head] 
wall ≈ T_stage0 + T_p2p + T_stage1 ≈ T_single + O(ms)   # 无 2× 加速
```

**紧 SLO 仍依赖 SP**（已验证 P=4: 74s→25s）。PP 解决 **显存/offload/吞吐**，不解决单请求延迟。

### 2.3 通信量

- P2P payload：`x` shape `[seq_local, dim]`（若叠 SP 则 seq 已切分）
- 480×832, seq=32760, dim=5120, bf16：**~320MB / stage 边界 / forward**
- 相对 ~18s/step 计算，通常 **≪ 1%** wall（需 profile 验证）

### 2.4 吞吐与「变成两个请求」

PP=2、**2 个请求** pipeline 满负荷时：

```text
时刻 →  GPU0: req_B stage0 | req_A stage0 | req_B stage0 | ...
        GPU1: idle          | req_A stage1 | req_B stage1 | ...
```

- 理论 GPU 利用率 → **~100%**（2 个 in-flight）
- 系统吞吐 ≈ **2 × 单 stage 吞吐**（理想）
- 与 **双租户 SP overlap** 正交：SP 藏同 rank 上 comm；PP 藏跨 rank stage bubble

### 2.5 当前代码：**完全未实现**

| 项 | 现状 |
|----|------|
| `pipe_p_size` | **不存在** |
| `set_parallel_config` | 仅 `tensor_p` 或 `cfg_p × seq_p` |
| Wan weights | 每 rank 加载 **全部** `blocks[0..39]` |
| Wan infer | 顺序 `for block in blocks` 无 P2P |
| Disagg 调度 | 仅 SP 槽拓扑，无 stage 队列 |

---

## 3. 目标架构

### 3.1 并行拓扑

**Phase A（纯 PP）**

```json
"parallel": { "pipe_p_size": 2 }
```

- `world_size = pipe_p_size`
- `device_mesh = (pipe_p,)`
- rank `r` 持有 layers `[r * L/pp : (r+1) * L/pp)`

**Phase B（PP × SP）**

```json
"parallel": { "pipe_p_size": 2, "seq_p_size": 4 }
```

- `world_size = 8`
- `device_mesh = (pipe_p, seq_p)`
- PP 维：层切分；SP 维：Ulysses（现有 `wan/model.py` `_seq_parallel_pre/post`）

**Rank 职责**

| pp_rank | 职责 |
|---------|------|
| 0 | `pre_infer` + `blocks[0..L/pp-1]` + **send** `x` |
| 中间 | **recv** `x` + local blocks + **send** `x` |
| pp_size−1 | **recv** `x` + local blocks + `post_infer`（head/norm） |

### 3.2 激活传递

- 张量：`x` — 主 hidden state，`[seq, dim]`（或 SP 切后 `[seq/seq_p, dim]`）
- 算子：`dist.send` / `dist.recv` 或 `batch_isend_irecv`（NCCL P2P）
- **不需** 传 `pre_infer_out` 全结构：context / embed 处理见 §3.3

### 3.3 非 block 权重放置

| 组件 | 放置 |
|------|------|
| `pre_weight` + `pre_infer` | **pp_rank == 0** |
| `norm` + `head` + `post_infer` | **pp_rank == pp_size-1** |
| `context`（cross-attn K/V） | **每 stage 都需要** — Phase A：各 rank **broadcast** `context`（~769 token，小）；或 rank0 算 K/V 后 broadcast |
| `cos_sin`（rope） | 各 rank 本地相同表 |
| `embed` / `embed0`（modulation） | rank0 pre 产生；**随 `x` 一起传** 或各 stage broadcast 小 tensor |

### 3.4 多请求 Pipeline（Disagg 层）

```
stage_queues[pp_rank]: FIFO of (room, x_handle, meta)

on_request_admitted → stage0_queue
stage0 worker: pop → forward local layers → enqueue stage1
stage1 worker: pop → forward → done → release PP chain slot
```

- **硬约束**：同一 `(pp_rank, gpu)` 同时最多 1 个租户的 forward（与 SP 双租户可再叠：需验证 OOM）
- **填谷指标**：`pp_pipeline_util = busy_time / wall`；目标 ≥2 请求时 >80%

---

## 4. 计划修改的代码

### 4.1 配置与并行环境

| 文件 | 改动 |
|------|------|
| `lightx2v/utils/set_config.py` | ① 增加 `pipe_p_size`；② mesh 分支：`pipe_p_only` → `(pipe_p,)`；`pipe_p × seq_p` → 2D；`pipe_p × seq_p × cfg_p` → 3D；③ `config["pipeline_parallel"]=True`，`pp_group` / `pp_rank` / `pp_size` |
| `configs/dist_infer/wan22_moe_i2v_pp2.json` | `pipe_p_size=2`, `cpu_offload=false` |
| `scripts/disagg/run_phase1_transformer_bench.py` | `--pipe_p_size`；与 `--seq_p_size` 互斥或组合 |

### 4.2 Wan 模型 — 权重按层切分

| 文件 | 改动 |
|------|------|
| `lightx2v/models/networks/wan/weights/transformer_weights.py` | ① 读取 `pp_rank/pp_size`；② `blocks` 只实例化 `range(start, end)`；③ `head/norm` 仅 `pp_rank==last` 创建；④ `register_offload_buffers`：PP 下 **禁用** block offload 或按 local layers 缩小 buffer |
| `lightx2v/models/networks/wan/weights/pre_weights.py` | 仅 rank0 加载；其他 rank 空壳或跳过 load |
| `lightx2v/models/networks/wan/model.py` | ① `pp_group` 初始化；② `_infer_cond_uncond` 改为 PP 路径：rank0 pre → PP forward → last rank post；③ 与 `seq_parallel` pre/post 嵌套：先 SP chunk（seq 维），再 PP 传 `x` |

### 4.3 Wan 推理 — Stage 间 P2P

| 文件 | 改动 |
|------|------|
| `lightx2v/models/networks/wan/infer/transformer_infer.py` | ① 新增 `infer_pp_stages(blocks, x, pre_infer_out)`；② `infer_main_blocks` 根据 `pipeline_parallel` 分支：local `infer_block` 循环 vs PP；③ 边界：`dist.isend/irecv` + `compute_stream` 同步 |
| 新建 `lightx2v/models/networks/wan/infer/pipeline_parallel.py` | `PPContext`：`send_activation` / `recv_activation` / `stage_barrier`；封装 P2P tags（`pp_rank`, `step_index`） |
| `lightx2v/models/networks/wan/infer/pre_infer.py` | 仅 `pp_rank==0` 执行；输出通过 `PPContext` 下发 |
| `lightx2v/models/networks/wan/infer/post_infer.py` | 仅 `pp_rank==last` 执行 |
| `lightx2v/models/networks/wan/infer/offload/transformer_infer.py` | PP 模式下 **assert not cpu_offload** 或 fallback 报错 |

**`infer_cross_attn` PP 注意**：`context` 在每层都要做 K/V — 各 PP stage 的 block 都含 cross-attn。context tensor 在 pre 后可 **`dist.broadcast(context, src=pp_first_rank, group=pp_group)`** 一次 per step。

### 4.4 MoE expert 切换

| 文件 | 改动 |
|------|------|
| `lightx2v/models/networks/wan/model.py` | `boundary_step` 切换 high/low noise 模型时：`dist.barrier(pp_group)`；各 rank `load_weights` 仅自己层段（或全体切换 scheduler 指向的 `WanModel` 实例） |
| `lightx2v/disagg/utils.py` | `load_wan_transformer`：MoE + PP 时加载策略文档化 |

### 4.5 Disagg 服务与调度

| 文件 | 改动 |
|------|------|
| `lightx2v/disagg/parallel/context.py` | **新建**（若尚无）：`ParallelPlan` 增加 `pipe_p_size`, `pp_rank`, `pp_stage_ranks` |
| `lightx2v/disagg/services/transformer.py` | ① PP 组内 rank 固定链式 `member_ranks=[r0,r1,...]`；② 可选 **stage 本地队列** 消费 phase1；③ 一个 logical 请求占用 **整条 PP 链** pp_size 张卡 |
| `lightx2v/disagg/services/controller.py` | `run_batch`：PP 链作为 **原子槽位**（容量 1 链 = 1 租户）；与 SP 叶槽区别：PP 链 rank 连续且有序 |
| `lightx2v/disagg/scheduler/cost_model.py` | **新建/扩展**：`T_i(pp, sp)`；单请求延迟 ≈ `sum(stage_time)`，不随 pp 降；吞吐模型含 `inflight_req ≥ pp` |
| `lightx2v/disagg/scheduler/batch_placer.py` | 费用流边：PP 链占用 `pp_size` 个 GPU；`fit_p` 可扩展 `fit_pp`（显存够则 pp=1，否则 pp=2） |
| `lightx2v/disagg/dynamic_sp_scheduling_plan.md` | 增 §PP：槽类型 `PPChain` vs `SPSlot`；攒批时 PP 链整链分配 |

### 4.6 测试与基准

| 文件 | 改动 |
|------|------|
| `scripts/disagg/run_phase1_pp_sweep.sh` | **新建**：pp∈{1,2,4}，测 OOM、延迟、P2P 时间 |
| `scripts/disagg/run_phase3_pp_pipeline_bench.py` | **新建**：2 请求填 PP=2 pipeline，测吞吐 vs 2×单卡 |
| `save_results/optimization_study/` | `phase1_wan22_pp_scaling.md` 结果模板 |

---

## 5. 分阶段实施

| 阶段 | 内容 | 验收 |
|------|------|------|
| **PP-0** | `set_config` + `PPContext` P2P 单测（随机 tensor） | 2 rank send/recv 正确 |
| **PP-1** | Wan 权重只 load 半层；静态 forward 无数值对比（rank0→rank1） | PP=2 单请求 E2E 输出与 P=1 **bitwise 接近** |
| **PP-2** | MoE 14B `cpu_offload=false` PP=2 | **不 OOM**；4 step 跑完 |
| **PP-3** | 2 请求 pipeline scheduler（单机 mock） | GPU 利用率较单请求 ↑；吞吐 >1.5× |
| **PP-4** | PP=2 × SP=2（4 卡） | 延迟由 SP 贡献；显存仍无 offload |
| **PP-5** | Disagg Controller 下发 PP plan + overlap 对比 | 无 offload 下 dual-overlap >10%（目标） |

---

## 6. 与 SP / 调度 / overlap 的集成图

```text
                    ┌─── Disagg Controller ───┐
                    │  Admission + run_batch   │
                    │  (urgency + fit_p)       │
                    └───────────┬─────────────┘
                                │ ParallelPlan
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
     Encoder(×1)          PP chain + SP          Decoder(×1)
                          ┌── rank0 SP slice ──P2P──► rank1 SP slice ──┐
                          │  tenant A comm ∥ tenant B compute (SP)    │
                          └───────────────────────────────────────────┘
```

- **SP**：降单请求延迟 + 双租户藏 all_to_all
- **PP**：降权重 / 去 offload + 多请求藏 stage bubble
- **调度**：PP 链占 `pp_size` GPU；`fit_p` 选 sp；`fit_pp` 选是否拆层（显存/SLO）

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 单请求 PP 无延迟收益 | 文档/调度明确：SLO 靠 SP，PP 靠显存与吞吐 |
| MoE 切换错权 | `boundary_step` 全 group barrier + 一致 checkpoint 路径 |
| P2P 死锁 | 固定 send/recv 顺序；使用 tag；timeout 日志 |
| PP+SP+双租户 OOM | 限制每卡 1 compute 租户；profile 后放开 |
| 与 PipeFusion 混淆 | 命名 **Layer PP**；PipeFusion 单独 milestone |
| PP 链碎片化 | 调度器以 **整链** 分配；暂不做运行期拆链（同 SP plan §11） |

---

## 8. PR 顺序（建议）

```
PR-PP-1  set_config pipe_p mesh + pipeline_parallel.py P2P 原语
PR-PP-2  WanTransformerWeights 按 pp_rank 切 blocks + head 末段
PR-PP-3  WanModel / transformer_infer PP forward + 数值对齐
PR-PP-4  MoE 14B PP=2 无 offload 基准
PR-PP-5  双请求 stage pipeline（bench 脚本）
PR-PP-6  PP × SP 4 卡 + Disagg ParallelPlan
PR-PP-7  Controller PP 链槽位 + cost_model T_i(pp,sp)
```

---

## 9. 参考

- `lightx2v/models/networks/wan/infer/transformer_infer.py` — 现有逐 block 顺序执行
- `lightx2v/models/networks/wan/infer/offload/transformer_infer.py` — block offload（PP 要替代的路径）
- `lightx2v/disagg/wan_tp_implementation_plan.md` — 备选 TP 去 offload
- `lightx2v/disagg/dynamic_sp_scheduling_plan.md` — SLO 费用流（urgency + fit_p）
- `lightx2v/disagg/disagfusion_extension_idea.md` — PipeFusion 对比（§3.5、§4.2）
- `save_results/optimization_study/phase3_sp_scaling.md` — SP 延迟收益基线
- `save_results/optimization_study/phase3_dual_overlap_no_offload.md` — offload 对 overlap 的压制
