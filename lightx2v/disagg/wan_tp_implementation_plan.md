# Wan Tensor Parallel（TP）— 可行性分析与实现计划

> 日期：2026-06-05 | 状态：设计稿  
> **目标**：在 Wan2.x（尤其 Wan2.2-MoE I2V 14B）上实现**真正的权重分片 TP**，使单卡显存可放下 `1/tp_size` 权重，**消除 block offload**，从而恢复 GPU 带宽供 NCCL / 双租户 comm overlap 使用。  
> **参考实现**：`lightx2v/models/networks/ltx2/`（已接入 `MMWeightTP` + `_load_weights_from_rank0`）  
> **实测基线**：`save_results/optimization_study/phase1_wan22_tp_scaling.md`

---

## 1. 为什么要做 TP

### 1.1 问题

| 现象 | 数据 / 代码 |
|------|-------------|
| 当前 `tensor_p_size>1` **无加速** | TP sweep：72.9s @ TP=1/2/4/8，speedup **1.00×** |
| 根因 | `set_config.py` 仅建 `tensor_p` mesh；**Wan weights/infer 未分片**，每 rank 仍载全量 |
| 14B MoE 无 offload OOM | 单卡 24GB 无法同时驻留 high+low 两套 expert（各 ~22GB 量级） |
| offload 下 overlap 失效 | 双租户 a2a-overlap 仅 **1.03–1.04×** vs serial（Phase 3） |
| SP 不能减权重显存 | Ulysses 每卡 **完整权重**，只切 activation；P=4 仍要 block offload |

### 1.2 TP 能带来的收益

| 维度 | 预期 |
|------|------|
| **显存** | 线性层权重约 **÷ tp_size**；`tp_size` 越大每卡权重越小（如 MoE 14B 在 24GB 上常选 tp∈{2,4,5,8}） |
| **去 offload** | `cpu_offload=false` 后 PCIe 权重搬运消失 → comm overlap 空间恢复（T2V 无 offload 下 overlap 可达 **~35%**） |
| **单请求延迟** | 理论上有一定加速（每层 all-reduce）；A10 上需实测，可能不如 SP 理想 |
| **与 SP** | 可组合 `world = tp_size × seq_p_size`（节点内 TP + SP） |

### 1.3 TP 的代价

| 维度 | 说明 |
|------|------|
| **通信** | 每个 Linear **row-split** 输出需 **all-reduce**；40 层 × 多 Linear，带宽压力大 |
| **实现复杂度** | 高于 Layer PP：所有 `MMWeight`、RMSNorm、部分 attention 路径要改 |
| **Disagg 灵活性** | 权重分片后 GPU 角色切换 / 动态改并行度成本高（需 gather 全量权重） |
| **量化** | int8-q8f 的 `MMWeightTP` 需验证 split 后仍走正确 kernel |

### 1.4 与 PP 的分工

- **TP**：切 **矩阵维度**，减权重、可能略降延迟；实现重，参考 LTX2。
- **Layer PP**：切 **层段**，减权重；单请求延迟几乎不降，靠多请求填 pipeline 提吞吐；实现相对轻。

**建议**：先选 **满足整除约束的最小 tp_size** 使无 offload 跑通，再按 SLO/吞吐需要增大 tp 或叠 SP；若 TP 工程周期过长，可并行推进 Layer PP（见 `wan_pp_implementation_plan.md`）。

---

## 2. 当前代码状态

### 2.1 已有（可复用）

| 组件 | 路径 | 说明 |
|------|------|------|
| `MMWeightTP` | `lightx2v/common/ops/mm/mm_weight.py` | col/row split + row 侧 all-reduce |
| `RMSWeightTP` | `lightx2v/common/ops/norm/rms_norm_weight.py` | 按 hidden 维切分 + all-reduce 方差 |
| LTX2 TP 全流程 | `lightx2v/models/networks/ltx2/model.py` | `_load_weights_from_rank0`、`_split_weight_for_tp`、`_is_tp_weight` |
| LTX2 TP weights | `lightx2v/models/networks/ltx2/weights/transformer_weights.py` | 各 Linear 用 `TensorParallel` 注册 |
| 并行 mesh（仅 TP 维） | `lightx2v/utils/set_config.py` | `tensor_p_size == world_size` 时 1D `tensor_p` mesh |
| TP bench 脚本 | `scripts/disagg/run_phase1_transformer_bench.py` | `--tensor_p_size` 探测（当前测到假 TP） |

### 2.2 缺失（Wan 特有）

| 组件 | 现状 |
|------|------|
| `WanTransformerWeights` | 全部 `MM_WEIGHT_REGISTER[mm_type]`（Default/int8），**无 TP** |
| `WanModel` | 无 `_load_weights_from_rank0`、无 `_tensor_parallel_pre_process` |
| `WanTransformerInfer` | 无 `tp_group`；self-attn 走 Ulysses 或 flash，**未与 TP head 切分对齐** |
| `WanPreWeights` / `head` | 未定义 TP 切分策略 |
| Disagg `transformer.py` | 未区分 TP 组启动与 rank 绑定 |
| int8 `MMWeightTP` | 需确认 `mm_type` 传入 `TensorParallel` 包装后 quant apply 正确 |

---

## 3. 目标架构

### 3.1 并行度 `tensor_p_size`（不限于 2）

实现支持 **任意合法 `tensor_p_size`**（与 `world_size` 或 `tp×sp×cfg` 分解一致），不限定为 2。选型由 **显存** 与 **整除约束** 决定，而非写死配置。

**通用约束**（启动时校验）：

| 约束 | 说明 |
|------|------|
| `tensor_p_size ≥ 1` | 1 = 单卡，无 TP |
| `tensor_p_size == world_size`（纯 TP） | 或 `cfg_p × tensor_p × seq_p == world_size`（混合） |
| `num_heads % tensor_p_size == 0` | TP 按 head 维切分 Q/K/V |
| `dim % tensor_p_size == 0` | col/row 切 hidden 维 |
| `ffn_dim % tensor_p_size == 0` | FFN 中间维 col 切分 |

**常见模型合法 tp（MoE I2V：`num_heads=40`, `dim=5120`, `ffn_dim=13824`）**：

| `tensor_p_size` | 每卡 heads | 典型用途 |
|---:|---:|---|
| 1 | 40 | 基线 / 仅 SP |
| 2 | 20 | 最小 TP，14B 去 offload 首选探测 |
| 4 | 10 | 显存仍紧或叠 SP 前留余量 |
| 5 | 8 | 40 可被 5 整除 |
| 8 | 5 | 8 卡纯 TP 或 `tp=2×sp=4` 中的 tp |
| 10 / 20 / 40 | — | 合法但 A10 集群少见 |

**T2V-1.3B（`num_heads=12`）**：合法 tp ∈ {1, 2, 3, 4, 6, 12}；与 SP 组合时需同时满足 `num_heads % seq_p == 0`。

**显存选型（粗估）**：单 expert 权重约 **W** GB；TP 后每卡约 **W / tp_size** + activation。选 **最小 tp** 使 `W/tp_size + act ≤ 24GB`，若仍 OOM 再增大 tp。

### 3.2 并行拓扑示例

**Phase A（纯 TP，`world = tp`）**

```json
"parallel": { "tensor_p_size": 4 }
```

- `world_size = tensor_p_size`（可为 2 / 4 / 8 / …）
- `device_mesh = (tensor_p,)`

**Phase B（TP × SP 混合，`world = tp × sp`）**

```json
"parallel": { "tensor_p_size": 2, "seq_p_size": 4 }
```

或 8 卡：`tensor_p_size: 4, seq_p_size: 2` 等，只要 `tp × sp == world_size`。

- `device_mesh = (tensor_p, seq_p)`（需扩展 `set_parallel_config`）
- 约束：`num_heads % tensor_p_size == 0` **且** `num_heads % seq_p_size == 0`

**Phase C（TP × SP × cfg，可选）**

- `world = cfg_p × tensor_p × seq_p`；例如 8 卡：`cfg=2, tp=2, sp=2`。

**权重加载与通信**：逻辑与 tp 无关，均为 **切 `tp_size` 份** + rank 0 `send` 到 `rank 1..tp_size-1`；每层约 **3× all-reduce / rank 组**（与 tp 大小无关，次数固定；单次 payload 随 tp 变化）。

### 3.3 Megatron 式切分规则（Wan DiT block，任意 tp_size）

每个 `WanTransformerAttentionBlock` 内 Linear 建议切分（与 LTX2 一致）：

| 模块 | 权重 | split | 通信 |
|------|------|-------|------|
| `self_attn_q/k/v` | col | 输出维 ÷ **tp_size** | 无 |
| `self_attn_o` | row | 输入维 ÷ **tp_size** | **all-reduce** |
| `self_attn_norm_q/k` | TP RMS | hidden ÷ **tp_size** | all-reduce（norm 内） |
| `cross_attn_q` | col | | 无 |
| `cross_attn_k/v` | **不切** 或 col（context 短，可选不切） | context 全 rank 相同 |
| `cross_attn_o` | row | | **all-reduce** |
| `ffn_0` | col | | 无 |
| `ffn_2` | row | | **all-reduce** |
| `head`（最后） | row | 仅 last rank 或 col+gather | 设计时二选一 |

**Self-attention 计算**（TP 下）：

- Q/K/V col-split → 每 rank `num_heads / tp_size` 个 head（**任意 tp_size** 满足整除即可）
- Attention 在 **local heads** 上算（序列维仍完整，直到叠 SP）
- O projection row-split + all-reduce

**与 Ulysses 叠 SP 时**：

- 先 TP 切 head，再 Ulysses 在 seq 维 all_to_all（顺序：实现中需在 `infer_self_attn` 明确：TP 组内 head 分片 + SP 组内 seq 分片）
- 建议 Phase A 仅 TP，Phase B 再叠 SP。

### 3.4 权重加载（任意 tp_size）

复用 LTX2 模式，**`_split_weight_for_tp(..., tp_size)` 参数化**，不写死 2 份：

1. **rank 0** 从 safetensors 读全量；
2. `_is_tp_weight(key)` 判定 Wan key 是否需要切分；
3. `_split_weight_for_tp(key, weight, tp_size)` 按 col/row 切成 **tp_size** 块；
4. rank 0 保留 `__tp_rank_0`，对 `rank_idx in 1..tp_size-1` 依次 `dist.send`；各 rank `recv` 自 rank 0；
5. 非 TP 权重（modulation 等）**broadcast** 到全体 TP rank。

**约束**：TP 路径下 **`cpu_offload=false`**（与 LTX2 一致）；MoE 每次只激活一个 expert 的 checkpoint。

**walkthrough（示例 tp_size=2）**：仅文档说明用；实现与 **tp_size=4/8** 相同，只是切分为 4/8 块并多几次 send/recv。

**任意 tp 的 per-rank shape（纯 TP、无 SP）**：

```text
local_heads = num_heads / tp_size
head_dim    = dim / num_heads   # 不变
Q/K/V out:   [seq, local_heads, head_dim]
FFN ffn_0 out: [seq, ffn_dim / tp_size]
每层 all-reduce 次数: 3（self_o, cross_o, ffn_2），与 tp_size 无关
```

---

## 4. 计划修改的代码

### 4.1 配置与并行环境

| 文件 | 改动 |
|------|------|
| `lightx2v/utils/set_config.py` | 扩展 `set_parallel_config`：支持 `(tensor_p, seq_p)` 2D mesh；`tensor_parallel` / `seq_parallel` 可同时为 true；world 分解断言 |
| `configs/dist_infer/wan22_moe_i2v_tensorp*.json` | 新增 TP 基准配置族（`cpu_offload=false`, `tensor_p_size` ∈ {2,4,5,8} 等） |
| `scripts/disagg/run_phase1_transformer_bench.py` | TP 路径强制 `load_from_rank0` / 初始化逻辑；验收指标 |

### 4.2 Wan 模型主体

| 文件 | 改动 |
|------|------|
| `lightx2v/models/networks/wan/model.py` | ① `__init__` 取 `tp_group/tp_rank/tp_size`（仿 LTX2）；② 实现 `_load_weights_from_rank0`、`_is_tp_weight`、`_get_split_type`、`_split_weight_for_tp`（Wan key 命名：`blocks.N.self_attn.q.weight` 等）；③ `_should_load_weights` 仅 rank0 读盘；④ 可选 `_tensor_parallel_pre_process`（若 head 也切 seq）；⑤ `infer()` 中与 `cpu_offload` / MoE expert 切换协调 |
| `lightx2v/models/networks/base_model.py` | 若 `_load_weights_from_rank0` 可上提到基类则抽取 Wan/LTX2 公共逻辑；否则先在 `WanModel` 复制 LTX2 再 refactor |

### 4.3 权重模块（核心工作量）

| 文件 | 改动 |
|------|------|
| `lightx2v/models/networks/wan/weights/transformer_weights.py` | ① `WanTransformerWeights.__init__`：若 `config["tensor_parallel"]`，向 block 传 `tp_group/tp_rank/tp_size`；② `WanSelfAttention` / `WanCrossAttention` / `WanFFN`：Linear 换 `MM_WEIGHT_REGISTER["TensorParallel"](mm_type=..., split_dim=...)`；③ RMS norm 换 `RMS_WEIGHT_REGISTER["TensorParallel"]`；④ `head` 的 TP 策略（建议 row-split + all-reduce 或仅 tp_rank==0 持有）；⑤ offload buffer 路径在 TP 下 **禁用** 或按 tp 分片复制 |
| `lightx2v/models/networks/wan/weights/pre_weights.py` | `pre` 阶段 Linear 是否 TP：一般 **broadcast 小权重** 或 rank0 算完 broadcast embedding（Phase A 可 rank0-only pre_infer + broadcast `pre_infer_out`） |

**Wan key → split 映射表**（实现 `_is_tp_weight` 时维护）：

```python
TP_COL_SUFFIXES = (
    ".self_attn.q.weight", ".self_attn.k.weight", ".self_attn.v.weight",
    ".cross_attn.q.weight",
    ".ffn.0.weight",
    # k/v: 可选不切
)
TP_ROW_SUFFIXES = (
    ".self_attn.o.weight",
    ".cross_attn.o.weight",
    ".ffn.2.weight",
    ".head.head.weight",
)
TP_NORM_SUFFIXES = (
    ".self_attn.norm_q.weight", ".self_attn.norm_k.weight",
    ".cross_attn.norm_q.weight", ".cross_attn.norm_k.weight",
)
```

### 4.4 推理路径

| 文件 | 改动 |
|------|------|
| `lightx2v/models/networks/wan/infer/transformer_infer.py` | ① `__init__` 缓存 `tp_group/tp_size`；② `infer_self_attn`：Q/K/V/O 已由 `MMWeightTP` 处理通信；本地 `num_heads = global_num_heads / tp_size`；③ `infer_cross_attn`：K/V 若不切 TP，保持现状；④ 纯 TP 阶段关闭 `seq_parallel`；⑤ TP×SP 阶段按 mesh 维顺序协调 |
| `lightx2v/models/networks/wan/infer/pre_infer.py` | TP 下 rank0 执行 pre 或全体执行（若 pre 权重 broadcast） |
| `lightx2v/models/networks/wan/infer/post_infer.py` | 若 head 仅在 last TP rank：中间 rank 跳过；或 head TP all-reduce 后全体 post |
| `lightx2v/common/ops/mm/mm_weight.py` | 验证 int8 `mm_type` 在 `MMWeightTP` 内 `apply` 路径；必要时为 Wan int8 增测试 |

### 4.5 Attention 算子

| 文件 | 改动 |
|------|------|
| `lightx2v/common/ops/attn/*.py` | TP 下每 rank `num_heads = global_heads / tp`；flash/sage 接口传入 local head 数；**Ulysses 叠加**时与 `ulysses_attn.py` 协调 head 维与 seq 维切分顺序 |

### 4.6 Disagg 集成

| 文件 | 改动 |
|------|------|
| `lightx2v/disagg/utils.py` | `load_wan_transformer`：TP 时 `init_parallel_env` + `set_parallel_config`；`load_from_rank0=true` |
| `lightx2v/disagg/services/transformer.py` | 多 rank TP 组：各 rank 同 `group_id`，`member_ranks` 为 TP 链；phase1 payload 带 `tensor_p_size` |
| `lightx2v/disagg/examples/run_service.py` | 启动 N 个 transformer 进程组成 TP 组（仿 SP 多 rank，但 mesh 维不同） |
| `lightx2v/disagg/dynamic_sp_scheduling_plan.md` | 后续：扩展 `ParallelPlan` 含 `tensor_p_size`（与 `seq_p_size` 独立）；Admission `T_i(tp, sp)` 二维表；合法 tp 集合按模型配置生成 |

### 4.7 测试与基准

| 文件 | 改动 |
|------|------|
| `scripts/disagg/run_phase1_tp_sweep.sh` | 扫 **tp ∈ {1,2,4,8}**（及 MoE 合法的 5）；各 tp **< TP=1 延迟**（或 tp 仅用于 OOM 通过）且 **不 OOM** |
| `save_results/optimization_study/phase1_wan22_tp_scaling.md` | 实现后重跑；对比 **扩展效率 = speedup/tp** 随 tp 变化 |
| 新增 `scripts/disagg/run_phase3_tp_overlap.sh` | 多 tp 无 offload + 双租户 overlap；对比 offload SP |

---

## 5. 分阶段实施

各阶段 **`tensor_p_size` 为配置参数**；下表「示例 tp」仅作首轮验收，通过 sweep 覆盖多档 tp。

| 阶段 | 内容 | 验收 |
|------|------|------|
| **TP-0** | `set_config` 支持任意合法 tp + 2D mesh + bench launch | `torchrun --nproc_per_node={tp}`，tp∈{2,4} 不 crash |
| **TP-1** | Wan `WanFFN` + `head` 单 block 单测 | **tp=2 与 tp=4** 均数值对齐单卡（atol 阈值） |
| **TP-2** | 全 40 block + self/cross attn TP | T2V-1.3B **tp∈{2,4}** 端到端；各档延迟 < 同卡 P=1 |
| **TP-3** | Wan2.2-MoE 无 offload | **最小可行 tp**（通常 2，不够则 4/5/8）不 OOM；sweep 记录 tp–显存–延迟 |
| **TP-4** | TP × SP 混合 | 例：8 卡 `tp=2,sp=4` 或 `tp=4,sp=2`；相对纯 SP / 纯 TP 的 Pareto |
| **TP-5** | Disagg 集成 | Controller 下发 `tensor_p_size`；E2E 支持多档 tp plan |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| int8 TP split 数值/性能异常 | 先 BF16/FP16 对齐，再切 int8 |
| 非法 tp（不整除 heads/dim/ffn） | 启动时 `assert` + 文档表 3.1；Disagg plan 只生成合法 tp |
| tp 过大 → 每卡算力过少、all-reduce 占比升 | sweep 选 **最小满足显存的 tp**；大 tp 不必追求线性加速 |
| all-reduce 次数/层固定，大 tp 时单次 payload 变小 | 与 SP 对比选优；TP 首要目标是 **放得下**，其次才是加速 |
| MoE 双 expert | 每 expert 独立 load；`boundary_step` 全 TP group barrier 后切换 |
| 与现有 Ulysses 冲突 | Phase 顺序：先纯 TP，再 TP×SP；文档写明 mesh 维顺序 |

---

## 7. PR 顺序（建议）

```
PR-TP-1  set_config 2D mesh + WanModel 权重分发骨架（抄 LTX2）
PR-TP-2  WanFFN / self_attn Linear → MMWeightTP
PR-TP-3  cross_attn + head + pre/post
PR-TP-4  全模型数值对齐 + phase1 bench TP sweep
PR-TP-5  MoE 14B 无 offload + dual-overlap 基准
PR-TP-6  Disagg transformer 多 rank TP 启动
```

---

## 8. 参考

- `lightx2v/models/networks/ltx2/model.py` — `_load_weights_from_rank0`
- `save_results/optimization_study/phase1_wan22_tp_scaling.md` — 当前假 TP 探测
- `save_results/optimization_study/phase3_dual_overlap_no_offload.md` — offload 对 overlap 的压制
- `lightx2v/disagg/wan_pp_implementation_plan.md` — 备选：Layer PP 去 offload
- `lightx2v/disagg/dynamic_sp_scheduling_plan.md` — SLO 调度（后续扩展 `tensor_p_size`）
