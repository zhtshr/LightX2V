# Disagg 动态 Ulysses SP + SLO 调度 — 实现与算法设计计划

> 日期：2026-06-02 | 状态：设计稿  
> **范围**：静态 `static_instance_slots`；encode 后再绑 `parallel_plan`；**攒批 min-cost max-flow** 槽位分配 + 双流多租户 overlap（Phase C）。**Autoscale / 运行期改 p** 见 [§11](#11-后续)。

---

## 1. 目标与范围

| # | 能力 |
|---|------|
| 1 | Disagg 内 Ulysses SP（Controller 拉 rank，不用 `torchrun`） |
| 2 | 跨请求 **双流**：某租户 **comm** 时，同 rank 上可跑其它租户的 **compute** |
| 3 | SLO 调度：最大化按时完成率；不可行 **reject** |

**前提**：单 Encoder / 单 Decoder；phase1 多 consumer 已修复；**不做** GPU 利用率/带宽闭环（用 ring 深度 + slot 账本即可）。

**不做**：Autoscale、运行中换 `p`（§11）、enc/dec 多实例、VAE batch、collective 中途换组、CP-SAT、per-vacancy 即时贪心补位。

---

## 2. 架构

### 2.1 数据流

```
Client → Controller
           ├─ admit → [request ring] → Encoder
           │              encode_done → ready_pool
           └─ run_batch (攒批) → place + plan → [phase1 ring] → Transformer(s) → [phase2 ring] → Decoder
```

- **request ring**：`slo_deadline_ts` + 任务参数，**无** `parallel_plan`。
- **encode_done**：Encoder `notify_encode_done`；请求进入 **ready_pool**，**不**立即写 phase1。
- **phase1 ring**：`run_batch` 匹配成功后由 Controller `produce`（含 `parallel_plan` + 槽位 `member_ranks`）。Phase A 可手填 plan 跳过调度器。

### 2.2 角色

| 组件 | 职责 |
|------|------|
| **Controller** | admission、reject、**ready_pool**、**partition 槽拓扑**、攒批触发、`run_batch`（flow + 可选 split/merge）、phase1 produce |
| **Encoder / Decoder** | FIFO 消费 ring |
| **Transformer leader** | 多租户双流；comm 租户 collective 全员参与；**leader 广播** collective 顺序；执行 Controller 下发的 `(room, plan)` |

### 2.3 状态机

`PENDING` → `IN_REQUEST_RING` → `ENCODING` → `ENCODE_DONE`（在 ready_pool 等 placement）→ `PHASE1_QUEUED` → `TRANSFORMER` → … → `COMPLETED` / `REJECTED`。

`PHASE1_QUEUED` 起 plan **pinned**；in-flight 不抢占。重排仅允许 `PENDING` / `ENCODE_DONE`（未下发 phase1 前）。

### 2.4 分区与槽拓扑

- 物理 **partition**（盘）：如 P0 = rank `{1..4}`，P1 = rank `{5..8}`；各盘独立攒批、独立拓扑 `π`。
- 每盘维护 **槽树** `π`：叶节点 = 可放置租户的逻辑槽，带 `member_ranks`、`p = |ranks|`。
- 启动时 `GroupRegistry` 预建 `sp_groups` 中各 `(group_id, p, member_ranks)` 对应的 NCCL PG。
- 初始 `π` 可配置（如 P0 默认 `2+2`：`{1,2}`、`{3,4}`），后续由 **batch 内至多 1 次** split/merge 演化。

```json
{
  "sp_groups": {
    "p0_p4_r1234": { "member_ranks": [1,2,3,4], "leader_rank": 1 },
    "p0_p2_r12":   { "member_ranks": [1,2],     "leader_rank": 1 },
    "p0_p2_r34":   { "member_ranks": [3,4],     "leader_rank": 3 },
    "p1_p4_r5678": { "member_ranks": [5,6,7,8], "leader_rank": 5 }
  },
  "partitions": {
    "p0": { "ranks": [1,2,3,4], "initial_topology": "2+2" }
  },
  "comm_slots_per_group": 2,
  "scheduler_tick_ms": 30,
  "batch_min_idle_slots": 3,
  "batch_wait_ms": 1000,
  "topology_cooldown_ms": 5000,
  "split_penalty": 1.0,
  "merge_penalty": 1.0
}
```

参数取值按系统负载与 SLO 调优；`batch_min_idle_slots` 宜与盘内槽数适配（如 2+2 仅 2 叶时可用 `min(配置值, num_leaves)` 或按 **槽容量单位** `Σ p_slot` 触发）。

静态阶段 `cell_id=None`（cell 仅 autoscale 时用，见 §11）。

---

## 3. 模块与数据结构

| 文件 | 职责 |
|------|------|
| `parallel/context.py` | `ParallelPlan`, `ParallelContext` |
| `parallel/group_registry.py` | 预建 PG lookup |
| `parallel/group_scheduler.py` | leader：comm/compute 双流、多租户 runtime |
| `scheduler/cost_model.py` | `T_i(p)`、`fit_p`、`finish` 预测 |
| `scheduler/resource_ledger.py` | ring 深度、partition 槽占用 |
| `scheduler/slot_topology.py` | 槽树 `π`、split/merge 邻居生成 |
| `scheduler/batch_placer.py` | 攒批触发、`min_cost_max_flow` |
| `scheduler/slo_scheduler.py` | admission、`run_batch` 编排 |
| `scheduler/request_state.py` | `RequestRecord`、状态 |

```python
@dataclass
class Slot:
    slot_id: str
    member_ranks: set[int]
    p: int
    tenant_room: int | None   # 当前占用；None 表示空闲

@dataclass
class PartitionState:
    partition_id: str
    ranks: set[int]
    slots: list[Slot]           # 当前拓扑 π 的叶槽列表
    idle_slots: set[str]        # 空闲叶 slot_id
    batch_timer_start: float | None
    last_topology_change_ts: float

@dataclass
class ParallelPlan:
    group_id: str
    slot_id: int
    seq_p_size: int
    member_ranks: list[int]
    attn_type: str = "ulysses"

@dataclass
class RequestRecord:
    room: int
    deadline_ts: float
    state: RequestState
    parallel_plan: ParallelPlan | None
    laxity: float
```

**phase1 payload 示例**（placement 成功后）：

```json
{
  "data_bootstrap_room": 123,
  "slo_deadline_ts": 1717300000.0,
  "parallel_plan": {
    "group_id": "p0_p2_r12",
    "member_ranks": [1, 2],
    "seq_p_size": 2,
    "slot_id": 0
  }
}
```

---

## 4. 代码改动

| 模块 | 改动 |
|------|------|
| `controller.py` | `SLOScheduler`、`ready_pool`、`PartitionState`、`run_batch`、phase1 produce |
| `encoder.py` | `notify_encode_done(room, metrics)`；不写 phase1 |
| `transformer.py` | 多租户 `RequestRuntime`；按下发 plan 绑定 `ParallelContext` |
| `run_service.py` | rank env，无 `torchrun` |

**Wan 模型（去掉静态 `seq_p_group`）**：

| 文件 | 改动 |
|------|------|
| `wan/infer/pre_infer.py` | 去掉 init 缓存 `seq_p_group` |
| `wan/infer/transformer_infer.py` | forward 用 `parallel_ctx` |
| `wan/model.py` | `_seq_parallel_pre/post` 用 ctx 的 group |
| `common/ops/attn/ulysses_attn.py` | 已支持动态 `seq_p_group` |

---

## 5. 算法设计

### 5.1 目标

最大化 `|{i : C_i ≤ deadline_i}|`；超时 **reject**。

### 5.2 拥塞与 finish 预测

```text
enc_pressure = pending(request_ring) / capacity
xf_pressure  = pending(phase1_ring) / capacity
dec_pressure = pending(phase2_ring) / capacity

finish_admission(i)    = now + enc_delay(enc_pressure) + min_k T_i(k) + dec_delay
finish_at_phase1(i, p) = now_ready + T_i(p) + dec_delay + batch_wait_est
```

`T_i(p)`：Phase 1 标定表 + EMA（**TetriServe**：不同请求对不同 `p` 扩展性不同）。

### 5.3 Admission

```text
if finish_admission(i) > deadline_i → reject
else → pending_heap（按 laxity 排序）→ produce(request ring)
```

不分配 plan、不占槽。

### 5.4 运行时模型：双流租户

- 无「主请求 / 填空请求」角色区分；每个活跃 **租户** = `(room, member_ranks, p)`。
- 租户结束 → 释放对应叶槽 → 进入 `idle_slots`，**不立即 place**，等攒批。
- 某租户做 SP **comm** 时，`member_ranks` 内 **每个 rank 参与** 该租户 collective。
- overlap = 同 GPU **`comm_stream(租户 X) ∥ compute_stream(租户 Y)`**；不是部分 rank 退出 collective。
- 每个 denoise step 约 **40 层** self-attn，每层多次 all_to_all；overlap 收益上限 ≈ Phase 3 comm 占比（P≥2 约 **30–35%** CUDA 时间）。

**rank 重叠约束**（placement 建边时过滤）：

| 关系 | 条件 |
|------|------|
| **同槽** | 请求占用的 `member_ranks` 须与槽完全一致 |
| **不交叠盘** | 不同 partition 的槽天然不交叠，可并行 |
| **禁止交错** | 部分重叠且非子集（如 `{1..4}` 与 `{3..6}`）不可同时作为活跃租户 |

### 5.5 攒批触发

按 **partition** 独立攒批：

```text
on_tenant_done(slot):     idle_slots += slot; 启动 batch_timer（若未启动）
on_encode_done(room):    ready_pool += room

should_run_batch(partition):
  idle_count >= effective_batch_min_slots
  OR now - batch_timer_start >= batch_wait_ms
  OR any idle_slot held > max_hold_ms          # 可选
  OR any ready room laxity < urgent_bypass_ms  # 可选：紧急单请求小规模 flow

run_batch(partition):
  求解 §5.6–5.7 → 应用 matching + 拓扑变更 → produce(phase1) 给新匹配请求
  清空已匹配 idle / ready；若仍有 idle 则保留 batch_timer
```

槽位空出后短时空闲（仅 comm、无 compute 租户）是换取全局最优分配的代价。

### 5.6 Batch placement：min-cost max-flow

对每个候选拓扑 `π'`（见 §5.7），建二部图：

- **左侧**：`ready_pool` 中该 partition 可见、且尚未占槽的请求（容量 1）。
- **右侧**：`π'` 下当前 **空闲叶槽**（容量 1）。
- **边** `i → s`：当且仅当
  - `member_ranks(i 在 p_s 下的 plan) == slot(s).member_ranks`
  - `finish_at_phase1(i, p_s) ≤ deadline_i`
- **费用**：`cost(i,s) = -(w1 · urgency(i) + w2 · fit_p(i, p_s))`  
  - `urgency(i) = 1 / max(laxity(i), ε)`  
  - `fit_p(i, k)` 如 `(T_i(1) - T_i(k)) / T_i(1)` 或 `1 / T_i(k)`

**求解**：min-cost max-flow（等价于最大基数匹配 + 最大权）。

**目标（字典序）**：

1. 最大按时可接入请求数（流量）；
2. 最小总费用（紧急度 + 适配度）；
3. 加上拓扑变更罚项 `split_penalty` / `merge_penalty`（§5.7）。

未匹配请求留在 `ready_pool`；未匹配空槽继续攒下一批。

### 5.7 拓扑变更：每批至多 1 次

分裂/合并 **不在** 单槽 vacancy 时做，仅在 `run_batch` 时作为 `π_current` 的 **1-hop 邻居** 候选：

| 候选 | 说明 | 罚项 |
|------|------|------|
| **C0** | 保持 `π_current`，直接 flow | 0 |
| **C1** | 对某个 **空闲** 叶槽做一次 **split**（按 `sp_groups` 预定义二分，如 `{1,2}→{1},{2}`） | `split_penalty` |
| **C2** | 对某对 **兄弟叶槽均空闲** 做一次 **merge** | `merge_penalty` |

```text
run_batch(partition):
  candidates = [C0]
  if cooldown_ok and splittable_idle_leaf exists:  candidates += split_variants
  if cooldown_ok and mergeable_idle_sibling_pair exists: candidates += merge_variants

  best = argmax_lex (max_flow, -total_cost) over candidates
  apply best.topology + best.matching
  if topology changed: last_topology_change_ts = now
```

**硬约束**：

- 每批 **最多采纳 C1 或 C2 之一**（或都不采纳）；
- 仅 **全空叶** 可 split/merge（不触碰仍有租户的槽）；
- 两次拓扑变更间隔 ≥ `topology_cooldown_ms`。

**碎片化**：多卡→少卡易、少卡→多卡难；靠 **攒批 + 匹配同 k 槽 + 受控 split/merge** 抑制滑向全程 p=1，而非 per-vacancy 见空就填更小 k。

### 5.8 Dispatch

```text
_scheduler_tick:
  admission → produce(request ring)
  per partition: if should_run_batch → run_batch
  reschedule: 仅 PENDING / ENCODE_DONE（未 phase1 前）
```

---

## 6. 分阶段实施

| 阶段 | 内容 | 验收 |
|------|------|------|
| **A** | `ParallelContext` + 静态 rank + phase1 手填 plan | N=1, P=4, denoise ~25s |
| **B** | admission + `ready_pool` + 单槽即时 place（简化）+ `T_i(p)` | SLO 命中率 > 无 admission |
| **C** | 多租户双流 + 攒批 `run_batch` + flow + 可选 split/merge | profiler：comm∥compute；`avg_tenant_p` 稳定 |

Phase B 可用贪心 place 过渡；Phase C 切攒批 flow。

---

## 7. 测试与指标

- Phase B：混合 deadline，对比 RoundRobin。
- Phase C：2+2 拓扑下多租户轮转；merge 场景（双槽同空 + p=4 请求）；split 场景（二卡槽久空 + 双 urgent p=1）；不交叠盘两 partition 并行。
- Metrics：`slo_on_time_rate`、`reject_count`、`wasted_encode`、`batch_placement_latency`、`avg_tenant_p`、`topology_changes_per_hour`、`rank_idle_during_comm`。

`DISAGG_DISABLE_AUTOSCALE=1`。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| cost 表不准 | Phase 1 标定 + EMA |
| encode 后长期无槽 | admission 粗筛；`batch_wait_ms` / urgent_bypass |
| 多租户 OOM | 限制同卡并发 compute 租户数 |
| NCCL 死锁 | 异租户 comm 不同 slot；组内 collective 同序 |
| 攒批延迟伤 SLO | `urgent_bypass`；`max_hold_ms` 强制触发 |
| rank 碎片化 | 攒批 flow + 拓扑罚项 + cooldown；避免 per-vacancy 贪心 |
| 交错 rank 计划 | placement 建边时过滤 |

---

## 9. 参考

- `save_results/optimization_study/phase4_decision.md`
- `phase1_transformer_summary.md`、`phase3_sp_scaling.md`
- TetriServe（per-request `p` 适配）
- `lightx2v/common/ops/attn/ulysses_attn.py`

---

## 10. PR 顺序

```
PR-1  context + group_registry + model 注入
PR-2  静态 rank 启动 + transformer 读 phase1 plan
PR-3  admission + ready_pool + encode_done 路径
PR-4  cost_model(T_i(p)) + resource_ledger + PartitionState + metrics
PR-5  多租户双流 runtime（Phase C 基础）
PR-6  batch_placer：攒批触发 + min_cost_max_flow + split/merge
```

---

## 11. 后续（Phase A–C 不实现）

**Autoscale**：扩缩 = 整组 GPU + 新 NCCL world（cell）；静态阶段无 cell。

**Elastic SP**：step 边界升/降 `p`、reshard；依赖 Phase C 多租户成熟。

**全局 epoch 调度**：与攒批 placement 正交，可选后续。
