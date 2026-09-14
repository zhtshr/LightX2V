# SLO / Goodput 实验说明

## 目标

测 DisagFusion 与 LightX2V baseline 在不同负载下的 SLO 达成率 (SLO attainment)，
得出两个系统在 90% attainment 目标下能支撑的最大到达率（主指标；可附带
`goodput = λ × attainment`）。产出论文图：attainment vs 到达率（可加 P95 延迟 vs 到达率）。

## 实验配置（已拍板）

- **Workload**: Wan2.2 I2V **4-step**，832×480，81 帧（与现有 mixed/topology 实验同一套模型参数）
- **硬件**: **同机** 8 × A10；两系统都只用 **8 卡 / 8 实例**
- **系统 A (DisagFusion)**:
  - Config: `configs/disagg/single_node/wan22_i2v_distill_controller.json`
  - `DISAGG_TOPOLOGY=single_node`，**固定 1:6:1**（8 卡全开），`DISAGG_DISABLE_AUTOSCALE=1`，`ENABLE_MONITOR=0`
  - 原因：纯 4-step SLO 下 autoscale 在低 λ 会在 1:6:1/1:5:2 间抖动；且控制器须在等待 ingress 时持续 drain 结果，否则 `finish_ts` 会被绑到下一次到达
- **系统 B (LightX2V baseline, 单体 DP)**:
  - Config: `configs/disagg/baseline/wan22_moe_i2v_baseline_dp8.json`（`seq_p_size=1`）
  - **8 个独立单卡 worker**（`--num_workers 8 --drop_parallel_config`），每卡各自跑完整请求；**不做 SP8**
  - 与 A 同一套模型侧参数：`cpu_offload=true`，`offload_granularity=block`，int8 4-step distill，`sage_attn2`
- **SLO**: 统一绝对阈值。点 1 轻载后用 DisagFusion P50×~1.5 取整校准；两边同一阈值

### 共享模型参数（A/B 一致）

与 `configs/disagg/baseline/wan22_moe_i2v_baseline.json` / 现有 disagg distill controller 相同：
int8-q8f 4-step ckpt、480×832、81 帧、`boundary_step_index=2`、step list
`[1000,750,500,250]`、`enable_cfg=false`、`sample_shift=5.0`。

## 怎么跑

```bash
# 单点：DisagFusion @ 0.01 req/s
SCHEME=disagfusion RATE=0.01 N=65 bash scripts/disagg/run_slo_bench.sh

# 单点：baseline SP8 @ 0.01
SCHEME=lightx2v RATE=0.01 N=65 bash scripts/disagg/run_slo_bench.sh
```

泊松开环：`interval ~ Exp(λ)`。DisagFusion 经 `run_slo_poisson_user`；
baseline 经 `run_controller --arrival_rate`。交付物为 `slo_<scheme>_<rate>.csv`。

## 负载扫描

参考容量（Fig 13b）：LightX2V ~0.022 req/s；DisagFusion ~0.077 req/s。本机可按点 1 微调。

| 点 | 到达率 (req/s) | 说明 |
|----|---------------|------|
| 1  | 0.010         | 轻载 + SLO 校准 |
| 2  | 0.018         | baseline 近饱和 |
| 3  | 0.028         | baseline 过载 |
| 4  | 0.045         | |
| 5  | 0.065         | DisagFusion 近饱和 |
| 6  | 0.085         | DisagFusion 过载 |

每点 × 每系统 = 12 次独立运行；每次前 `kill_service` 重置。
baseline 点 4–6 若 attainment≈0，~30 请求确认即可提前停。

## 每次运行要求

1. 开环泊松；`arrival_ts` = 注入系统时刻
2. 默认 N=65（warmup=5）；点 1–2 可 N=45
3. 单请求客户端等待超时 600 s → `status=timeout`，`finish_ts` 空
4. 同 prompt / 图；seed 默认 0

## CSV

```csv
scheme,arrival_rate,req_id,arrival_ts,finish_ts,is_warmup,status
```

- attainment：非 warmup 中 `latency≤SLO` 比例（timeout/error 未达成）
- 主 goodput：最大 λ 使得 attainment ≥ 90%

## 运维

同机 8 卡会用到 GPU 1/3；近期若稳定可先用。出现 Xid/掉卡则记录并停该点。

## 延迟口径（重要）

- SLO latency = `finish_ts - arrival_ts`
- DisagFusion：`finish_ts` 必须是控制器**及时**收到 decoder 结果的时刻。控制器在等下一次 ingress 时也会非阻塞 drain（勿再阻塞 `receive`）。
- 若跑的是旧控制器：可用 `slo_build_csv.py --finish_from decoder_done` 按 decoder `output_enqueued_ts` 重算（去掉“等下次到达才收结果”的伪延迟）。
- LightX2V baseline：`finish_ts` 来自 worker 推理结束时刻，不受控制器收包节奏影响。
