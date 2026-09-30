# SF 14B：NVLink 上的双请求 SP overlap 实测

日期：2026-09-29。环境：8×NVIDIA L20X（约 140 GiB/卡），GPU 间 NVLink；PyTorch 2.8.0 + CUDA 12.8。

## 实验配置与口径

- Krea SF 14B，BF16，832×480，每请求 81 帧；7 个 AR chunk，每 chunk 4 步并保留 KV rerun。
- Ulysses SP=1/2/4/8；两个请求共享同一个 P 卡组和模型权重，各自持有 scheduler/KV cache。GPU 总数为 P。
- Torch SDPA；DiT 和 KV 均不量化、不 offload；复用上一轮的模型、prompt 和 T5 编码。两请求使用独立 latent 轨迹。
- 每种模式预热 1 次，正式测量 3 次，交替测量顺序。每次使用完整视频，取所有 rank 的最大墙钟时间，再取三次平均。
- 计时包含 scheduler 准备、KV 分配/重置、完整 AR 去噪及最终同步；不含模型加载、T5、VAE 和正确性验证。
- 吞吐 `2 / T_pair`，总帧吞吐 `162 / T_pair`。双请求完成时间不能除以 2 后作为请求响应延迟。

## 实测结果

`b2b` 为连续两次标准 `model.infer`；`分层串行` 与 overlap 使用相同分层路径，但关闭通信重叠。SP=1 没有 A2A overlap，下表该行吞吐使用 b2b。

| SP/GPU 数 | 单请求 s | 双请求 b2b s | 分层串行 s | overlap s | 吞吐 req/s | 总 FPS | vs b2b | vs 分层串行 | 吞吐扩展效率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.4009 | 30.8133 | — | — | 0.0649 | 5.26 | 1.000× | 1.000× | 100.0% |
| 2 | 8.8179 | 17.6358 | 17.8684 | 17.5407 | 0.1140 | 9.24 | 1.005× | 1.019× | 87.8% |
| 4 | 4.9373 | 9.8780 | 10.2921 | 9.8124 | 0.2038 | 16.51 | 1.007× | 1.049× | 78.5% |
| 8 | 3.3965 | 6.7802 | 7.2599 | 7.6739 | 0.2606 | 21.11 | 0.884× | 0.946× | 50.2% |

吞吐扩展效率分母为理想独立副本吞吐 `P / T1`，本轮 T1=15.4009s。独立副本吞吐未另行实测。

### 三次原始 overlap 样本

| SP | 样本 s | 平均 ± 样本标准差 s | PyTorch allocated 峰值 GiB/rank |
|---:|---|---:|---:|
| 2 | 17.5019, 17.4958, 17.6244 | 17.5407 ± 0.0726 | 53.08 |
| 4 | 9.7958, 9.8146, 9.8269 | 9.8124 ± 0.0157 | 40.41 |
| 8 | 7.5845, 7.9138, 7.5233 | 7.6739 ± 0.2100 | 34.08 |

## 与已有 motivation 文档的关系

参考 [draft_background_motivation.md §2.2](draft_background_motivation.md) 的同 SP 组双请求机制，以及 [sf_ar_optimization_summary.md §3.3](sf_ar_optimization_summary.md) 的 SF 实验。那些结果不能直接作为这台机器的性能预期：本次是 Krea SF 14B、L20X/NVLink、Torch SDPA，模型、GPU/互连、后端和测量环境均有变化。

- SP=2：overlap 相对标准 b2b 吞吐变化 **+0.54%**；相对相同分层串行变化 **+1.87%**。
- SP=4：overlap 相对标准 b2b 吞吐变化 **+0.67%**；相对相同分层串行变化 **+4.89%**。
- SP=8：overlap 相对标准 b2b 吞吐变化 **-11.65%**；相对相同分层串行变化 **-5.39%**。

当前实现中，2/4 卡分层路径的重叠收益大部分被分层调度开销抵消；8 卡 overlap 比标准 b2b 及分层串行都慢。不能据此把旧文档中“小模型 overlap 恢复至理想吞吐”的结论推广到本配置。
这些墙钟测量可以确定收益大小，尚不能将回退单独归因于 NCCL、NVLink、算子竞争或 CPU 调度；本轮未做 CUDA 时间线归因。

## 正确性与 benchmark 修正

复用了 `run_sf_transformer_dual_overlap_bench.py` 和 Phase-3 `A2AOrchestrator` 的调度机制，并修正以下问题：

1. 首个 A 请求 self-attention 前重新绑定 A，避免此前 B 的 KV/scheduler 状态残留。
2. overlap callback 执行另一请求 cross-attention/FFN 后，恢复在途请求的 scheduler、KV manager、block index、RoPE 和 attention 元数据。
3. 为独立 comm/compute stream 增加生产者与消费者依赖，下一层读取跨 stream 结果前显式等待。
4. 每轮重置两路 KV；计时结束等待 GPU/rank 同步。各模式独立预热，重复三次，并把缓存设置开销纳入外层计时。
5. 正式测量前释放验证用租户/KV 引用；最终双卡数据已在释放后重测。

验证时使用每请求/每步独立种子（42/43），使随机噪声不随请求调度顺序改变；正式计时恢复原 scheduler 的全局随机数行为。

| SP | 分层串行通过 | overlap 通过 | rank 0 两请求最大绝对误差 |
|---:|---|---|---:|
| 2 | True | True | 0 |
| 4 | True | True | 0 |
| 8 | True | True | 0 |

各 rank 均通过 finite、relative L2 < 0.02、cosine > 0.999 校验。表中误差来自 rank 0；未进行 VAE 解码或像素质量测评。

## 结果与复现

- [完整汇总](../../save_results/sf_14b_480p_dual_overlap/summary.md)
- [原始 JSON 目录](../../save_results/sf_14b_480p_dual_overlap/)（`p1.json`、`p2.json`、`p4.json`、`p8.json`；`p2_initial.json` 为显存修正前的留档，不参与最终汇总）
- [运行脚本](../../scripts/self_forcing/benchmark_sf_14b_dual_scaling.sh)
- [测量与校验入口](../../scripts/self_forcing/benchmark_sf_14b_dual.py)

复用已准备的 `.venv`、`models/`、`save_results/sf_14b_480p_scaling/config.json` 和 `encoder_inputs.pt`：

```bash
bash scripts/self_forcing/benchmark_sf_14b_dual_scaling.sh
```

模型加载、T5 和 VAE 未纳入这里的吞吐。该结果衡量固定两请求的去噪阶段，不是完整 serving 系统的到达率、排队延迟或 SLO 达成率。
