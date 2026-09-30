# 原并行实验复现：L20X / NVLink

原文：[`draft_background_motivation.md`](draft_background_motivation.md)、[`motivation.md`](motivation.md)。本次保留原模型，按用户要求改用当前 GPU 可运行的算子。

## 结论

- 普通 SP、TP、PP、PP×SP hybrid 均能运行；PP/hybrid 输出与相同分区下正确层序的逐请求执行完全一致。
- 原 quad overlap 在 4/8 卡均出现 NaN；补齐 CUDA stream 依赖后，四路输出均与串行参考完全一致，但 lps=2 吞吐未超过关闭 SP overlap 的对照。
- 普通 TP2 all-reduce 双请求 overlap（修复 stream 依赖、标准 NCCL norm）通过检查，与串行耗时基本相同；它与下面的六阶段/P2P norm 方案是不同路径。
- 原 SP block-offload 双请求分解路径（有/无 overlap）在 P=2/4/8 均出现 NaN，未获得有效吞吐。
- TP 六阶段路径所需的 P2P norm 存在重复运行差异；phase serial / overlap 均出现 NaN，未获得有效吞吐。普通 TP 不受这条 P2P norm 路径影响。
- 原文“TP=2→8 基本不扩展”和“小模型 overlap 补回到理想吞吐”的性能结论没有在本机复现。

性能归因见 [SP 无 offload profiler 分析](sp_nooffload_profile_analysis.md)：8 卡损失中 NCCL 约 37%，额外拷贝/拼接约 29%；主计算接近线性扩展。

优化实测见 [SP 数据重排优化](sp_layout_optimization.md)：跳过空文本处理并简化输出通信布局，2/4/8 卡六轮交替对照验证。

Head 切分验证见 [SP 注意力计算效率](sp_head_compute_efficiency.md)：40→5 heads 保留约 97.9% 的独立计算效率。

自回归补测见 [SF head 切分效率](sf_head_compute_efficiency.md)：Lq=4680 时 SP8 纯注意力效率约75.9%。

SF并行端到端对照见 [Ulysses / stripe / stripe_hier](sf_parallel_comparison.md)，包含计算通信拆分及数值校验状态。

## 环境与测量范围

- 当前：8×NVIDIA L20X，SM90，约 140 GiB/卡，NVLink NV18；原文标注 8×A10 24GB。绝对延迟及 A10 OOM 边界不可直接比较。
- Wan2.2 I2V A14B：原版 high/low noise 4-step INT8 权重，未换成 high-noise 1030 版。另测 Wan2.1 T2V 1.3B BF16。全部下载文件经 ModelScope 元数据 SHA256 校验。
- 480×832、81 帧、4 steps；latent 严格断言 `[16,21,60,104]`。I2V 输入通过 VAE 重新生成，强制目标高宽。小模型复用同一 T5 文本条件，去除图像条件。
- 替换：`int8-q8f → int8-triton`、`sage_attn2 → torch_sdpa`、RoPE 使用 Torch。SP 大模型包含 block-offload 和无 offload 两组；TP/PP/hybrid/quad 无 offload。
- 完整 Transformer wall：包含 scheduler.prepare、去噪和最终 GPU 同步；排除模型加载、T5、VAE。模式预热后交替顺序测 3 轮，取所有 rank 最大 wall。lps 补充扫描仅 1 个正式样本。
- overlap 适配器在通信窗口结束后等待通信与另一请求 compute stream，属于保守同步修复；这些性能结果不代表更细粒度调度可达到的吞吐上限。
- 校验为与**相同并行分区**逐请求参考的最终 latent 对比（finite 且 relative L2 < 0.02）；不是跨并行度、原 Q8/Sage 后端或视频质量等价验证。未通过的模式不计吞吐。
- 本轮覆盖主分辨率的并行方法与 lps 可运行性。未重跑正方形/混合分辨率曲线、Phase0 E2E、NCCL/SM profiler 或真实 DP 集群吞吐；动态调度和 SLO/Pareto 是文档设计/待评估部分。
- 历史 `save_results/optimization_study/` 不在工作区，无法逐文件核查历史数字。

## MoE 主表：3 次均值

扩展效率 η = (请求数 / wall) ÷ (总卡数 × 单卡吞吐)。MoE 无 offload 使用 sp1_nooffload/single；block-offload 使用 sp1_block/single；小模型使用 small_sp1/single。混合并行总卡数为 PP×SP；100% 表示达到单卡吞吐的线性扩展，非实测 DP 吞吐。

| 方法 | 请求数 | wall (s) | req/s | 扩展效率 |
|---|---:|---:|---:|---:|
| SP1 block-offload | 1 | 14.935 | 0.0670 | 100.0% |
| SP2 block-offload | 1 | 8.184 | 0.1222 | 91.2% |
| SP4 block-offload | 1 | 4.371 | 0.2288 | 85.4% |
| SP8 block-offload | 1 | 2.354 | 0.4248 | 79.3% |
| SP1 无 offload | 1 | 14.760 | 0.0678 | 100.0% |
| SP2 无 offload | 1 | 8.056 | 0.1241 | 91.6% |
| SP4 无 offload | 1 | 4.170 | 0.2398 | 88.5% |
| SP8 无 offload | 1 | 2.135 | 0.4685 | 86.4% |
| TP2 | 1 | 8.951 | 0.1117 | 82.4% |
| TP4 | 1 | 5.534 | 0.1807 | 66.7% |
| TP8 | 1 | 3.703 | 0.2701 | 49.8% |
| TP2 双请求分解串行 | 2 | 17.917 | 0.1116 | 82.4% |
| TP2 双请求 all-reduce overlap（修复同步） | 2 | 17.952 | 0.1114 | 82.2% |
| PP2 GPipe lps=2 | 2 | 15.427 | 0.1296 | 95.7% |
| PP2×SP2 hybrid lps=2 | 2 | 8.344 | 0.2397 | 88.4% |
| PP2×SP4 hybrid lps=2 | 2 | 4.333 | 0.4616 | 85.2% |
| PP2×SP2 quad，关闭 SP overlap | 4 | 16.713 | 0.2393 | 88.3% |
| PP2×SP2 quad，修复 stream 依赖 | 4 | 16.848 | 0.2374 | 87.6% |
| PP2×SP4 quad，关闭 SP overlap | 4 | 8.671 | 0.4613 | 85.1% |
| PP2×SP4 quad，修复 stream 依赖 | 4 | 8.728 | 0.4583 | 84.6% |

比较 TP/PP/hybrid 时使用 SP 无 offload 行；block-offload 行保留为历史对照。请求数不同，吞吐比较使用 req/s。

## SP 无 offload 补测：3 次均值

Wan2.2 I2V A14B MoE INT8，480×832、81 帧、4 steps；计时范围同上。双请求 overlap 使用保守同步适配器。显存为单请求所有 rank 的最大峰值 allocated，非整机总显存。

| SP / 卡数 | 单请求 (s) | 单请求 req/s | b2b 两请求 (s) | 分解串行两请求 (s) | overlap 两请求 (s) | overlap req/s | 单请求峰值显存 (GiB/卡) | 扩展效率：单请求 / b2b / 分解串行 / overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.7597 | 0.0678 | 29.5190 | — | — | — | 30.68 | 100.0% / 100.0% / — / — |
| 2 | 8.0561 | 0.1241 | 16.1145 | 16.1085 | 16.0813 | 0.1244 | 30.21 | 91.6% / 91.6% / 91.6% / 91.8% |
| 4 | 4.1697 | 0.2398 | 8.3510 | 8.3818 | 8.3616 | 0.2392 | 28.88 | 88.5% / 88.4% / 88.0% / 88.3% |
| 8 | 2.1345 | 0.4685 | 4.2640 | 4.2706 | 4.3069 | 0.4644 | 28.22 | 86.4% / 86.5% / 86.4% / 85.7% |

原始文件：`sp{1,2,4,8}_nooffload.json`；cpu_offload=false。SP1 未运行跨卡通信 overlap。

## Wan2.1 1.3B：3 次均值

| SP | 单请求 (s) | 双请求串行 (s) | 修复后 overlap (s) | overlap / b2b 吞吐比 | 扩展效率：单请求 / b2b / overlap |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.1635 | 6.3280 | — | — | 100.0% / 100.0% / — |
| 2 | 1.7310 | 3.4601 | 3.5994 | 0.961× | 91.4% / 91.4% / 87.9% |
| 3 | 1.1790 | 2.3601 | 2.3741 | 0.994× | 89.4% / 89.4% / 88.8% |
| 4 | 0.8989 | 1.7976 | 1.8144 | 0.991× | 88.0% / 88.0% / 87.2% |
| 6 | 0.6081 | 1.2188 | 1.2517 | 0.974× | 86.7% / 86.5% / 84.2% |

理想吞吐为公式 P/T₁，未实测 P 路独立请求。小模型 12 heads，Ulysses P=8 不合法。

## lps 补充扫描

lps=2 来自主表 3 个样本，其余点各预热后测 1 次，仅用于可运行性与粗粒度趋势；不能据此认定稳定最优。各耗时后的括号为扩展效率。

| lps | PP2 两请求 (s) | PP2×SP2 hybrid / quad (s) | PP2×SP4 hybrid / quad (s) |
|---:|---:|---:|---:|
| 1 | 15.307 (96.4%) | 8.232 (89.7%) / 16.576 (89.0%) | 4.291 (86.0%) / 8.695 (84.9%) |
| 2 | 15.427 (95.7%) | 8.344 (88.4%) / 16.848 (87.6%) | 4.333 (85.2%) / 8.728 (84.6%) |
| 5 | 16.340 (90.3%) | 8.854 (83.3%) / 17.770 (83.1%) | 4.589 (80.4%) / 9.295 (79.4%) |
| 10 | 18.056 (81.7%) | 9.780 (75.5%) / 19.618 (75.2%) | 5.057 (73.0%) / 10.163 (72.6%) |
| 20 | 21.546 (68.5%) | 11.679 (63.2%) / 23.326 (63.3%) | 6.052 (61.0%) / 12.120 (60.9%) |

本轮 lps=1 的单样本略快于 lps=2 的三次均值，差异约 1%；仅能认为两者接近，原文“lps=2 最优”没有严格复现。

## 修复与失败证据

1. **PP 元数据 dtype**：旧传输把未识别 dtype 默认为 FP16；Torch RoPE complex128 收发字节不一致导致阻塞。已扩展 dtype 协议并拒绝未知类型；独立双卡传输及完整 PP/PP×SP 实测通过。
2. **交错 PP 单请求基线**：旧 `_infer_cond_uncond_pp` 先执行 rank0 全部本地层、再执行 rank1，不能用于 lps<20 的交错权重。新基准使用 GPipe m=1 执行正确全局层序；没有把原错误层序的数字当基线。
3. **quad stream 依赖**：新适配器连接生产 stream、通信 stream 和另一请求 compute stream，并恢复被回调修改的 scheduler/block_idx/cos_sin 状态。修复前后 JSON 分开保存（`*_safe.json` 为修复后）。
4. **SP offload**：旧 `_preload_blocks` 将不同层号缓存为同一个可变 offload GPU buffer。真实测试中 dual_serial 和 dual_overlap 均 NaN；此路径尚未修复，不能宣称复现其吞吐收益。
5. **TP P2P norm / 六阶段**：普通单请求重复 seed 的 relative L2 为 0.1859；b2b 两路为 0.2436 / 0.2110；phase_serial、phase_overlap 为 NaN。见 `tp2_phase.json`，不报告速度。
6. **输入尺寸**：旧编码器按原图纵横比计算分辨率，配置 480×832 实际可能为 720×544。首次诊断数据已隔离到 `diagnostic_720x544/`，不纳入主表。
7. **计时外汇总**：混合 PP/SP 时新基准全局 NCCL 校验归约曾阻塞，改用 CPU/Gloo 归约及 barrier；模型 GPU 通信仍为原 NCCL 实现。

## 入口与产物

- [`reproduce_parallel_methods.py`](../../scripts/disagg/reproduce_parallel_methods.py)：明确配置、shape 断言、逐请求参考校验、预热与重复计时；校验失败返回非零状态（早期运行主要依据 JSON passed 字段）。
- [`reproduction_overlap_safety.py`](../../scripts/disagg/reproduction_overlap_safety.py)：可选 `--safe-overlap` 适配器。
- [`summary.md`](../../save_results/original_parallel_reproduction/summary.md)：全部模式、样本数与校验状态。
- `save_results/original_parallel_reproduction/`：配置、模型清单/哈希、逐项日志、样本 JSON、硬件与包版本、输入缓存；`run_matrix.py` / `run_extra.py` 是本机批量入口。
- 老路径 `/root/zht/LightX2V`、旧 conda 路径、硬编码 A10 单卡基线没有沿用；原草稿的历史数字保留。
