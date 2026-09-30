# SP 按 head 切分的计算效率验证

自回归配置补测见 [SF head 切分效率](sf_head_compute_efficiency.md)：Lq=4680 时，SP8 纯注意力效率约75.9%，与本页长序列结果不同。

## 结论

当前 Wan2.2 A14B、480p×81 帧配置下，SP 按 head 切分使注意力计算效率略降，但没有明显利用率塌陷。40→5 heads 时独立测试保留约 97.9% 的计算效率，不能解释完整 SP8 约 86.4% 的扩展效率。

## 控制变量与口径

- GPU0：NVIDIA L20X，132 SM；BF16，head_dim=128，batch=1，非因果 Torch SDPA FlashAttention，与当前模型后端一致。
- 固定全局序列长度 32760 = 21×60×104/4。Ulysses 按 heads 切分后，每个 head 仍处理完整序列。
- 仅改变 heads：40/20/10/5。输入来自同一 Q/K/V 的 head 子集，各子集输出与完整 40 heads 对应部分逐元素相等。
- 数据生成、连续化、输入重排都在计时外；不运行 NCCL。各形状预热 10 次，6 轮正反顺序交替，每轮 20 次，CUDA events 计时。
- 相对计算效率 = T40×H/(40×TH)，衡量单位工作量的执行效率，不是硬件计数器 measured occupancy。TFLOP/s 按 QKᵀ 和 AV 的 4×L²×H×D 运算量估算，不含 softmax。

## 独立计算结果

| 等效 SP | 每卡 heads | 单次注意力 (ms) | 相对计算效率 | QK+AV TFLOP/s |
|---:|---:|---:|---:|---:|
| 1 | 40 | 62.385 | 100.00% | 352.3 |
| 2 | 20 | 31.247 | 99.82% | 351.7 |
| 4 | 10 | 15.921 | 97.96% | 345.1 |
| 8 | 5 | 7.966 | 97.89% | 344.9 |

## 真实模型轨迹交叉核对

| SP | 注意力累计 (s) | 相对计算效率 | kernel grid | 总线程块 |
|---:|---:|---:|---|---:|
| 1 | 10.0325 | 100.00% | [256, 1, 40] | 10240 |
| 2 | 5.0373 | 99.58% | [256, 1, 20] | 5120 |
| 4 | 2.5592 | 98.00% | [256, 1, 10] | 2560 |
| 8 | 1.2674 | 98.95% | [256, 1, 5] | 1280 |

## 为什么 5 个 heads 仍能有效利用 GPU

- FlashAttention 不仅沿 heads 并行，还沿 query 序列块并行。本配置每个 head 有 256 个 query blocks；5 heads 仍有 1280 个线程块，约 9.7 个 block/SM 的总工作量。
- 这不是同时驻留 9.7 个 block/SM；它表示可调度的总工作量。所有并行度使用相同 block 大小和 kernel 特化，head 减少不直接减少每个 block 的线程数。
- 尾部调度、固定开销和资源效率仍可能造成几个百分点的下降，但本次未测量硬件 occupancy/SM active/tensor core 指标，不能进一步细分原因。PyTorch trace 中 occupancy=0 的估计字段未用作利用率证据。
- 结论仅适用于此长序列、后端和硬件。短序列、更多卡数、更少 heads 或不同内核可能出现明显计算效率下降。

## 产物

- [基准脚本](../../scripts/disagg/bench_sp_head_efficiency.py)。
- save_results/original_parallel_reproduction/sp_head_efficiency/results.json：逐轮 CUDA 计时、校验和计算效率；results.trace.json：独立实验轨迹；model_kernel_grids.json：真实模型网格。
