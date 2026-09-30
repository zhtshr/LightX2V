# SF 14B：Ulysses、stripe、stripe_hier 对照

## 结论

**本次没有测出可采用的 stripe / stripe_hier 加速。** 在本机相同 Torch FlashAttention 后端下，诊断去噪与端到端延迟均慢于 Ulysses；同时所有 stripe 路径与 Ulysses 的最终 latent 差异超过预设阈值，不能作为正确性通过的替代方案。

本次 hybrid 按用户确认指 stripe_hier。现有分层实现支持4/6卡，40-head Ulysses不支持6卡，故三者公平对照为4卡；另测2/8卡Ulysses与stripe。

## 方法与范围

- Krea SF 14B BF16，832×480、81帧，7 chunks，每chunk4次去噪并包含KV rerun，无DiT/KV offload；相同seed42，每次重建scheduler和KV cache，显式重置RNG。
- 普通stripe为原默认gather-Q、局部attention、gather部分out/LSE并合并路径；未将stripe_formC等其他变体当作默认stripe。
- 本机没有外部flash_attn包。仅在基准中用Torch aten FlashAttention返回out/LSE适配stripe局部计算，Ulysses继续使用Torch SDPA的FlashAttention；不是原A10 Sage/外部FA后端的复现。
- 各项预热后测3次。完整AR去噪为输入条件已缓存、到全部chunk结束；端到端为T5编码→条件广播→完整AR/rerun→VAE解码到GPU视频张量，权重常驻。
- T5与VAE仅rank0执行，均无offload，VAE为完整81帧decode而非chunk流式decode。不包含排队、模型加载、视频文件编码或传输到客户端。与早先只测Transformer的表分开。
- 先校验再计时；失败方案通过显式--diagnostic-allow-drift继续采集，所有耗时均保留失败标记。

## 正常运行延迟（3次均值，秒）

| 卡数 | 方法 | 完整AR去噪 | T5 | VAE | 端到端 | 去噪相对Ulysses | latent相对L2 | 状态 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2 | ulysses | 8.753 | 0.028 | 1.643 | 10.449 | 1.00× | 0.0000 | 参考路径 |
| 2 | stripe | 9.308 | 0.028 | 1.644 | 11.014 | 1.06× | 0.7174 | 未通过，仅诊断 |
| 4 | ulysses | 4.795 | 0.028 | 1.639 | 6.520 | 1.00× | 0.0000 | 参考路径 |
| 4 | stripe | 6.894 | 0.028 | 1.642 | 8.590 | 1.44× | 0.8559 | 未通过，仅诊断 |
| 4 | stripe_hier | 5.897 | 0.028 | 1.643 | 7.617 | 1.23× | 0.7442 | 未通过，仅诊断 |
| 8 | ulysses | 3.282 | 0.028 | 1.643 | 5.039 | 1.00× | 0.0000 | 参考路径 |
| 8 | stripe | 8.063 | 0.028 | 1.642 | 9.744 | 2.46× | 0.8270 | 未通过，仅诊断 |

去噪相对值越小越好。T5/VAE、去噪独立均值不必严格相加为端到端，因为端到端还包括广播、同步、校验和运行时变化。

## 去噪GPU计算与通信（rank0单独profiler运行，秒）

| 卡数 | 方法 | 注意力kernel | 其他非NCCL GPU操作 | 非NCCL合计 | NCCL | 两者重叠 | GPU空闲间隙 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | ulysses | 3.950 | 4.091 | 8.041 | 0.305 | 0.000 | 0.708 |
| 2 | stripe | 3.913 | 4.508 | 8.421 | 0.508 | 0.000 | 0.613 |
| 4 | ulysses | 2.012 | 2.109 | 4.122 | 0.215 | 0.000 | 0.976 |
| 4 | stripe | 1.912 | 3.642 | 5.554 | 0.901 | 0.000 | 0.781 |
| 4 | stripe_hier | 2.534 | 2.654 | 5.188 | 0.725 | 0.484 | 0.914 |
| 8 | ulysses | 1.261 | 1.230 | 2.491 | 0.166 | 0.000 | 1.201 |
| 8 | stripe | 0.988 | 4.768 | 5.756 | 1.697 | 0.000 | 0.975 |

- 按GPU事件区间并集统计；非NCCL包括计算、重排复制、缓存维护、softmax合并等，不等于纯矩阵计算。注意力列包含self/cross attention kernel。
- NCCL包含传输及kernel内等待。分层stripe存在计算通信重叠，不能把两列直接相加。
- profiler只采rank0、单次完整去噪，产生CPU调度/采集扰动；空闲间隙不能直接归为生产运行损失，不能用profiler跨度替换上表正常计时。

## 数值校验说明

- 以同卡数Ulysses的最终完整latent作参考，finite且relative L2<0.02为通过。Ulysses列为参考自身，不代表与独立真值逐项验证。
- stripe/stripe_hier均为finite但未通过。独立随机注意力分片合并测试相对L2约0.00290，说明不能仅由最终差异断定LSE适配器算式错误；完整AR数值误差累积与分片实现差异尚未完成逐层定位。
- 没有降低阈值、把图像质量视作已验证或宣称等价加速；本次速度结论严格限于当前实现和适配后端的诊断观察。

## 解释

- stripe保留更多heads能改善部分attention计算效率，但默认gather部分输出及LSE、合并和额外搬运引入更大代价。4卡attention小幅加速不足以抵消通信和其他GPU工作增加。
- stripe_hier较默认stripe降低完整去噪延迟，并产生可见计算通信重叠，但仍未超过Ulysses。
- 不能由本次结果推断所有stripe变体都无收益；本次没有扫描Form C、量化通信或重新调优内核。

## 产物

- [基准脚本](../../scripts/disagg/bench_sf_parallel_compare.py)。
- save_results/sf_parallel_compare/：逐项JSON/日志、rank0 trace、Ulysses参考latents、summary.json、analyze.py、write_report.py。
- save_results/original_parallel_reproduction/run_sf_parallel_compare.py与run_sf_parallel_diagnostic.py：批量入口。

## 8 卡 stripe 额外开销定位

使用已保存的 rank0 trace，将GPU事件的External id映射回CPU侧PyTorch算子。以下是对应GPU事件累计时间，不是CPU算子提交时间。

| 算子类别 | Ulysses (s) | stripe (s) | 增量 (s) |
|---|---:|---:|---:|
| 合并减法 | 0.0001 | 1.6278 | +1.6276 |
| 逐元素乘法 | 0.1289 | 0.9151 | +0.7862 |
| sigmoid / log_sigmoid | 0.0000 | 0.0453 | +0.0453 |
| 复制与类型转换 | 0.1548 | 1.2323 | +1.0775 |
| NCCL | 0.1657 | 1.6968 | +1.5311 |
| 注意力 | 1.2607 | 0.9877 | -0.2730 |
| 线性层矩阵乘 | 0.7017 | 0.7028 | +0.0011 |

结论：合并相关减法、乘法、sigmoid/log_sigmoid合计增加约2.459秒；复制/类型转换增加约1.078秒，两者合计3.537秒，几乎解释全部非通信增量。线性层矩阵乘基本未变。

代码位置：lightx2v/common/kvcache/base.py 的 _stripe_merge_block 与 sp_kvcache_attn_stripe 默认分支。每次attention将8份部分out/LSE收集到每卡，转换到FP32后依次合并7次，最后才切出本卡query行。

本配置每份完整out形状为 [1,4680,40,128]，共23,961,600元素，FP32约91.4 MiB。每次合并对完整out执行两次减法和一次乘法，并生成中间结果；小LSE上的sigmoid/log_sigmoid本身只增加约45 ms，并非主要瓶颈。

1400次自注意力 × 7次合并 = 9800次合并。trace恰好出现9800个新增乘法事件、49000个新增减法事件（每次合并包含out和LSE更新）、各9800个sigmoid/log_sigmoid事件，与实现一致。

由于最后每卡仅保留1/8 query行，默认实现存在冗余的完整输出合并。优化方向是交换/合并仅本卡需要的query行（现有Form C方向），以及融合FP32转换、加权合并和输出转换；这不是已验证的性能收益，还需先处理最终latent数值漂移。

限制：当前stripe为Torch LSE适配后的诊断实现，最终latent校验未通过。NCCL累计时间仍包含等待，现有trace不能进一步把它拆成纯链路传输与rank等待。原始汇总为stripe_overhead.json；本次未重跑GPU基准。

优化补测见 [stripe合并优化](sf_stripe_optimization.md)：Form C对原stripe约2.1倍去噪加速，且与原stripe逐元素相等。
