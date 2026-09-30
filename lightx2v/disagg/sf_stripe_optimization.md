# SF 8 卡 stripe 合并优化

## 改动与校验

- base：原stripe，收集完整部分输出，在每卡合并全部query行后截取本卡行。
- local：保留原All-Gather通信；在FP32转换和7次合并之前，先对每份out/LSE取本卡query行，消除冗余合并。新增LIGHTX2V_STRIPE_LOCAL_MERGE=1开关，默认关闭，full_q模式不启用。
- formc：启用现有Form C路径，All-to-All交换本卡需要的部分out/LSE，再合并本卡query行；不再All-Gather完整部分输出。通过基准--stripe-optimization formc启用。
- 两项优化均与本次原stripe的完整81帧最终latent逐元素相等，relative L2=0；同输入seed42、7chunks、4steps加KV rerun，3次正式计时。
- 原stripe相对Ulysses的latent差异仍为0.82697，优化保持该结果而未解决该差异。因此可以确认优化了stripe自身，但尚不能作为数值校验通过的Ulysses替代。

## 8 卡延迟（秒，3次均值）

| 方案 | AR去噪 | 端到端 | 去噪相对原stripe加速 | 与原stripe逐元素相等 |
|---|---:|---:|---:|---|
| base | 8.043 | 9.725 | 1.00× | True |
| local | 5.319 | 7.019 | 1.51× | True |
| formc | 3.779 | 5.497 | 2.13× | True |
| Ulysses（此前同配置对照） | 3.282 | 5.039 | — | 不相等 |

端到端包含权重常驻的T5、广播、完整AR/rerun和rank0 VAE解码；不含加载、队列及文件编码。此前Ulysses不是本轮交替计时，因此小差异不能视为严格配对统计。

## 去噪profiler拆分（rank0，秒）

| 方案 | 注意力 | 其他非NCCL操作 | NCCL |
|---|---:|---:|---:|
| base | 0.988 | 4.768 | 1.690 |
| local | 0.986 | 1.994 | 1.698 |
| formc | 0.985 | 1.710 | 0.449 |

profiler在正常计时之外独立运行，非NCCL包括合并、复制、类型转换等；各项为GPU事件区间并集，不能代替端到端wall。NCCL包含等待。

## 判断

- 提前截取query行显著降低冗余合并，证明此前定位的完整输出合并确实是主要开销来源。
- Form C进一步减少输出通信与临时搬运；本轮约2.1倍去噪加速，已明显接近Ulysses，但仍未超过。
- 只引入可选开关，不将优化默认用于其他模型、full_q或未验证后端。未验证其他卡数和视频配置。

## 入口

- [KV缓存实现](../common/kvcache/base.py)：LIGHTX2V_STRIPE_LOCAL_MERGE=1。
- [基准](../../scripts/disagg/bench_sf_parallel_compare.py)：--stripe-optimization base/local/formc。
- save_results/sf_parallel_compare/run_optimization.py、analyze_optimization.py、optimization_summary.json及p8_stripe_opt_*.json/.log/.trace.json。

长视频补测见 [165帧对照](sf_stripe_10s_comparison.md)。

约30秒补测见 [489帧对照](sf_stripe_30s_comparison.md)。

## 5秒继续优化：连续缓冲区All-to-All

2026-09-29补测：8卡、81帧。将Form C逐peer的输出/LSE临时张量列表改为连续buffer，调用all_to_all_single，接收结果按rank取view。合并顺序保持不变；通过LIGHTX2V_STRIPE_SINGLE_A2A=1开启，默认关闭。

| 方案 | 去噪均值 (s) | 端到端均值 (s) |
|---|---:|---:|
| Form C 列表通信 | 3.7746 | 5.4882 |
| Form C 连续缓冲区 | 3.7397 | 5.4183 |

去噪延迟降低0.92%，端到端降低1.27%。每种预热后3次，为分次对照，尚非多次交替验证的小收益结论。

两方案完整latent均与原stripe逐元素相等。相对Ulysses的数值差异仍存在；新方案也仍慢于此前Ulysses的3.282秒去噪。

| 方案 | attention (s) | 非NCCL合计 (s) | NCCL (s) |
|---|---:|---:|---:|
| list | 0.9845 | 2.6941 | 0.4538 |
| single | 0.9846 | 2.6663 | 0.4517 |

本轮主要减少约28ms的非NCCL GPU操作，NCCL几乎不变。因此这项优化是小幅改进，并未解决剩余合并开销。下一步若追求更大收益，应评估融合FP32转换/逐份softmax合并/输出转换，但不能提前保证逐元素一致或收益。

入口：save_results/sf_parallel_compare/run_a2a_optimization.py；结果p8_stripe_a2a_list/single.json、对应trace、a2a_summary.json。
