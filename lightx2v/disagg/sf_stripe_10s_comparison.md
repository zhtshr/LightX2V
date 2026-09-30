# SF 8 卡：约10秒视频对照

## 配置与校验

- Krea SF 14B BF16、480×832、8张L20X、Ulysses与stripe Form C；无DiT/KV offload，完整历史KV，同seed42，4steps并包含每chunk的KV rerun。
- SF按3个latent帧向下对齐。请求161帧会实际输出153帧，因此本次选165帧，即42个latent帧、14chunks，按16fps约10.31秒。实际decode输出形状已断言。
- 每chunk的Lq仍为4680，最后chunk的Lk增至65520；81帧对照为7chunks、最大Lk32760。
- 原stripe仅运行一次用于保存165帧参考。Form C与该参考最终latent逐元素相等，relative L2=0；相对Ulysses的relative L2约0.93455，仍未通过0.02阈值。下表Form C仍为数值漂移方案的诊断性能。
- 每种预热后3次计时；端到端包含常驻T5、广播、完整AR/rerun及rank0完整VAE解码，不含模型加载、排队或视频文件编码。

## 延迟对照（秒）

| 输出帧数 | Ulysses去噪 | Form C去噪 | Form C去噪增幅 | Ulysses端到端 | Form C端到端 |
|---:|---:|---:|---:|---:|---:|
| 81 | 3.282 | 3.779 | 15.1% | 5.039 | 5.497 |
| 165 | 8.653 | 9.104 | 5.2% | 12.136 | 12.532 |

81帧为此前相同配置数据，非本轮交替测量。165帧两方案T5均约0.028秒，VAE约3.35秒。

## 165帧计算与通信（rank0单独profiler，秒）

| 方案 | 注意力kernel | 其他非NCCL操作 | NCCL |
|---|---:|---:|---:|
| ulysses | 4.625 | 2.439 | 0.304 |
| stripe | 3.532 | 3.396 | 0.902 |

profiler在正常计时外单独运行，GPU事件按区间并集统计；非NCCL包含计算、合并、复制转换，NCCL包含等待。不能用这些列替换正常运行wall。

## 判断

- 视频变长后，Form C相对Ulysses的去噪差距从约15.1%缩小到约5.2%，但本轮仍没有反超；端到端差距约3.3%。
- Form C注意力kernel比Ulysses少约1.09秒，但其他非NCCL操作多约0.96秒，NCCL多约0.60秒；计算收益仍被额外开销抵消。
- 固定chunk的Lq不随视频时长增长；无窗口截断时历史KV增长，attention工作量随chunk累积，因而耗时超过简单的帧数线性增长。
- 不能仅凭两个长度外推更长视频一定会反超，且数值漂移尚未解决。

## 产物

- save_results/sf_parallel_10s/：config.json、逐项日志/JSON/trace、参考latent、summary.json、run.py、analyze.py。
- [stripe优化实现与81帧基线](sf_stripe_optimization.md)。


约30秒补测见 [489帧对照](sf_stripe_30s_comparison.md)。
