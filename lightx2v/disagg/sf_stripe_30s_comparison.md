# SF 8卡：约30秒视频对照

## 设置

- Krea SF 14B BF16，480×832，8×L20X；489帧、123个latent帧、41chunks，16fps约30.56秒。选取向上满足3个latent帧chunk对齐的长度。
- 4steps加每chunk KV rerun，完整历史KV、无DiT/KV offload，seed42。Lq固定4680，最大Lk191880。
- 去噪与端到端分别预热后3次，profiler另外采集。端到端包含权重常驻T5、条件广播、全AR/rerun及rank0 VAE完整解码，不含加载、排队和文件编码。

## 长度对照（秒）

| 帧数 | Ulysses去噪 | Form C去噪 | Form C相对去噪延迟 | Ulysses端到端 | Form C端到端 |
|---:|---:|---:|---:|---:|---:|
| 81 | 3.282 | 3.779 | 1.151× | 5.039 | 5.497 |
| 165 | 8.653 | 9.104 | 1.052× | 12.136 | 12.532 |
| 489 | 48.879 | 45.406 | 0.929× | 59.239 | 55.352 |

相对延迟小于1表示Form C更快。不同长度为分次实验，未做跨长度交替运行。

## 489帧GPU拆分（rank0单次profiler，秒）

| 方法 | attention kernel | 其他非NCCL操作 | NCCL |
|---|---:|---:|---:|
| ulysses | 37.206 | 7.094 | 0.884 |
| stripe | 28.168 | 9.911 | 3.399 |

GPU事件区间并集统计；其他非NCCL包含合并/搬运/转换/缓存维护。NCCL包含等待。profiler有采集扰动，不能用其时间直接替换正常wall。

## 数值与收益边界

- Form C与同长度原stripe的最终latent逐元素相等：True；relative L2=0.0。
- Form C相对Ulysses的最终latent relative L2=0.78813，阈值0.02，通过状态=False。速度仍按诊断性能报告，不代表已证明视频质量等价。
- 本轮Form C去噪相对Ulysses加速=1.076×，端到端加速=1.070×。
- 长视频保持每chunk Q长度不变，但KV持续增长，注意力计算权重增大；结合GPU拆分判断收益，不能外推所有长度和窗口配置。

## 产物

- save_results/sf_parallel_30s/：config、run.py、JSON/日志、参考latents、trace、summary.json。
- [约10秒对照](sf_stripe_10s_comparison.md)；[stripe优化](sf_stripe_optimization.md)。
