# SP 数据重排优化尝试

## 改动

在不改变精度和注意力算法的前提下，增加显式可选实验适配器：

1. return：输出 All-to-All 直接传 sequence-major 数据，接收后做一次转置，省去发送前 head-major 转置复制。
2. notext：Wan 图像自注意力没有文本 token，跳过通用 Ulysses 的空文本拼接、空文本 gather 和最终拼接路径。仅启用在本次 dense TorchSDPA、无通信量化等受限条件。
3. both：组合两项。无跨请求可变缓冲区缓存。

## 验证方法

- 同模型、输入和无 offload 配置：Wan2.2 I2V A14B MoE INT8，480×832、81 帧、4 steps。
- 首轮 8 卡分别测 base / return / notext / both / base，各 3 次；候选最终 latent 与原路径保存的 seed42/43 输出逐元素相等。
- 复测在同一进程交替运行 base/both，奇数轮反转顺序，各 6 次。两模式均与原路径参考 max_abs=0，所有 rank 的校验归约通过。计时为最大 rank wall，排除加载/T5/VAE。

## 同进程交替对照（6 次均值）

| SP | 原路径 (s) | 组合优化 (s) | 吞吐提升 | 更快轮数 |
|---:|---:|---:|---:|---:|
| 2 | 8.0239 | 7.9089 | 1.45% | 6/6 |
| 4 | 4.1356 | 4.0796 | 1.37% | 6/6 |
| 8 | 2.1214 | 2.0904 | 1.48% | 6/6 |

## 8 卡单独改动（首轮，各 3 次）

| 方案 | 均值 (s) |
|---|---:|
| base | 2.1153 |
| return | 2.1020 |
| notext | 2.0948 |
| both | 2.0828 |
| base_1 | 2.1155 |

## 判断与限制

- 这次找到了正确且在本轮所有配对中更快的改动，收益约 1%–2%，没有获得大幅加速。不能把之前归类的全部额外拷贝都当作可删除开销；接收后的布局转换、Q/K/V 打包等仍然存在。
- 当前目录没有此前优化失败的完整原始结果，无法判断你之前各尝试的具体失败原因。本次记录了原路径对照，避免只与候选自己的串行路径比较。
- 保留为实验选项，不默认修改生产路径；未验证其他 attention 后端、量化通信、其他模型或 overlap 的收益。没有覆盖旧的实测基线表。

## 复现入口

- [实验适配器](../../scripts/disagg/sp_layout_experiment.py)。
- [基准脚本](../../scripts/disagg/reproduce_parallel_methods.py)：--sp-layout base/return/notext/both/compare；compare 自动选择 base/both 模式，并要求逐元素相等。
- 批量入口：save_results/original_parallel_reproduction/run_sp_layout.py、run_sp_layout_compare.py；原始日志与结果保存在 sp_layout/。
