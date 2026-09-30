# SF 自回归 head 切分的计算效率

## 结论

Krea SF 14B 的 480×832 配置下，Lq=4680；每卡 40→5 heads 时，纯自注意力计算效率下降约 **24.1%**。因此此前 Lq=32760 的非自回归结果不能用于解释 SF：SF 的短 query 更容易在高 SP 下出现计算效率损失。

## 方法

- 与之前 SF 实测配置对应：每 chunk 3 个 latent 帧、每帧 30×52 tokens，Lq=4680；7 个 chunk，local_attn_size=-1，Lk=4680×chunk_index（包含当前 chunk）。
- GPU0 L20X，132 SM，BF16，head_dim=128，batch=1；Torch SDPA FlashAttention，is_causal=False，与缓存注意力调用一致。历史因果性由可见 KV 范围提供。
- 固定 Lq/Lk，只改变 40/20/10/5 heads；随机输入的 head 子集与完整 heads 对应输出逐元素相等。所有 28 个形状通过。
- 每个形状预热10次、6轮正反顺序交替，每轮20次，CUDA events计时。输入生成/复制/连续化和通信均不在计时内。
- 相对计算效率=T40/(SP×Tshard)，是单位计算量执行效率，不是硬件 occupancy。约100%的微小上下波动不解读为显著差异。

## 各 chunk 计算效率（以40 heads为100%）

| chunk | Lq | Lk | SP2 / 20 heads | SP4 / 10 heads | SP8 / 5 heads |
|---:|---:|---:|---:|---:|---:|
| 1 | 4680 | 4680 | 99.2% | 92.6% | 73.4% |
| 2 | 4680 | 9360 | 99.7% | 93.3% | 73.6% |
| 3 | 4680 | 14040 | 101.5% | 95.1% | 75.0% |
| 4 | 4680 | 18720 | 101.2% | 95.4% | 75.6% |
| 5 | 4680 | 23400 | 100.1% | 94.7% | 76.1% |
| 6 | 4680 | 28080 | 100.9% | 95.7% | 76.4% |
| 7 | 4680 | 32760 | 101.1% | 95.9% | 77.0% |

## 七个 chunk 合计

每个 chunk 各调用一次自注意力的合计时间。若每个 chunk 的层数、去噪及 rerun 调用次数相同，整体注意力计算效率保持此比例；不是完整模型视频耗时。

| 等效 SP | heads | 七次注意力合计 (ms) | 相对计算效率 |
|---:|---:|---:|---:|
| 1 | 40 | 37.663 | 100.0% |
| 2 | 20 | 18.692 | 100.7% |
| 4 | 10 | 9.892 | 95.2% |
| 8 | 5 | 6.201 | 75.9% |

## kernel 网格证据与解释

- 实际选中的 query tile 为128，故 ceil(4680/128)=37 个 query blocks/head。全部7种KV长度均使用同样的网格形式。
- 40/20/10/5 heads 分别产生1480/740/370/185个线程块；SP8仅185块，对132个SM约1.40块/SM的总工作量。长序列实验的SP8则有1280块。
- 这支持短query在head切分后可调度工作量不足、尾部调度损失更显著的解释。这里是总网格工作量，不是同时驻留blocks/SM；未采集硬件占用率，不能把24.1%全部精确归为某一种硬件等待。
- 你记得的74块与Lq=4680、query tile=64吻合；本机实际tile=128，所以是37块。A10后端的具体tile仍需原trace确认。
- 本次直接测得SP8注意力效率约75.9%，但不能把完整SF模型SP8的56.3%扩展效率全归因于此；还包含其他算子、通信及重排。

## 复现与产物

- [微基准脚本](../../scripts/disagg/bench_sp_head_efficiency.py)：新增 --kv-sequence，默认仍等于 --sequence。
- 批量入口：save_results/original_parallel_reproduction/run_sf_head_efficiency.py。
- 原始结果：save_results/original_parallel_reproduction/sf_head_efficiency/ 中chunk1至chunk7的JSON、日志、trace及summary.json。
- [此前长序列对照](sp_head_compute_efficiency.md)。

完整模型并行验证见 [SF Ulysses/stripe/stripe_hier 对照](sf_parallel_comparison.md)。
