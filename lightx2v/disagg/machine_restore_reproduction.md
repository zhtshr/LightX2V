# 新机器恢复与并行实验复现指南

记录日期：2026-09-30。本文对应本仓库这次 L20X 实测和云盘备份；详细结果见下方索引。

## 1. 备份内容与最快恢复路径

云盘位置：`/mnt/tidal-alsh-share2/dataset/zht/server-backup/LightX2V/`。
已完成一次全量复制，文件总大小约 100.39 GB（十进制），包含代码、`.git`、`.venv`、模型、输入缓存、结果、日志和 profiler trace。GitHub 只保存代码、文档和依赖快照；`models/`、`save_results/`、`.venv/` 被忽略，因此仅 clone 不能恢复完整实验。

先用原有云盘挂载方式挂载该目录；挂载凭据不属于本仓库。建议恢复到原路径，避免修改绝对路径：

```bash
CLOUD=/mnt/tidal-alsh-share2/dataset/zht
mountpoint -q "$CLOUD" || exit 1
mkdir -p /root/LightX2V
rsync -a --info=progress2 "$CLOUD/server-backup/LightX2V/" /root/LightX2V/
cd /root/LightX2V
```

这会更新目标同名文件，适用于新机器的空目录。已有工作请先保留。恢复到其他路径时检查配置中的 `dit_original_ckpt`、模型目录内的符号链接，以及 `save_results/` 下 runner 的 `/root/LightX2V` 绝对路径。

再次上传使用同方向的增量复制，不加 `--delete`：

```bash
rsync -a --stats /root/LightX2V/ "$CLOUD/server-backup/LightX2V/"
rg 'juicefs_staging_(blocks|block_bytes)' "$CLOUD/.stats"
```

上传结束后，两个 staging 指标都为 0 再释放机器。云盘备份中的 Python 环境不包含系统驱动、外部 Python 安装或云盘挂载服务。

## 2. 实测环境与依赖

| 项目 | 当前实测环境 |
|---|---|
| GPU | 8 × NVIDIA L20X，单卡约 140 GiB，SM90 |
| 互联 | `nvidia-smi topo -m` 显示 NV18 |
| 驱动 | 570.148.08 |
| Python | 3.12.11 |
| PyTorch / CUDA runtime | 2.8.0 / 12.8 |
| torchvision / torchaudio | 0.23.0 / 2.8.0 |
| Triton | 3.4.0 |
| transformers / diffusers | 4.57.6 / 0.35.2 |
| huggingface_hub / accelerate | 0.36.2 / 1.15.0 |
| Attention | 当前复现实验使用 Torch SDPA |

当前完整 `pip freeze` 已保存为 [依赖快照](../../requirements/reproduction/l20x-cu128-py312.txt)。实验当时的快照也保存在各 `save_results/.../packages.txt` 中。此快照是实际安装状态，并非所有 GPU 通用的依赖方案。

**注意：复制的 `.venv` 不能保证直接运行。** 它的 Python 来自 `/opt/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin`，包含绝对符号链接。新机器缺少该解释器时，先准备 Python 3.12，再重建：

```bash
cd /root/LightX2V
# 仅在复制的环境不能运行、需要重建时执行；保留原环境用于排查。
mv .venv .venv.restored
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements/reproduction/l20x-cu128-py312.txt
.venv/bin/python -m pip check
```

不要覆盖已有的 `.venv.restored`。`save_results/sf_14b_480p_scaling/wheels/` 保留了下载过的 wheel，但未确认构成完整离线安装源。新机器需提供适配 CUDA 12.8 的驱动；不用为了这些实验安装外部 FlashAttention2。

网络排障：本机部分下载和 GitHub 连接被代理拒绝，去掉 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 及小写同名变量后直连可用；按新机器网络情况决定，不要全局盲目删除代理设置。

环境检查：

```bash
nvidia-smi
nvidia-smi topo -m
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability()); print(torch.ones(1,device="cuda"))'
```

## 3. 模型清单

| 用途 | 恢复后的相对路径 | 说明 |
|---|---|---|
| Krea SF 14B | `models/Self-Forcing/checkpoints/krea-realtime-video-14b.safetensors` | BF16，约 28.58 GB，40 heads |
| SF 基础资产 | `models/Wan-AI/Wan2.1-T2V-14B/` | config、UMT5 tokenizer、T5、VAE；这里没有另存完整的 Wan14B DiT，使用 SF checkpoint |
| Wan2.2 MoE | `models/lightx2v/Wan2.2-Distill-Models/` | high/low noise 两份 `wan2.2_i2v_A14b_*_noise_int8_lightx2v_4step.safetensors`，各约 15 GB；不是 `_1030` 版本 |
| 小模型 | `models/Wan-AI/Wan2.1-T2V-1.3B/` | BF16 DiT 约 5.68 GB，12 heads |

MoE 目录里的 T5、VAE、tokenizer 可能是指向基础资产的符号链接，必须随基础资产一起恢复。SF 来源为 ModelScope `krea/krea-realtime-video`，revision `930cb53cef68fa0fa960a454e83bb55659c33d5a`。

模型来源、大小及 SHA256 记录：

- `save_results/sf_14b_480p_scaling/model_sources.json`
- `save_results/original_parallel_reproduction/model_metadata.json`

SF checkpoint SHA256：`792f6042a645e227f5d71851845dc4098a7338fec9ec55279f88ebbc3527eec7`。需要验证复制完整性时可运行 `sha256sum` 对照；大文件校验需要时间。下载入口是 `scripts/self_forcing/download_sf_14b.sh`，恢复已有模型通常更快。

## 4. 复现入口

### 4.1 SF 480p，1/2/4/8 卡 Ulysses

```bash
cd /root/LightX2V
BENCH_OUTPUT=save_results/restore_sf_scaling \
  bash scripts/self_forcing/benchmark_sf_14b_scaling.sh
```

脚本生成 config、输入缓存、环境记录和 summary，默认 warmup 1 次、计时 3 次；默认依次用前 1/2/4/8 张卡。新输出目录防止覆盖原始结果。该入口只统计完整自回归 Transformer denoise（含 KV rerun），不含 T5、VAE、加载。

配置：BF16、无 offload、832×480、81 帧。按 16 fps 约 5 秒，实际 21 latent frames、每 chunk 3 latent frames，共 7 chunks；每 chunk `Lq=3×30×52=4680`，历史增加使 Lk 增长。非自回归整段的 `32760=21×30×52` 不适用于 SF 单 chunk。

### 4.2 Stripe / Ulysses 与端到端比较

入口：`scripts/disagg/bench_sf_parallel_compare.py`。先恢复原始 `save_results/sf_14b_480p_scaling/config.json` 和 `encoder_inputs.pt`；此脚本的输入缓存路径目前固定，上一节新输出目录不会自动替换它。

```bash
export PYTHONPATH=/root/LightX2V
export DTYPE=BF16 OMP_NUM_THREADS=8 PROFILING_DEBUG_LEVEL=0 LOGURU_LEVEL=WARNING
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/disagg/bench_sf_parallel_compare.py --seq_p_size 8 \
  --seq_p_attn_type ulysses --output save_results/restore_compare/p8_ulysses.json
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/disagg/bench_sf_parallel_compare.py --seq_p_size 8 \
  --seq_p_attn_type stripe --stripe-optimization formc --diagnostic-allow-drift \
  --output save_results/restore_compare/p8_stripe_formc.json
```

必须先在同一个输出目录跑 Ulysses，生成参考 latent。`--diagnostic-allow-drift` 只用于复现性能诊断：当前 stripe 与 Ulysses 的 latent 相对 L2 未通过阈值，因此不能据此认定生成质量等价。优化版本与原始 stripe 则已测得逐元素完全一致；若要重新验证此项，先在同目录跑 `--stripe-optimization base` 生成 stripe 基线。

`LIGHTX2V_STRIPE_SINGLE_A2A=1` 是最后增加的连续 buffer 通信优化，默认关闭；对照时显式设为 0/1。`stripe_hier` 是分层 stripe，不是 PP×SP，本轮三方法共同对照主要为 4 卡。

此入口另测真实 T5 + AR denoise + 完整 VAE decode 的端到端时间，排除模型加载、排队、视频编码和传输；不是流式逐 chunk VAE。Profiler 单独运行，通信 kernel 包括等待，不能把 profiler 分项直接相加视为正常 E2E。

10 秒/30 秒的配置和 batch runner 在云盘恢复的 `save_results/sf_parallel_10s/`、`sf_parallel_30s/` 中。使用 165/489 实际帧（约 10.3/30.6 秒）；161 请求会因 chunk 对齐得到 153 实际帧，不能只看配置中的期望帧数。

### 4.3 原模型的 PP、TP、SP、overlap

通用入口：`scripts/disagg/reproduce_parallel_methods.py`。恢复目录 `save_results/original_parallel_reproduction/` 内保存了 `run_matrix.py`、`run_extra.py`、各配置、`moe_inputs.pt`、`small_inputs.pt`、原始日志和汇总。先检查 runner 输出目录后再运行，防止覆盖原实验。

- 配置包括 `moe.json`、`moe_block.json`、`pp2.json`、`pp2sp2.json`、`pp2sp4.json`、`small.json`。
- MoE 使用当前硬件支持的 `int8-triton`，不是历史 A10 的 `int8-q8f`。
- head 效率入口：`scripts/disagg/bench_sp_head_efficiency.py`。
- SF 双请求入口：`scripts/self_forcing/benchmark_sf_14b_dual_scaling.sh`；先检查其输出设置。
- Git 中保留通用脚本；完整本地 runner、trace、配置和缓存由云盘恢复。

## 5. 主要实测结果

以下为当前机器、当前后端的结果，不代表 A10 原环境逐数值复现。完整采样、验证和口径以各专题文档为准。

| 模型与口径 | 1 卡 | 2 卡 | 4 卡 | 8 卡 |
|---|---:|---:|---:|---:|
| SF 14B 81 帧，Ulysses denoise 秒 | 15.389 | 8.768 | 4.964 | 3.415 |
| MoE INT8 无 offload，SP 单请求秒 | 14.7597 | 8.0561 | 4.1697 | 2.1345 |
| 上行 MoE 扩展效率 | 100% | 91.6% | 88.5% | 86.4% |

扩展效率为 `T1/(N×TN)`。SF 后续独立比较批次 8 卡 Ulysses 为 3.282 秒，和初始 scaling 不是同一批采样。MoE SP 双请求 overlap 相比连续两个请求没有稳定收益；PP 结果需同时考虑 offload、驻留方式及流水调度，不能全部归因于 overlap。

| SF 8 卡长度 | Ulysses denoise / E2E 秒 | Stripe Form C denoise / E2E 秒 |
|---|---:|---:|
| 81 帧 | 3.282 / 5.039 | 3.779 / 5.497 |
| 165 帧 | 8.653 / 12.136 | 9.104 / 12.532 |
| 489 帧 | 48.879 / 59.239 | 45.406 / 55.352 |

原始 81 帧 stripe 约 8.043 / 9.725 秒；局部行合并降至 5.319 / 7.019，Form C 再降至表中水平。最后连续 buffer A2A 同批比较从 3.775 / 5.488 到 3.740 / 5.418 秒，收益较小，仅 3 次采样；30 秒表没有包含这一最后优化。所有这些性能结果都保留上述 stripe 对 Ulysses 的数值差异限制。

纯 attention 计算效率：长序列 MoE 的 8 卡 head 切分约 97.9%，SF 短 query 聚合约 75.9%；SF 短 query 下减少 heads 确实降低计算效率，不能将所有损失都归为 NVLink 通信。这里是时间测量得到的效率，不是硬件 occupancy 计数器。

## 6. 踩坑与已处理事项

1. **量化后端与架构不匹配**：原 q8f 面向 Ada，本机 SM90 改用 int8-triton；Sage/外部 FA 改为 Torch SDPA，RoPE 使用 Torch。不可宣称完全复现原后端。
2. **依赖的间接 import**：除 torch/diffusers 外，曾补装 gguf、pyzmq、easydict、omegaconf、decord、langdetect、pydantic 等。使用快照比逐个猜依赖可靠；旧 `finish_setup.py` 还会启动 benchmark，不要当纯安装脚本盲跑。
3. **旧路径与 GPU 黑名单**：历史脚本使用 `/root/zht/LightX2V`，并排除 A10 上的部分卡；当前 SF runner 提供 `--allow_all_gpus`，新脚本自动推导仓库路径。
4. **输入实际尺寸**：保宽高比处理曾生成 720×544，诊断结果已隔离，不纳入 480×832 正式表；要核对 latent shape 和实际输出帧数。
5. **随机数**：SF scheduler 的 seed 参数不足以重置所有状态，比较脚本显式 `seed_all(42)`，每轮重建 scheduler 并重置缓存。
6. **PP 数据协议**：Torch RoPE 的复数张量需要正确传输 dtype；已补 float64/complex64/complex128，空 tensor 不发送 payload。未知 dtype 显式报错。
7. **多组通信与 overlap**：控制统计改用 Gloo，避免 NCCL 多组顺序死锁；双请求安全适配处理 stream 依赖和状态恢复。旧 block-offload 拆解路径共享 GPU buffer 可导致 NaN；TP phase 路径也存在不稳定结果，不能只看时间。
8. **PP 基线**：交错流水要与正确全局层序的 GPipe m1 对照。lps sweep 的部分设置仅 1 次计时，不能认定某个 lps 是稳定最优。
9. **显存差异**：本机 SF 单卡峰值约 52.47 GiB，MoE 单卡约 30.68 GiB；140 GiB 卡可关闭 offload，不能照搬到 24 GiB A10。
10. **Stripe 正确性边界**：局部行合并与连续 buffer 优化默认 opt-in；对 stripe 基线精确一致不等于对 Ulysses 一致。不要将诊断开关当作生产正确性保证。
11. **Profiler 开销与文件体积**：长视频 trace 可达 GB 级，单独保存和测量。编译、加载和 warmup 不应混入 steady-state 延迟。

## 7. 详细报告索引

- [原模型 PP / TP / SP / overlap](original_parallel_reproduction.md)
- [SF scaling 与双请求](sf_14b_nvlink_overlap_benchmark.md)
- [SP 无 offload 分项分析](sp_nooffload_profile_analysis.md)
- [SP 数据重排优化](sp_layout_optimization.md)
- [长 query 的 head 效率](sp_head_compute_efficiency.md)
- [SF 自回归 head 效率](sf_head_compute_efficiency.md)
- [SF 三种 SP 比较](sf_parallel_comparison.md)
- [Stripe 优化全过程](sf_stripe_optimization.md)
- [10 秒结果](sf_stripe_10s_comparison.md)
- [30 秒结果](sf_stripe_30s_comparison.md)

这些 Markdown 跟随 GitHub 提交；原始 JSON、trace、latents、模型和安装日志保留在云盘，复查时必须同时恢复。
