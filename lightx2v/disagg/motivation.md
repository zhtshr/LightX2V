# 分离式 Diffusion Serving：Motivation 实验详表

> 本文档 **§1** 为按分辨率汇总的配置总览；**§2.2–§2.3.1** 等为分主题详表与解读。  
> 结论与总表见 [`draft_background_motivation.md`](draft_background_motivation.md)。

---

## 1. 分辨率 × 配置总览

**口径**：去噪 = `transformer_compute_s`（`scheduler.prepare` + denoise loop，不含 load/encoder/decoder）。**T₁** = 该分辨率 **P=1 单卡**去噪实测（小模型无 offload；大模型 MoE **block-offload P=1**）。**单请求扩展效率**：纯 **SP** = `T₁/(T_multi×P)`；**PP×SP** = `T₁/(T_multi×N)`（**N** = 实际占用 GPU 数，含 PP 与 SP 两维）。**overlap 扩展效率** = `n·T₁/(T_overlap×N)`（`n` = in-flight 请求数，`N` = GPU 数；vs `N` 路独立单卡理想吞吐）。**无实测 T₁ 时不填扩展效率**（**—**）。**—** = 未测或失败。

**×** 判定（同 §2.2）：**有 overlap 数据时**，**overlap wall > 单卡去噪 T₁** 或 **overlap 扩展效率 < 70%** 即标 ×；**无 overlap 数据时**，**单请求扩展效率 < 70%** 即标 ×。

### 1.1 小模型（Wan2.1-T2V-1.3B）

无 offload，Ulysses SP，4 steps。正方形 256²/512² 含双请求 overlap（P=3/4/6）；480×832 为在线分辨率。

#### 256×256（T₁ = **1.3 s**，seq = 5,376）

| × | 配置 | GPU | 单请求 (s) | 扩展效率 | 2-req overlap (s) | overlap 效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 1 | 1.3 | 100% | — | — |
| | SP P=2 | 2 | 0.9 | 70.8% | — | — |
| | SP P=3 | 3 | 0.7 | 61.1% | 1.0 | 86.6% |
| | SP P=4 | 4 | 0.6 | 58.1% | 0.8 | 79.8% |
| | SP P=6 | 6 | 0.5 | 47.7% | 0.6 | 70.7% |

#### 512×512（T₁ = **8.0 s**，seq = 21,504）

| × | 配置 | GPU | 单请求 (s) | 扩展效率 | 2-req overlap (s) | overlap 效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 1 | 8.0 | 100% | — | — |
| | SP P=2 | 2 | 5.1 | 78.5% | — | — |
| | SP P=3 | 3 | 3.7 | 71.8% | 5.4 | 98.4% |
| | SP P=4 | 4 | 2.8 | 71.6% | 4.1 | 98.4% |
| | SP P=6 | 6 | 2.0 | 65.9% | 2.9 | 92.2% |

#### 1024×1024（T₁ = **86.2 s**，seq = 86,016）

| × | 配置 | GPU | 单请求 (s) | 扩展效率 | 2-req overlap (s) | overlap 效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 1 | 86.2 | 100% | — | — |
| | SP P=2 | 2 | 44.0 | 97.8% | — | — |
| | SP P=3 | 3 | 30.6 | 93.9% | — | — |
| | SP P=4 | 4 | 22.8 | 94.6% | — | — |
| | SP P=6 | 6 | 15.6 | 92.0% | — | — |

#### 2048×2048（T₁ = **1373 s**，seq = 344,064）

| × | 配置 | GPU | 单请求 (s) | 扩展效率 | 2-req overlap (s) | overlap 效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 1 | 1373 | 100% | — | — |
| | SP P=2 | 2 | 731 | 93.9% | — | — |
| | SP P=3 | 3 | 483 | 94.8% | — | — |
| | SP P=4 | 4 | 359 | 95.7% | — | — |
| | SP P=6 | 6 | 222 | 103.0% | — | — |

#### 480×832（T₁ = **14.9 s**）

| × | 配置 | GPU | 单请求 (s) | 扩展效率 | 2-req overlap (s) | overlap 效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | 1 | 14.9 | 100% | — | — |
| | SP P=2 | 2 | 9.0 | 82.9% | 14.7 | 101.4% |
| | SP P=3 | 3 | 6.5 | 76.3% | 9.9 | 100.3% |
| | SP P=4 | 4 | 5.0 | 74.9% | 7.4 | 100.9% |
| | SP P=6 | 6 | 3.5 | 71.7% | 5.0 | 99.5% |

---

### 1.2 大模型（Wan2.2-MoE I2V）

**DiT offload**：**SP** 行均为 `block`（`cpu_offload=true`）；**PP×SP** 行为 `否`（`cpu_offload=false`）。**PP 路径当前不支持 block offload**（`NotImplementedError`），故无 offload 的 PP×SP OOM 配置以 **同 GPU 数 SP+block** 作对照（标注 `block`）。**hybrid** = 2 req GPipe；**quad** = 4 req GPipe + SP a2a overlap。

#### 256×256（T₁ = **6.7 s**，seq = 5,376）

| × | 配置 | DiT offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | block | 1 | 6.7 | 100% | — | — | 1 |
| × | SP P=2 | block | 2 | 6.2 | 53.8% | — | — | 1 |
| × | SP P=4 | block | 4 | 5.7 | 29.4% | 13.1 | 25.4% | 2 |
| × | SP P=8 | block | 8 | 5.5 | 15.2% | 11.8 | 14.2% | 2 |
| × | PP×SP P=2×2 hybrid | 否 | 4 | 5.7 | 29.4% | 4.8 | 69.2% | 2 |
| × | PP×SP P=2×2 quad | 否 | 4 | 5.6 | 29.9% | 8.1 | 82.8% | 4 |
| × | PP×SP P=2×4 hybrid | 否 | 8 | 4.1 | 20.4% | 2.8 | 58.7% | 2 |
| | PP×SP P=2×4 quad | 否 | 8 | 4.1 | 20.4% | 4.3 | 78.0% | 4 |

#### 512×512（T₁ = **37.8 s**，seq = 21,504）

| × | 配置 | DiT offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | block | 1 | 37.8 | 100% | — | — | 1 |
| | SP P=2 | block | 2 | 23.6 | 80.0% | — | — | 1 |
| × | SP P=4 | block | 4 | 13.9 | 68.0% | 29.4 | 64.4% | 2 |
| × | SP P=8 | block | 8 | 7.9 | 60.0% | 20.0 | 47.3% | 2 |
| × | PP×SP P=2×2 quad | 否 | 4 | 23.5 | 40.2% | 44.7 | 84.7% | 4 |
| × | PP×SP P=2×4 hybrid | 否 | 8 | 14.3 | 33.0% | 13.9 | 68.2% | 2 |
| | PP×SP P=2×4 quad | 否 | 8 | 14.5 | 32.6% | 22.7 | 83.2% | 4 |

#### 1024×1024（T₁ = **360.6 s**，seq = 86,016）

| × | 配置 | DiT offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | block | 1 | 360.6 | 100% | — | — | 1 |
| | SP P=2 | block | 2 | 199.0 | 90.6% | — | — | 1 |
| | SP P=4 | block | 4 | 101.5 | 88.8% | 188.5 | 95.7% | 2 |
| | SP P=8 | block | 8 | 54.2 | 83.1% | 103.1 | 87.5% | 2 |
| | PP×SP P=2×2 hybrid | 否 | 4 | — (OOM) | — | — | — | 2 |
| | PP×SP P=2×2 quad | 否 | 4 | — (OOM) | — | — | — | 4 |
| | PP×SP P=2×4 hybrid | 否 | 8 | 99.6 | 45.3% | 106.9 | 84.3% | 2 |
| | PP×SP P=2×4 quad | 否 | 8 | — (err‡) | — | — | — | 4 |

PP×SP P=2×2：**否** offload → OOM；**block** 对照见上行 SP P=4 **双请求 overlap**（188.5 s）。P=2×4 quad：err‡；对照见 SP P=8 overlap（103.1 s）。

#### 2048×2048（**无实测 T₁**，seq = 344,064）

| × | 配置 | DiT offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | block | 1 | — (OOM) | — | — | — | 1 |
| | SP P=2 | block | 2 | — (OOM) | — | — | — | 1 |
| | SP P=4 | block | 4 | 1419 | — | 未测 | — | 1 |
| | SP P=8 | block | 8 | 705 | — | 未测 | — | 1 |
| | PP×SP P=2×2 hybrid | 否 | 4 | — (OOM) | — | — | — | 2 |
| | PP×SP P=2×2 quad | 否 | 4 | — (OOM) | — | — | — | 4 |
| | PP×SP P=2×4 hybrid | 否 | 8 | — (OOM) | — | — | — | 2 |
| | PP×SP P=2×4 quad | 否 | 8 | — (OOM) | — | — | — | 4 |

P=1/P=2 已开 **block** 仍 OOM。PP×SP 四配置 **否** offload 均 OOM；SP 双请求 overlap **未测**（单路 P=4 已 ~24 min）。

#### 480×832（T₁ = **72.6 s**）

| × | 配置 | DiT offload | GPU | 单请求 (s) | 扩展效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | SP P=1 | block | 1 | 72.6 | 100% | — | — | 1 |
| × | SP P=2 | block | 2 | 43.4 | 83.6% | 86.3 | 84.2% | 2 |
| | SP P=4 | block | 4 | 24.5 | 74.0% | 50.5 | 72.0% | 2 |
| × | SP P=8 | block | 8 | 13.3 | 68.3% | 29.1 | 62.4% | 2 |
| | PP×SP P=2×2 hybrid | 否 | 4 | 43.0 | 42.2% | 46.1 | 79.0% | 2 |
| × | PP×SP P=2×2 quad | 否 | 4 | 42.8 | 42.4% | 83.8 | 86.8% | 4 |
| | PP×SP P=2×4 hybrid | 否 | 8 | 23.8 | 38.1% | 24.8 | 73.4% | 2 |
| | PP×SP P=2×4 quad | 否 | 8 | 23.8 | 38.1% | 42.3 | 86.0% | 4 |

数据来源：小模型 §2.2；大模型 SP §2.3.0、PP×SP §2.3.1、480×832 PP×SP §2.3.3–§2.3.6。

---

## 2.2 小模型：分辨率 × SP（Wan2.1-T2V-1.3B）

无 offload，Ulysses SP，4 denoise steps，`enable_cfg=false`。**扩展效率** = `(T₁/T_SP)/P`；**双请求 overlap 延迟** = 2 req in-flight 的总 wall time；**overlap 扩展效率** = 相对同 GPU 数独立单卡理想吞吐 `N/T₁`（即 `2·T₁/(P·T_overlap)`）。

**×** 判定：**有 overlap 数据时**，**overlap wall > 单卡去噪 T₁** 或 **overlap 扩展效率 < 70%** 即标 ×；**无 overlap 数据时**，**单请求扩展效率 < 70%** 即标 ×。

### 2.2.1 分辨率汇总

| 分辨率 | seq tokens | 单卡去噪 (s) | P=4 扩展效率 | P=6 扩展效率 |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 5,376 | **1.3** | **58.1%** | **47.7%** |
| 512×512 | 21,504 | **8.0** | 71.6% | 65.9% |
| 1024×1024 | 86,016 | **86.2** | **94.6%** | 92.0% |
| 2048×2048 | 344,064 | **1373** | **95.7%** | 103.0% |

数据来源：`p1_t2v_1.3b_{res}x{res}_seqp{1,4,6}.json`。

### 2.2.2 按并行度分解

正方形分辨率双请求 overlap 仅测 256²/512²（P=3/4/6）；480×832 为在线分辨率对照。

#### SP P=1

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | 480×832 | 14.9 | 14.9 | 100% | — | — |

#### SP P=2

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | 256×256 | 1.3 | 0.9 | 70.8% | — | — |
| | 512×512 | 8.0 | 5.1 | 78.5% | — | — |
| | 1024×1024 | 86.2 | 44.0 | 97.8% | — | — |
| | 2048×2048 | 1373 | 731 | 93.9% | — | — |
| | 480×832 | 14.9 | 9.0 | 82.9% | 14.7 | 101.4% |

#### SP P=3

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | 256×256 | 1.3 | 0.7 | 61.1% | 1.0 | 86.6% |
| | 512×512 | 8.0 | 3.7 | 71.8% | 5.4 | 98.4% |
| | 1024×1024 | 86.2 | 30.6 | 93.9% | — | — |
| | 2048×2048 | 1373 | 483 | 94.8% | — | — |
| | 480×832 | 14.9 | 6.5 | 76.3% | 9.9 | 100.3% |

#### SP P=4

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | 256×256 | 1.3 | 0.6 | 58.1% | 0.8 | 79.8% |
| | 512×512 | 8.0 | 2.8 | 71.6% | 4.1 | 98.4% |
| | 1024×1024 | 86.2 | 22.8 | 94.6% | — | — |
| | 2048×2048 | 1373 | 359 | 95.7% | — | — |
| | 480×832 | 14.9 | 5.0 | 74.9% | 7.4 | 100.9% |

#### SP P=6

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | 256×256 | 1.3 | 0.5 | 47.7% | 0.6 | 70.7% |
| | 512×512 | 8.0 | 2.0 | 65.9% | 2.9 | 92.2% |
| | 1024×1024 | 86.2 | 15.6 | 92.0% | — | — |
| | 2048×2048 | 1373 | 222 | 103.0% | — | — |
| | 480×832 | 14.9 | 3.5 | 71.7% | 5.0 | 99.5% |

数据来源：正方形分辨率 `p1_t2v_1.3b_{res}x{res}_seqp{1,P}.json`、`p3_dual_overlap_t2v_1.3b_{res}x{res}_seqp{P}.json`；480×832 `p3_t2v_1.3b_sp_seqp{1,P}.json`、`p3_dual_overlap_t2v_1.3b_seqp{P}.json`。汇总 JSON：`phase3_dual_overlap_t2v_resolution.json`。

---

## 2.3.0 大模型：分辨率 × SP（Wan2.2-MoE I2V）

`cpu_offload=true`（block），Ulysses SP，4 denoise steps；口径同 §2.2。**所有 SP 行 DiT offload = block**。正方形分辨率双请求 overlap 已测 256²/512²（P=4/8）；480×832 为在线分辨率对照。**×** 含义同 §2.2。

**单请求 vs 双请求 overlap 路径**：「单请求延迟」= **`model.infer()`**；「双请求 overlap」= **decomposed + a2a patch 交织**（见 `p3_dual_overlap_moe_i2v_*`），通常比同 P 两路 `model.infer()` 之和慢约 **20–25%**（P=8 上 512²+480×832：21.2 s vs 26.3 s decomposed b2b）。

### 分辨率汇总

| 分辨率 | seq tokens | 单卡去噪 (s) | P=2 扩展效率 | P=4 扩展效率 | P=8 扩展效率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256×256 | 5,376 | **6.7** | **53.8%** | **29.4%** | **15.2%** |
| 512×512 | 21,504 | **37.8** | 80.0% | **68.0%** | **60.0%** |
| 1024×1024 | 86,016 | **360.6** | **90.6%** | **88.8%** | **83.1%** |
| 2048×2048 | 344,064 | — (OOM) | — | — | — |

数据来源：`p1_moe_i2v_{res}x{res}_seqp{1,2,4,8}.json`；480×832 P=1 `p3_sp_seqp1.json`；汇总见 `phase1_moe_i2v_resolution_sp_scaling.md`。

### 按并行度分解

#### SP P=1

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | 480×832 | 72.6 | 72.6 | 100% | — | — |
| | 2048×2048 | — (OOM) | — (OOM) | — | — | — |

#### SP P=2

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| × | 256×256 | 6.7 | 6.2 | 53.8% | — | — |
| | 512×512 | 37.8 | 23.6 | 80.0% | — | — |
| | 1024×1024 | 360.6 | 199.0 | 90.6% | — | — |
| | 2048×2048 | — (OOM) | — (OOM) | — | — | — |
| × | 480×832 | 72.6 | 43.4 | 83.6% | **86.3** | 84.2% |

#### SP P=4

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| × | 256×256 | 6.7 | 5.7 | 29.4% | **13.1** | 25.4% |
| × | 512×512 | 37.8 | 13.9 | 68.0% | 29.4 | 64.4% |
| | 1024×1024 | 360.6 | 101.5 | 88.8% | **188.5** | 95.7% |
| | 2048×2048 | — (OOM) | 1419 | — | — | — |
| | 480×832 | 72.6 | 24.5 | 74.0% | 50.5 | 72.0% |

#### SP P=8

| × | 分辨率 | 单卡去噪 (s) | 单请求延迟 (s) | 扩展效率 | 双请求 overlap (s) | overlap 扩展效率 |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| × | 256×256 | 6.7 | 5.5 | 15.2% | **11.8** | 14.2% |
| × | 512×512 | 37.8 | 7.9 | 60.0% | 20.0 | 47.3% |
| | 1024×1024 | 360.6 | 54.2 | 83.1% | **103.1** | 87.5% |
| | 2048×2048 | — (OOM) | 705 | — | — | — |
| × | 480×832 | 72.6 | 13.3 | 68.3% | 29.1 | 62.4% |

数据来源：正方形分辨率 `p1_moe_i2v_{res}x{res}_seqp{1,P}.json`、`p3_dual_overlap_moe_i2v_{res}x{res}_seqp{P}.json`；480×832 P=1 `p3_sp_seqp1.json`、P=2/4 `p3_sp_seqp{P}.json`、`p3_dual_overlap_seqp{P}.json`、P=8 `p3_dual_overlap_moe_i2v_480x832_seqp{P}.json`。

### 混分辨率 SP 双请求 a2a overlap

**可用性判据（混分辨率，比 §2.2 更严）**：小分辨率租户若跟批跑 overlap，需同时满足：(1) **`T_overlap` ≤ 同 P 两路 `model.infer` 之和**（`dual_back_to_back_s`）；(2) **`T_overlap` ≤ 小分辨率 P=1 单卡 T₁**（否则小图等整批 wall 不如自己单卡）。任一不满足即标 ×。

**结论：混分辨率 SP a2a overlap 实测均不可用。** 6 组配置（512+1024 / 512+480 × P=2/4/8）**全部**违反 (2)（小分辨率 T₁=**37.8 s**，overlap 最短亦 **24.5 s** 量级）；512+480 在 P=2/8 还违反 (1)（overlap 不比 `model.infer` 串行快）。同 P **同分辨率**双请求则不同——如 512²+512² P=8 overlap **20.0 s** < T₁，有 overlap 收益。混分辨率仅验证「能跑通」，**不适合**作为在线混批调度依据。

#### 512² + 1024²

tenant A = **512×512**（T₁ = **37.8 s**），tenant B = **1024×1024**（T₁ = **360.6 s**）；block offload、4 steps。路径：a2a serial/overlap = decomposed 交织；**SP 单请求之和 / b2b infer** = 两路 `model.infer()`（与 §2.3.0 单请求列同口径）。

| × | P | GPU | SP 单请求之和 (s) | b2b infer (s) | a2a overlap (s) | vs b2b | vs T₁(512) |
| :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| × | 2 | 2 | 222.6 | 221.8 | **207.6** | 快 6% | **5.5×** |
| × | 4 | 4 | 115.4 | 115.4 | **108.8** | 快 6% | **2.9×** |
| × | 8 | 8 | 62.1 | 62.3 | **60.9** | 快 2% | **1.6×** |

相对 `model.infer` 串行仅有 **2–6%** 收益，但小图租户 wall 仍为单卡 **1.6–5.5×**。数据来源：`p3_dual_overlap_moe_i2v_mixed_512_1024_seqp{2,4,8}.json`。

#### 512² + 480×832

tenant A = **512×512**（T₁ = **37.8 s**），tenant B = **480×832**（T₁ = **72.6 s**）；口径同上。

| × | P | GPU | SP 单请求之和 (s) | b2b infer (s) | a2a overlap (s) | vs b2b | vs T₁(512) |
| :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| × | 2 | 2 | 67.0 | 67.2 | **67.4** | 慢 | **1.8×** |
| × | 4 | 4 | 38.4 | 38.5 | **38.3** | ≈ | **1.0×** |
| × | 8 | 8 | 21.2 | 21.1 | **24.5** | **慢 16%** | 0.65× |

P=4 overlap **38.3 s** 与 T₁ 打平仍无收益；P=8 虽 **< T₁**，但比 `model.infer` 串行 **慢 16%**，小图同 P 单请求仅 **7.9 s**。数据来源：`p3_dual_overlap_moe_i2v_mixed_512_480_seqp{2,4,8}.json`。

---

## 2.3.1 PP×SP hybrid / quad vs 分辨率（256²–2048²）

`pp_layers_per_stage=2`，4 steps，8×A10 24GB。**PP×SP 行 DiT offload = 否**（`cpu_offload=false`；**PP 不支持 block offload**）。OOM 的 PP×SP 以 **同 GPU 数 SP+block 的双请求 overlap** 作对照，见 §1.2 / §2.3.0 对应 SP 行的 overlap 列。

**T₁ 与效率口径（与 §2.3.0 统一）**：MoE 在 A10 24GB 上 **单卡只能 block-offload 跑通**（§2.3.0「单卡去噪」列）；不存在可实测的「单卡无 offload」基线。PP×SP 路径在 **多卡** 上去掉 block offload（PP 切层）；表中 **T₁** = 各分辨率 **block-offload P=1 实测**。**无实测 T₁ 时不填单请求 / overlap 扩展效率**（如 2048² P=1 OOM）。**多卡单请求** = N 卡 PP×SP 下单路 wall，**不是**单卡时间。**单请求扩展效率** = `T₁/(T_multi×N)`（**N** = 占用 GPU 数；纯 SP 时 N=P，PP×SP 时 N=PP×SP 总卡数）；**overlap 扩展效率** = `n·T₁/(T_overlap·N)`（vs **N 路互不干扰的 block-offload 单卡** 理想吞吐 `N/T₁`）。**—** = 该配置 OOM 或 quad 管线非法访存。**×** 含义同 §2.2。

**DiT 无 offload 可行性**（MoE 权重常驻 GPU、`cpu_offload=false`；8×A10 24GB）。**OOM 格**的对照数据见 §1.2 同 GPU 数 **SP overlap** 列：

| 分辨率 | P=2×2 hybrid | P=2×2 quad | P=2×4 hybrid | P=2×4 quad | SP overlap 对照 |
| --- | :---: | :---: | :---: | :---: | :---: |
| 256² | ✓ | ✓ | ✓ | ✓ | — |
| 512² | ✓ | ✓ | ✓ | ✓ | — |
| 1024² | OOM | OOM | ✓ | err‡ | SP P=4 **188.5 s** / P=8 **103.1 s** |
| 2048² | OOM | OOM | OOM | OOM | 未测 |

480×832 四配置均可无 offload（§2.3.3–§2.3.6）。**✓** = bench 跑通；**OOM** = 去噪阶段显存不足；**err‡** = 1024² quad 在 SP a2a overlap 阶段 illegal memory access（非单纯 OOM）。

| × | 配置 | 分辨率 | DiT offload | T₁ block P=1 (s) | 多卡单请求 (s) | 单请求效率 | overlap (s) | overlap 效率 | 并发 |
| :---: | --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| × | PP×SP P=2×2 hybrid | 256×256 | 否 | 6.7 | **5.7** | **29.4%** | 4.8 | 69.2% | 2 req |
| × | PP×SP P=2×2 quad | 256×256 | 否 | 6.7 | **5.6** | **29.9%** | **8.1** | 82.8% | 4 req |
| × | PP×SP P=2×4 hybrid | 256×256 | 否 | 6.7 | **4.1** | **20.4%** | 2.8 | 58.7% | 2 req |
| | PP×SP P=2×4 quad | 256×256 | 否 | 6.7 | **4.1** | **20.4%** | 4.3 | 78.0% | 4 req |
| | PP×SP P=2×2 hybrid | 512×512 | 否 | 37.8 | **23.4** | **40.4%** | 24.8 | 76.1% | 2 req |
| × | PP×SP P=2×2 quad | 512×512 | 否 | 37.8 | **23.5** | **40.2%** | **44.7** | 84.7% | 4 req |
| × | PP×SP P=2×4 hybrid | 512×512 | 否 | 37.8 | **14.3** | **33.0%** | 13.9 | 68.2% | 2 req |
| | PP×SP P=2×4 quad | 512×512 | 否 | 37.8 | **14.5** | **32.6%** | 22.7 | 83.2% | 4 req |
| | PP×SP P=2×2 hybrid | 1024×1024 | 否 | 360.6 | **—** (OOM) | **—** | **—** | **—** | 2 req |
| | PP×SP P=2×2 quad | 1024×1024 | 否 | 360.6 | **—** (OOM) | **—** | **—** | **—** | 4 req |
| | PP×SP P=2×4 hybrid | 1024×1024 | 否 | 360.6 | **99.6** | **45.3%** | 106.9 | 84.3% | 2 req |
| | PP×SP P=2×4 quad | 1024×1024 | 否 | 360.6 | **—** (err‡) | **—** | **—** | **—** | 4 req |
| | PP×SP P=2×2 hybrid | 2048×2048 | 否 | — (OOM) | **—** | **—** | **—** | **—** | 2 req |
| | PP×SP P=2×2 quad | 2048×2048 | 否 | — (OOM) | **—** | **—** | **—** | **—** | 4 req |
| | PP×SP P=2×4 hybrid | 2048×2048 | 否 | — (OOM) | **—** | **—** | **—** | **—** | 2 req |
| | PP×SP P=2×4 quad | 2048×2048 | 否 | — (OOM) | **—** | **—** | **—** | **—** | 4 req |

对比 480×832（§2.3.3–§2.3.6，lps=2）：P=2×2 hybrid 单请求 **43.0 s** / 2-req **46.1 s**；P=2×2 quad 单请求 **42.8 s** / 4-req **83.8 s**；P=2×4 hybrid **23.8 s** / **24.8 s**；P=2×4 quad **23.8 s** / **42.3 s**。低分辨率下单请求更快，但 512² hybrid 2-req wall（24.8 s）已略高于单请求（23.4 s），GPipe bubble 占比上升；256² 上 P=2×4 hybrid 2-req（2.8 s）仍明显低于单请求（4.1 s）。

**1024² / 2048² 解读**：

1. **1024² P=2×2（4 卡，DiT offload=否）**：双 MoE expert 常驻 + SP seq/2 激活，去噪 a2a OOM（缺 ~400 MiB）。**block 对照**：同 4 卡 SP P=4 双请求 overlap **188.5 s**，效率 **95.7%**（§1.2）。
2. **1024² P=2×4（8 卡，否）**：多卡单请求 **99.6 s**（单请求效率 **45.3%**，8 卡理想应达 ~45 s 却用了 99.6 s）；hybrid 2-req **106.9 s**，overlap 效率 **84.3%**。**quad err‡**；对照 SP P=8 overlap **103.1 s**（§1.2）。
3. **2048²**：SP block 单请求 P=4 **1419 s** / P=8 **705 s**；PP×SP **否** offload 四配置 OOM。SP 双请求 overlap **未测**。

数据来源：`wan22_pp2_sp{2,4}_{hybrid,quad}_{res}x{res}_lps2_*.json`；`p1_moe_i2v_*_seqp*.json`；`p3_dual_overlap_moe_i2v_{res}x{res}_seqp{P}.json`。

---

## 2.3.2 PP=2：`pp_layers_per_stage` sweep（2 GPU，GPipe m=2）

2 请求 in-flight；吞吐 = `2 / dual_wall_s`；理想上限 = `2 / 72.67 = 0.0275 req/s`。数据来源：`wan22_pp_lps_sweep.json`。

| lps | stages | 单请求去噪 (s) | 双租户 wall (s) | 吞吐 (req/s) | vs 2×P1 理想 | 理论 slot util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 2 | 71.0 | 106.3 | 0.0188 | 68% | 66.7% |
| 10 | 4 | 70.5 | 88.3 | 0.0227 | 82% | 80.0% |
| 5 | 8 | 71.2 | 81.9 | 0.0244 | 89% | 88.9% |
| 4 | 10 | 71.3 | 80.3 | 0.0249 | 91% | 90.9% |
| **2** ★ | **20** | **70.6** | **77.0** | **0.0260** | **94%** | **95.2%** |
| 1 | 40 | 70.5 | 79.1 | 0.0253 | 92% | 97.6% |

★ 吞吐最优。lps=1 理论 util 最高（97.6%），但 39 次 P2P/forward 使实测 wall 慢于 lps=2（profile 见 `wan22_pp2_lps1_lps2_stage_comm_profile.json`）。

---

## 2.3.3 PP×SP P=2×2 hybrid：`pp_layers_per_stage` sweep（4 GPU，GPipe m=2）

2 请求 in-flight；吞吐 = `2 / gpipe_wall_s`；理想上限 = `4 / 72.67 = 0.0550 req/s`。T₁ block P=1 = **72.6 s**。**×** 同 §2.2。

| × | lps | stages | 单请求去噪 (s) | 双租户 wall (s) | 吞吐 (req/s) | vs 4×P1 理想 | 理论 slot util |
| :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| × | 20 | 2 | 42.9 | 63.3 | 0.0316 | 57% | 66.7% |
| × | 10 | 4 | 42.8 | 52.9 | 0.0378 | 69% | 80.0% |
| | 5 | 8 | 42.9 | 48.8 | 0.0410 | 75% | 88.9% |
| | **2** ★ | **20** | **43.0** | **46.1** | **0.0434** | **79%** | **95.2%** |

★ 吞吐最优。未测 lps=1/4；粗分（lps≥10）在修复前 pipeline 有 bug，旧数据已弃用。

---

## 2.3.4 PP×SP quad：`pp_layers_per_stage` sweep（4 GPU，4 req，GPipe m=2 + SP a2a）

4 请求 in-flight；吞吐 = `4 / quad_wall_s`（**quad pp+sp overlap** 模式）；理想上限 = `0.0550 req/s`。T₁ = **72.6 s**。**×** 同 §2.2。

| × | lps | stages | 单请求去噪 (s) | quad wall (4 req, s) | 吞吐 (req/s) | vs 4×P1 理想 | 理论 PP util (m=2) |
| :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| × | **2** ★ | **20** | **42.8** | **83.8** | **0.0478** | **86.8%** | **95.2%** |
| × | 4 | 10 | 42.9 | 85.9 | 0.0466 | 84.6% | 90.9% |
| × | 5 | 8 | 43.1 | 87.6 | 0.0457 | 83.0% | 88.9% |
| × | 10 | 4 | 42.8 | 96.0 | 0.0417 | 75.8% | 80.0% |

★ 吞吐最优。同 lps 下 quad pp-only（无 SP overlap）约低 **~10%** 吞吐；SP overlap 节省 wall **~8–10 s**（见 §2.3.5）。

---

## 2.3.5 PP×SP quad overlap：4 请求叠 PP GPipe 与 SP a2a

[`draft_background_motivation.md`](draft_background_motivation.md) §2.3.1 / §2.3.3 中 **PP×SP P=2×2** 仅在 **2 个 in-flight 请求** 下用 GPipe 填 PP bubble，SP 维仍走标准 Ulysses，**未**在 comm 窗口做双租户 a2a overlap。进一步地，我们实现 **quad overlap**：在 **4 GPU、4 并发请求** 上同时叠加两层重叠：

| 机制 | 作用 | 参数 |
| --- | --- | --- |
| **PP GPipe** | 两列 `pipe_p` 各跑 m=2 微批，填 stage bubble | lps=2 → 20 stage；理论 slot 利用率 **m·S/(pp·(m+S−1)) = 40/42 ≈ 95.2%** |
| **SP a2a overlap** | 同一 GPipe slot 内，请求对 (0↔2)、(1↔3) 在 `seq_p` 上做 layer-wise a2a 与对端计算重叠 | `A2AOrchestrator` patch `all_to_all` / `all_gather` |

**与 2-req hybrid 的区别**：hybrid 4 卡只服务 2 req（吞吐 **0.0434**）；quad 需 **4 req** 同时 in-flight，第 3、4 路分别与第 1、2 路组成 SP 对，在各自 PP 列内仍共享 m=2 GPipe 时间表。

实测（Wan2.2-MoE I2V，480×832，4 steps，无 offload，`wan22_pp2_sp2_quad_overlap_lps2.json`）。T₁ = block-offload P=1 **72.6 s**（§2.3.0）。**×** 同 §2.2。

| × | 模式 | 并发 | wall (s) | 吞吐 (req/s) | vs 4×单卡理想 |
| :---: | --- | ---: | ---: | ---: | ---: |
| × | 单请求（4 卡 SP+PP） | 1 | 42.8 | 0.0234 | 42% |
| | PP GPipe m=2 | 2 | 46.0 | 0.0435 | 79% |
| × | quad，仅 PP（无 SP overlap） | 4 | **92.5** | 0.0432 | 78.5% |
| × | **quad，PP + SP overlap** | **4** | **83.8** | **0.0478** | **86.8%** |
| × | 4×串行 naive | 4 | 171.2 | — | — |

**解读**：

1. **SP overlap 增量**：quad pp-only → quad pp+sp 节省 **~8.7 s** wall（**~10%**），与 SP P=2 上 a2a-overlap 的收益量级一致；4 req 串行 naive 171 s → 83.8 s，约 **2.05×** 加速。
2. **vs SP P=4 dual**：quad **0.0478** vs SP **0.0396 req/s**（**+21%**）——在同样 4 卡、无 offload 前提下，**PP 填 bubble + SP overlap** 优于纯 SP 双租户。
3. **vs PP×SP 2-req**：**0.0478** vs **0.0434**（**+10%**）——多 2 路并发换 SP comm 重叠窗口，但未到 4×单卡理想 **0.0550**（**87%**），距 PP 理论 slot **95%** 仍有 **~8%** 缺口。
4. **Profile 结论**（`wan22_pp2_sp2_quad_overhead_v2.json`）：GPU SM 稳态 **~99%**；关键路径主要在 **stage 内 a2a stream 等待** 与 **GPipe slot barrier**，非 step 级 barrier 或 metadata；lps=1、减 barrier、async P2P 等微优化在 ±1 s 噪声内**无稳定收益**。

实现入口：`scripts/disagg/pp_sp_quad_overlap.py`（`run_gpipe_quad_sp_pipeline`）；bench：`scripts/disagg/run_pp_sp_quad_overlap_bench.py`。

---

## 2.3.6 PP×SP quad P=2×4：`pp_layers_per_stage` sweep（8 GPU，4 req，GPipe m=2 + SP a2a）

4 请求 in-flight（`2 × m`，每 PP 微批 1 对 SP dual overlap，租户 (0↔1)、(2↔3)）；吞吐 = `4 / quad_wall_s`；理想上限 = `8 / 72.67 = 0.1101 req/s`。T₁ = **72.6 s**。**×** 同 §2.2。

| × | lps | stages | 单请求去噪 (s) | quad wall (4 req, s) | 吞吐 (req/s) | vs 8×P1 理想 | 每路延迟 (s) | 理论 PP util (m=2) |
| :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | **2** ★ | **20** | **23.8** | **42.3** | **0.0946** | **86.0%** | **~42** | **95.2%** |
| | 4 | 10 | 24.0 | 44.6 | 0.0898 | 81.5% | ~45 | 90.9% |
| | 5 | 8 | 24.6 | 45.5 | 0.0880 | 79.9% | ~45 | 88.9% |
| | 10 | 4 | 23.9 | 50.0 | 0.0800 | 72.7% | ~50 | 80.0% |

★ 吞吐最优。SP overlap 相对 pp-only 节省 wall **~7 s**（lps=2：49.7 s → 42.3 s）。**每路延迟** ≈ quad wall（4 路 step 同步），非 `wall/4` 摊销值（~10.6 s）。

---

## 2.4 为何不用 Tensor Parallel（TP）：实测结论

在 Wan2.2-MoE I2V 上我们曾探索 **Tensor Parallel（TP）** 作为「消除 CPU offload + 双租户通信重叠」的路径：权重按列切分到多卡、每层多次 `all_reduce`，并在 COMM 窗口内 pump 另一租户的 COMP。工程上已完成真分片（`MMWeightTP` / `tp_load`），但**端到端与微基准一致表明：TP 不应作为 480×832 降延迟或双卡提吞吐的手段；同卡数下 SP 全面优于 TP。**

数据口径：`transformer_compute_s` = scheduler.prepare + denoise loop（不含 load/encoder/decoder）；480×832×81、4 denoise steps、int8-q8f，A10 24GB。详见 `save_results/optimization_study/` 下 `wan22_distill_tp_fair_comparison.md`、`p3_sp_seqp*.json`、`wan22_256_tp2_e2e_suite.json`。

### 2.4.1 单请求延迟：TP 仅略优于 offload 单卡，远逊于 SP

| 方案 | 配置要点 | 去噪时间 | vs TP=1 offload |
| --- | --- | ---: | ---: |
| TP=1 + block offload | 单卡塞不下权重 | **73.3 s** | 1.00× |
| SP P=1（无 offload） | 单卡无 offload 基线 | **72.6 s** | ≈1.00× |
| **TP=2**（真分片，无 offload） | `tensor_p_size=2` | **57.9 s** | **1.26×** |
| **SP P=2** | `seq_p_size=2` | **43.4 s** | **1.69×** |
| **SP P=4** | `seq_p_size=4` | **24.5 s** | **2.99×** |

**解读**：

- TP=2 相对 offload 单卡的 **~21% 加速**，主要来自 **去掉 CPU offload**（PCIe 权重搬运消失），而非有效的多卡算力扩展。
- 同样 2 卡、同样无 offload，**SP P=2（43 s）比 TP=2（58 s）快约 33%**——若目标是单请求变快，没有理由选 TP。
- SP P=4 可将去噪压到 **~25 s**（[`draft_background_motivation.md`](draft_background_motivation.md) §2.3.1）；TP 在 4 卡上无法接近这一水平（见下节）。

### 2.4.2 扩展性：TP 在 MoE 上几乎不随卡数扩展

公平复测（`cpu_offload=false`，`unload_modules=false`，warmup=2，iters=5，配置 `wan22_moe_i2v_tp_fair_bench.json`）：

| TP 度 | 去噪 mean (s) | 相对 TP=2 |
| --- | ---: | ---: |
| 2 | **57.85** | 1.00× |
| 4 | FAILED（OOM / 不稳定） | — |
| 8 | **58.84** | **0.98×** |

TP=2→TP=8 **无加速**（spread 仅 1.7%）。根因是 Wan MoE 每层 **self-attn、cross-attn、FFN 均含 row/column `all_reduce`**，通信频率高；int8 GEMM 在 TP 切分后 local batch 变小，算力效率也不升。对比 SP：P=2→P=4 仍有 **~1.75×** 增量加速（43 s→25 s）。

**结论**：多卡买延迟应走 **SP 提高 `seq_p_size`**，而非堆 `tensor_p_size`。

### 2.4.3 双租户 overlap：TP 专用 6-phase 管线未带来 E2E 收益

我们实现了 TP 专用的 **6-phase dual-tenant pipeline**（COMP/COMM 拆相位、`tp_norm_p2p`、stream 同步与 barrier 修复等），并在 256×256 上做了完整 E2E 对比（`wan22_256_tp2_e2e_suite`）：

| 模式（256×256，2 请求，4 steps） | 吞吐 (req/s) | vs dual b2b |
| --- | ---: | ---: |
| dual back-to-back（两次 `model.infer`） | **0.1515** | 100% |
| dual 6-phase **overlap**（最佳修复栈） | 0.1359 | **90%** |
| dual AR-overlap（SP 风格移植） | 0.1417 | 94% |

256×256 上 **TP=2 单请求（6.55 s）与 TP=1（6.70 s）几乎相同**；双卡绑在一个 TP 组内跑两租户，吞吐只有「两路独立单卡」的 **~50%**，overlap 仍慢于串行 b2b。

480×832 上：默认精度双租户 overlap **OOM**；f32 配置下 ar-overlap 仅比 b2b 好 **~2%**（0.050 vs 0.0499 req/s），与 MoE 上 SP a2a-overlap 的 **~4%**（[`draft_background_motivation.md`](draft_background_motivation.md) §2.3）同属「带宽饱和、overlap 空间极小」一类，不构成选型理由。

**微基准陷阱（勿作 go/no-go 依据）**：孤立 A2B1 pair 曾报告 overlap **13 ms < serial_sum 16 ms**，但前者用 GPU Event、后者用分段 `cuda.synchronize` 的 comm+comp 之和，**与 E2E 的 `pair_wall` 口径不一致**；对齐后 isolated 至多打平，E2E 40-layer 均值仍为 overlap **慢 ~1 ms/pair**。详见 `save_results/nsys/tp2_overlap_root_cause.md` 与 `wan22_256_tp2_pair_vs_b2b.md`。

### 2.4.4 通信与硬件：为何 TP 更难 overlap、也难拼过 SP

| 维度 | SP（Ulysses） | TP |
| --- | --- | --- |
| 主要 collective | self-attn 内 `all_to_all`（序列维切分） | 每层多次 **`all_reduce`**（row/column） |
| 通信量 vs 计算 | 高分辨率时 attention O(S²) 淹没 comm | 每层 FFN/attn 后均有固定宽度 AR，**comm 占比高且频** |
| 单请求多卡收益 | P=4 实测 **2.9×**（§2.3.1 总表） | P=2 **1.26×**（相对 offload），P≥4 **~1×** |
| 双租户 overlap | 小模型有效（[`draft_background_motivation.md`](draft_background_motivation.md) §2.2）；MoE **~4%** | 256 E2E **负收益**；480 MoE 可忽略 |
| 与 flash-attn / PCIe | A2A 窗口相对集中 | NCCL AR 与 sage attention **同卡争 PCIe**，Nsight 并发 **<45%** |

TP overlap 失败不是单一 bug，而是 **AR 与 dense attention 在同一张 A10（PCIe）上无法时间并行** 叠加 **分解调度路径相对 `model.infer` 的固定开销**；256 上 b2b 已是实际上限。

### 2.4.5 与 SP 的对照：该怎么选

| 目标 | 推荐 | 不推荐 |
| --- | --- | --- |
| 480×832 **单请求**降延迟 | **SP P=2 / P=4**（43 s / 25 s） | TP=2（58 s），TP≥4 |
| 480×832 **两路吞吐** | 多卡各跑 **独立 SP P=1 或副本** | TP=2 绑卡 + overlap |
| 256×256 任意 | **单卡 × N 副本** | TP=2 或 SP（compute 太短，comm 主导） |
| MoE + offload 下 overlap | 收益均 **≤4%**；靠调度/降 P，非 TP | 投入 TP 6-phase overlap 工程 |

> **小结（TP）**：TP=2 的唯一实惠是「无 offload 跑通 14B」且比 offload 单卡快一截，但 **同卡数 SP 更快、多卡 SP 可继续扩展而 TP 不能**；为 TP 定制的双租户 overlap 在 E2E 上未跑赢 b2b。**生产路径应默认 Ulysses SP（如 `wan22_i2v_distill_sp4_controller.json`），不将 TP 作为并行策略。**
