# Pull Request: Enable Disaggregation Feature

## 概述 (Overview)

本PR为LightX2V框架添加了**分布式计算（Disaggregation）**功能，允许将视频生成流程分解为独立的服务组件，实现更灵活的资源调度和部署架构。

This PR adds **Disaggregation** capabilities to the LightX2V framework, enabling the video generation pipeline to be decomposed into independent service components for more flexible resource scheduling and deployment architectures.

## 🚀 主要特性 (Key Features)

### 1. 服务解耦 (Service Decoupling)
- **编码器服务 (Encoder Service)**: 独立的VAE编码器和文本/图像编码器服务
- **Transformer服务**: 独立的Transformer模型推理服务
- 支持跨网络通信，实现计算资源的物理分离

### 2. 通信协议 (Communication Protocol)
- 基于ZeroMQ的高性能消息传递
- 支持HTTP异步通信
- 集成Mooncake传输引擎进行高效数据传输

### 3. 灵活部署 (Flexible Deployment)
- 支持单机分离部署（不同GPU/设备）
- 支持跨机器分布式部署
- 支持多种硬件加速器（GPU, GCU, MUSA等）

## 📁 新增文件 (New Files)

### 核心模块 (Core Modules)
```
lightx2v/disagg/
├── __init__.py                          # 包初始化
├── conn.py                              # 数据连接和管理器
├── mooncake.py                          # Mooncake传输引擎集成
├── protocol.py                          # 通信协议定义
├── utils.py                             # 工具函数
├── services/
│   ├── encoder.py                       # 编码器服务实现
│   └── transformer.py                   # Transformer服务实现
└── examples/
    ├── mooncake_client.py               # Mooncake客户端示例
    ├── mooncake_server.py               # Mooncake服务端示例
    ├── wan_i2v.py                       # WAN I2V单机示例
    ├── wan_i2v_service.py               # WAN I2V服务化示例
    ├── wan_t2v.py                       # WAN T2V单机示例
    └── wan_t2v_service.py               # WAN T2V服务化示例
```

## 🔧 技术实现 (Technical Implementation)

### 分布式模式 (Disaggregation Modes)
```python
class DisaggregationMode(Enum):
    NULL = "null"              # 未启用分布式
    ENCODE = "encode"          # 编码器模式
    TRANSFORMER = "transformer" # Transformer模式
```

### 数据流 (Data Flow)
1. **编码阶段**: 编码器服务处理输入（文本/图像），生成潜在特征
2. **数据传输**: 通过高性能传输引擎发送数据到Transformer服务
3. **推理阶段**: Transformer服务执行去噪推理
4. **结果返回**: 解码后的视频数据返回到客户端

### 性能优化 (Performance Optimizations)
- 连续张量分组传输，减少通信开销
- 支持CPU offload和块级别offload
- 异步数据传输，提高吞吐量
- 支持FP8量化，降低传输数据量

## 📝 使用示例 (Usage Examples)

### 启动编码器服务 (Start Encoder Service)
```python
from lightx2v.disagg.services.encoder import EncoderService

encoder_service = EncoderService(
    model_path=model_path,
    task="i2v",
    data_bootstrap_addr="127.0.0.1",
    data_bootstrap_room=0
)
encoder_service.start()
```

### 启动Transformer服务 (Start Transformer Service)
```python
from lightx2v.disagg.services.transformer import TransformerService

transformer_service = TransformerService(
    model_path=model_path,
    task="i2v",
    data_bootstrap_addr="127.0.0.1",
    data_bootstrap_room=0
)
transformer_service.start()
```

### 运行推理 (Run Inference)
详细示例请参考 `lightx2v/disagg/examples/` 目录下的示例代码。

## 🎯 适用场景 (Use Cases)

1. **资源受限环境**: 在单GPU内存不足时，将编码和推理分离到不同设备
2. **多机部署**: 在多个服务器之间分配计算负载
3. **边缘计算**: 将轻量级编码放在边缘，重量级推理放在云端
4. **服务化部署**: 构建微服务架构的视频生成系统

## ✅ 测试和验证 (Testing & Validation)

- ✅ 测试了WAN I2V和T2V模型的分布式推理
- ✅ 验证了跨设备通信的稳定性
- ✅ 测试了不同硬件加速器的兼容性
- ✅ 确认了性能和准确性与单机模式一致

## 📚 文档更新 (Documentation Updates)

示例代码和使用说明已添加到 `lightx2v/disagg/examples/` 目录。

## 🔄 向后兼容性 (Backward Compatibility)

本功能为**可选特性**，不影响现有代码：
- 默认模式（DisaggregationMode.NULL）保持原有行为
- 现有API和接口保持不变
- 用户可以选择性地启用分布式功能

## 🚧 已知限制 (Known Limitations)

1. 需要网络连接稳定性保证
2. 跨网络通信会引入轻微延迟
3. 需要手动配置服务地址和端口

## 🔮 未来计划 (Future Work)

- [ ] 添加自动服务发现和负载均衡
- [ ] 支持更多模型架构
- [ ] 优化跨网络传输性能
- [ ] 添加服务健康检查和故障恢复

## 👥 贡献者 (Contributors)

- @zhtshr

## 📄 License

Apache License 2.0

---

## 审查清单 (Review Checklist)

- [x] 代码遵循项目编码规范
- [x] 添加了必要的文档和示例
- [x] 功能经过测试验证
- [x] 保持向后兼容性
- [x] 无安全漏洞引入
