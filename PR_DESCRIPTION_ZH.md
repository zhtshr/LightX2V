# 启用分布式计算（Disaggregation）功能

## 概述

本PR为LightX2V引入了**分布式计算架构**，支持将视频生成流程部署到多个设备或多台机器上。

## 新增功能

### 核心特性
- **服务解耦**: 独立的编码器和Transformer服务，可独立运行
- **高性能通信**: 基于ZeroMQ和HTTP的消息传递，集成Mooncake传输引擎
- **灵活部署**: 支持单机多GPU和跨机器分布式部署

### 新增组件
- `lightx2v/disagg/`: 完整的分布式计算包
  - `conn.py`: 数据连接和管理
  - `services/encoder.py`: 编码器服务实现
  - `services/transformer.py`: Transformer服务实现
  - `examples/`: WAN I2V和T2V模型的使用示例

## 主要优势

1. **资源灵活性**: 将计算密集型任务分布到多个设备
2. **可扩展性**: 便于生产环境的水平扩展
3. **内存效率**: 在硬件受限的环境中运行大模型
4. **服务化**: 构建基于微服务的视频生成系统

## 使用示例

```python
# 启动编码器服务
from lightx2v.disagg.services.encoder import EncoderService
encoder = EncoderService(model_path=path, data_bootstrap_addr="127.0.0.1")
encoder.start()

# 启动transformer服务（可以在不同机器/GPU上）
from lightx2v.disagg.services.transformer import TransformerService
transformer = TransformerService(model_path=path, data_bootstrap_addr="127.0.0.1")
transformer.start()
```

完整的工作示例请参考 `lightx2v/disagg/examples/` 目录。

## 向后兼容性

✅ 这是一个**可选功能**，不影响现有功能：
- 默认模式保持当前行为
- 所有现有API保持不变
- 用户可以选择性地使用分布式功能

## 测试

- ✅ 已测试WAN I2V和T2V模型
- ✅ 已验证跨设备通信稳定性
- ✅ 已验证准确性与单机模式一致
- ✅ 已在多种硬件加速器上测试（GPU、GCU、MUSA）

## 文件变更

- 新增: `lightx2v/disagg/` 包及所有分布式模块
- 修改: 无（纯新增功能）

## 未来增强

- 自动服务发现
- 多worker负载均衡
- 增强的监控和健康检查

---

**类型**: 新功能  
**破坏性变更**: 无  
**文档**: 包含在 `lightx2v/disagg/examples/` 中
