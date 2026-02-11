# Enable Disaggregation Feature

## Summary

This PR introduces a **disaggregation architecture** to LightX2V, enabling distributed deployment of the video generation pipeline across multiple devices or machines.

## What's New

### Core Functionality
- **Service Decoupling**: Separate encoder and transformer services that can run independently
- **High-Performance Communication**: ZeroMQ and HTTP-based messaging with Mooncake transfer engine
- **Flexible Deployment**: Support for single-machine multi-GPU and cross-machine distributed setups

### New Components
- `lightx2v/disagg/`: Complete disaggregation package
  - `conn.py`: Data connection and management
  - `services/encoder.py`: Encoder service implementation
  - `services/transformer.py`: Transformer service implementation
  - `examples/`: Usage examples for WAN I2V and T2V models

## Key Benefits

1. **Resource Flexibility**: Distribute compute-intensive tasks across multiple devices
2. **Scalability**: Easy horizontal scaling for production deployments
3. **Memory Efficiency**: Run large models on hardware-constrained environments
4. **Service-Oriented**: Build microservice-based video generation systems

## Usage Example

```python
# Start encoder service
from lightx2v.disagg.services.encoder import EncoderService
encoder = EncoderService(model_path=path, data_bootstrap_addr="127.0.0.1")
encoder.start()

# Start transformer service (can be on different machine/GPU)
from lightx2v.disagg.services.transformer import TransformerService
transformer = TransformerService(model_path=path, data_bootstrap_addr="127.0.0.1")
transformer.start()
```

See `lightx2v/disagg/examples/` for complete working examples.

## Backward Compatibility

✅ This is an **optional feature** that doesn't affect existing functionality:
- Default mode preserves current behavior
- All existing APIs remain unchanged
- Users can opt-in to use disaggregation when needed

## Testing

- ✅ Tested with WAN I2V and T2V models
- ✅ Verified cross-device communication stability
- ✅ Validated accuracy matches single-machine mode
- ✅ Tested on multiple hardware accelerators (GPU, GCU, MUSA)

## Files Changed

- Added: `lightx2v/disagg/` package with all disaggregation modules
- Modified: None (purely additive)

## Future Enhancements

- Automatic service discovery
- Load balancing across multiple workers
- Enhanced monitoring and health checks

---

**Type**: Feature  
**Breaking Changes**: None  
**Documentation**: Included in `lightx2v/disagg/examples/`
