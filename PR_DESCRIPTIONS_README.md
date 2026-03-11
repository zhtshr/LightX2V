# Pull Request Description Files

This directory contains pull request descriptions for the "Enable Disaggregation" feature.

## Files

1. **PULL_REQUEST_DESCRIPTION.md** (完整版 / Full Version)
   - Comprehensive bilingual (Chinese/English) PR description
   - Includes detailed technical documentation
   - Best for internal documentation and detailed review

2. **PR_DESCRIPTION_EN.md** (English Version)
   - Concise English-only description
   - Suitable for GitHub PR body
   - Focused on key features and changes

3. **PR_DESCRIPTION_ZH.md** (中文版 / Chinese Version)
   - Concise Chinese-only description
   - 适合GitHub PR描述
   - 专注于核心特性和变更

## How to Use

### For GitHub Pull Request

Copy the content from either `PR_DESCRIPTION_EN.md` or `PR_DESCRIPTION_ZH.md` and paste it as your PR description on GitHub.

**English PR:**
```bash
cat PR_DESCRIPTION_EN.md
```

**中文PR:**
```bash
cat PR_DESCRIPTION_ZH.md
```

### For Documentation

Use `PULL_REQUEST_DESCRIPTION.md` for detailed project documentation or team reviews.

## Feature Summary

The **Disaggregation** feature enables distributed deployment of LightX2V's video generation pipeline:

- 🚀 Split encoder and transformer into separate services
- 🌐 Support cross-machine distributed deployment  
- ⚡ High-performance ZeroMQ and Mooncake data transfer
- 🔧 Flexible resource allocation and scaling
- ✅ Backward compatible (optional feature)

## Quick Start

```python
# Encoder Service
from lightx2v.disagg.services.encoder import EncoderService
encoder = EncoderService(model_path=path, data_bootstrap_addr="127.0.0.1")
encoder.start()

# Transformer Service  
from lightx2v.disagg.services.transformer import TransformerService
transformer = TransformerService(model_path=path, data_bootstrap_addr="127.0.0.1")
transformer.start()
```

See `lightx2v/disagg/examples/` for complete examples.

---

**Generated on:** 2026-02-11  
**Feature:** Disaggregation Architecture  
**Status:** Ready for Review
