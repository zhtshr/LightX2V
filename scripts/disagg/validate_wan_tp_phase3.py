#!/usr/bin/env python3
"""TP-3: MoE I2V load without OOM (memory smoke for no-offload TP)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lightx2v.disagg.utils import load_wan_transformer
from scripts.disagg.validate_wan_tp_common import init_distributed, init_tp_config, is_main, sync_cuda


def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/dist_infer/wan22_moe_i2v_tensorp.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--model_type", default="wan2.2_moe_high_noise")
    parser.add_argument("--tensor_p_size", type=int, default=0)
    parser.add_argument("--max_mem_gb", type=float, default=24.0)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        if is_main():
            print(f"TP-3 skipped: model_path not found: {args.model_path}")
        return 0

    config = init_tp_config(
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        tensor_p_size=args.tensor_p_size,
    )
    init_distributed(config)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_wan_transformer(config, model_type=args.model_type)[0]
    sync_cuda()
    peak_gb = _gpu_mem_gb()

    if is_main():
        print(f"TP-3 peak GPU memory: {peak_gb:.2f} GB (limit {args.max_mem_gb} GB)")
        print(f"TP-3 use_tp={getattr(model, 'use_tp', False)} tp_size={getattr(model, 'tp_size', 1)}")
    assert peak_gb <= args.max_mem_gb, f"OOM risk: peak {peak_gb:.2f}GB > {args.max_mem_gb}GB"
    if is_main():
        print("TP-3 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
