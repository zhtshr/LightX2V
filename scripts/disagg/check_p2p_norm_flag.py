#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.utils.utils import seed_all


def _load_p1():
    spec = importlib.util.spec_from_file_location(
        "p1", Path(__file__).parent / "run_phase1_transformer_bench.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p1 = _load_p1()
    ns = argparse.Namespace(
        config_json="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json",
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v", model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="t", seed=42, seq_p_size=1, tensor_p_size=2,
        inputs_cache="save_results/optimization_study/phase1_encoder_inputs_256x256.pt",
        refresh_inputs_cache=False, negative_prompt="",
    )
    config = p1._load_config(ns)
    config["tp_norm_p2p"] = True
    seed_all(42)
    p1._init_distributed(config)
    model = load_wan_transformer(config)
    wan = model.model[0] if hasattr(model, "model") else model
    ph = wan.transformer_weights.blocks[10].compute_phases[0]
    print(
        f"rank={dist.get_rank()} tp_norm_p2p={config.get('tp_norm_p2p')} "
        f"norm_q.use_p2p_norm={ph.self_attn_norm_q.use_p2p_norm} "
        f"norm_k.use_p2p_norm={ph.self_attn_norm_k.use_p2p_norm}",
        flush=True,
    )
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
