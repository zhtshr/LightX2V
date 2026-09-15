#!/usr/bin/env python3
"""Compare one-step noise_pred: TP norm NCCL vs P2P (max abs diff)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _one_step(model, scheduler, payload, step_index=0):
    scheduler.prepare(
        seed=int(payload["seed"]),
        latent_shape=payload["latent_shape"],
        image_encoder_output=payload["image_encoder_output"],
    )
    scheduler.step_pre(step_index=step_index)
    model.set_scheduler(scheduler)
    model.infer(payload["inputs"])
    torch.cuda.synchronize()
    scheduler.step_post()
    return scheduler.noise_pred.detach().clone()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--inputs_cache", required=True)
    args = parser.parse_args()

    p1 = _load_phase1()
    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        seed=42,
        seq_p_size=1,
        tensor_p_size=2,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="镜头晃动",
    )
    config = p1._load_config(ns)
    seed_all(42)
    p1._init_distributed(config)
    payload = p1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=ns.prompt,
        image_path=ns.image_path,
        seed=42,
        force=False,
        task="i2v",
        negative_prompt=ns.negative_prompt,
    )
    payload = p1._prepare_payload_on_device(payload)

    config_nccl = dict(config)
    config_nccl["tp_norm_p2p"] = False
    model = load_wan_transformer(config_nccl)
    sched = WanScheduler(config_nccl)
    model.set_scheduler(sched)
    ref = _one_step(model, sched, payload)
    del model
    torch.cuda.empty_cache()

    config_p2p = dict(config)
    config_p2p["tp_norm_p2p"] = True
    model2 = load_wan_transformer(config_p2p)
    sched2 = WanScheduler(config_p2p)
    model2.set_scheduler(sched2)
    out = _one_step(model2, sched2, payload)

    if is_main_process():
        diff = (ref.float() - out.float()).abs()
        print(
            f"max_abs_diff={diff.max().item():.6e} "
            f"mean_abs_diff={diff.mean().item():.6e} "
            f"rel_max={ (diff.max() / ref.float().abs().max()).item():.6e}"
        )

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
