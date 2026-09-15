#!/usr/bin/env python3
"""TP-2: full Wan transformer load + one denoise step smoke / numerical check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import seed_all
from scripts.disagg.validate_wan_tp_common import init_distributed, init_tp_config, is_main, local_device, sync_cuda


def _latent_shape(config: dict) -> list[int]:
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def _dummy_inputs(config: dict, device: torch.device) -> dict:
    text_len = int(config.get("text_len", 512))
    dim = int(config["dim"])
    context = torch.randn(1, text_len, dim, device=device, dtype=torch.bfloat16)
    text_encoder_output = {"context": context, "context_null": None}
    if config["task"] == "t2v":
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }
    latent_shape = _latent_shape(config)
    _, t, h, w = latent_shape
    vae_out = torch.randn(1, t, h, w, config.get("num_channels_latents", 16), device=device, dtype=torch.bfloat16)
    return {
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": {"clip_encoder_out": None, "vae_encoder_out": vae_out},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/dist_infer/wan_t2v_tensorp.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.1-T2V-1.3B")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--model_cls", default="wan2.1")
    parser.add_argument("--model_type", default="wan2.1")
    parser.add_argument("--tensor_p_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_load", action="store_true", help="Skip model load (mesh-only smoke)")
    args = parser.parse_args()

    if args.skip_load:
        config = init_tp_config(
            config_json=args.config_json,
            model_path=args.model_path,
            task=args.task,
            model_cls=args.model_cls,
            tensor_p_size=args.tensor_p_size,
        )
        init_distributed(config)
        sync_cuda()
        if is_main():
            print("TP-2 mesh smoke passed (skip_load).")
        return 0

    if not Path(args.model_path).exists():
        if is_main():
            print(f"TP-2 skipped: model_path not found: {args.model_path}")
        return 0

    config = init_tp_config(
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        tensor_p_size=args.tensor_p_size,
    )
    seed_all(args.seed)
    init_distributed(config)
    device = local_device()

    model = load_wan_transformer(config, model_type=args.model_type)[0]
    scheduler = WanScheduler(config)
    latent_shape = _latent_shape(config)
    inputs = _dummy_inputs(config, device)

    scheduler.prepare(seed=args.seed, latent_shape=latent_shape, image_encoder_output=inputs.get("image_encoder_output"))
    scheduler.step_pre(step_index=0)
    noise_pred = model.infer(inputs)
    scheduler.step_post()
    sync_cuda()

    assert torch.isfinite(noise_pred).all(), "noise_pred contains non-finite values"
    if is_main():
        print(f"TP-2 OK: noise_pred shape={tuple(noise_pred.shape)} dtype={noise_pred.dtype}")
        print("TP-2 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
