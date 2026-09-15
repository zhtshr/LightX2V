#!/usr/bin/env python3
"""Phase 1: transformer denoise compute micro-benchmark (seq parallel sweep).

Measures only scheduler.prepare + denoise loop (aligned with disagg
transformer_compute_delay_s). Excludes model load, encoder, decoder, and E2E.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.examples.wan_i2v import compute_latent_shape_from_image, get_vae_encoder_output
from lightx2v.disagg.utils import load_wan_text_encoder, load_wan_transformer, load_wan_vae_encoder, read_image_input, set_config
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def _run_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


def _move_tensor_tree(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_tensor_tree(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tensor_tree(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(value, device) for value in obj)
    return obj


def _prepare_payload_on_device(payload: dict[str, Any]) -> dict[str, Any]:
    device = _run_device()
    moved = _move_tensor_tree(payload, device)
    inputs = moved["inputs"]
    return {
        "seed": moved["seed"],
        "latent_shape": moved["latent_shape"],
        "image_encoder_output": moved["image_encoder_output"],
        "inputs": inputs,
    }


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task=getattr(args, "task", "i2v"),
        model_cls=getattr(args, "model_cls", "wan2.2_moe"),
        config_path=args.config_json,
    )
    cli_tensor_p = int(getattr(args, "tensor_p_size", 0) or 0)
    cfg_path = Path(args.config_json)
    cfg_parallel = {}
    if cfg_path.is_file():
        cfg_parallel = json.loads(cfg_path.read_text(encoding="utf-8")).get("parallel", {}) or {}

    if cli_tensor_p > 0:
        if cli_tensor_p > 1:
            config["parallel"] = {"tensor_p_size": cli_tensor_p}
        else:
            config["parallel"] = False
    elif args.seq_p_size > 1:
        attn_type = cfg_parallel.get("seq_p_attn_type", "ulysses")
        config["parallel"] = {"seq_p_size": args.seq_p_size, "seq_p_attn_type": attn_type}
    elif cfg_parallel.get("tensor_p_size", 1) > 1:
        config["parallel"] = {"tensor_p_size": int(cfg_parallel["tensor_p_size"])}
    elif int(cfg_parallel.get("pipe_p_size", 0) or 0) > 1:
        config["parallel"] = dict(cfg_parallel)
    elif cfg_parallel.get("seq_p_size", 1) > 1:
        config["parallel"] = dict(cfg_parallel)
    else:
        config["parallel"] = False
    return config


def _init_distributed(config: dict[str, Any]) -> None:
    if not config.get("parallel"):
        return
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _latent_shape_from_config(config: dict[str, Any]) -> list[int]:
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def _prepare_inputs_cache(
    config: dict[str, Any],
    cache_path: Path,
    prompt: str,
    image_path: str,
    seed: int,
    force: bool = False,
    *,
    task: str = "i2v",
    negative_prompt: str = "",
) -> dict[str, Any]:
    if cache_path.is_file() and not force:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if is_main_process():
            print(f"Loaded encoder inputs cache: {cache_path}")
        return payload

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload

    print("Preparing encoder inputs (excluded from transformer bench timing)...")
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))

    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    if config.get("enable_cfg", False) and negative_prompt:
        context_null = text_encoder.infer([negative_prompt])
        context_null = torch.stack(
            [torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context_null],
        )
    else:
        context_null = None
    text_encoder_output = {"context": context, "context_null": context_null}

    if task == "t2v":
        latent_shape = _latent_shape_from_config(config)
        image_encoder_output = None
        inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }
        del text_encoder
    else:
        vae_encoder = load_wan_vae_encoder(config)
        img, _ = read_image_input(image_path)
        latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)
        vae_encoder_out = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
        image_encoder_output = {"clip_encoder_out": None, "vae_encoder_out": vae_encoder_out}
        inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }
        del text_encoder, vae_encoder

    payload = {
        "seed": seed,
        "latent_shape": latent_shape,
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": image_encoder_output,
        "inputs": inputs,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"Wrote encoder inputs cache: {cache_path}")

    _sync_device()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
    return payload


def _run_transformer_compute(
    scheduler: WanScheduler,
    model: Any,
    payload: dict[str, Any],
) -> None:
    seed = int(payload["seed"])
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]

    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    infer_steps = scheduler.infer_steps
    for step_index in range(infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()
    _sync_device()


def _bench_transformer_once(scheduler: WanScheduler, model: Any, payload: dict[str, Any]) -> float:
    _sync_device()
    start = time.perf_counter()
    _run_transformer_compute(scheduler, model, payload)
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_baseline.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v", choices=("i2v", "t2v"))
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    parser.add_argument(
        "--negative_prompt",
        default="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )
    parser.add_argument(
        "--prompt",
        default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_p_size", type=int, default=1)
    parser.add_argument(
        "--tensor_p_size",
        type=int,
        default=0,
        help="Tensor parallel size; 0 = use seq_p_size or config parallel block",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--refresh_inputs_cache", action="store_true")
    parser.add_argument(
        "--encoder_only",
        action="store_true",
        help="Only build encoder inputs cache (excluded from transformer timing); skip denoise bench",
    )
    args = parser.parse_args()

    config = _load_config(args)
    seed_all(args.seed)
    _init_distributed(config)

    payload = _prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=args.prompt,
        image_path=args.image_path,
        seed=args.seed,
        force=args.refresh_inputs_cache,
        task=args.task,
        negative_prompt=args.negative_prompt,
    )
    if args.encoder_only:
        if is_main_process():
            print(f"Encoder-only mode; cache ready at {args.inputs_cache}")
        return 0

    payload = _prepare_payload_on_device(payload)

    load_start = time.perf_counter()
    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)
    model_load_s = time.perf_counter() - load_start
    if is_main_process():
        print(f"Transformer load time (excluded): {model_load_s:.3f}s")

    for _ in range(args.warmup):
        _bench_transformer_once(scheduler, model, payload)

    measure_samples: list[float] = []
    for _ in range(args.measure_iters):
        measure_samples.append(_bench_transformer_once(scheduler, model, payload))

    avg_transformer_compute_s = None
    if measure_samples:
        avg_transformer_compute_s = sum(measure_samples) / len(measure_samples)
    parallel_mode = "none"
    if int(args.tensor_p_size or 0) > 1 or (
        config.get("parallel") and config["parallel"].get("tensor_p_size", 1) > 1
    ):
        parallel_mode = "tensor_p"
    elif args.seq_p_size > 1 or (
        config.get("parallel") and config["parallel"].get("seq_p_size", 1) > 1
    ):
        parallel_mode = "seq_p"
    elif config.get("pipeline_parallel") or (
        config.get("parallel") and int(config["parallel"].get("pipe_p_size", 0) or 0) > 1
    ):
        parallel_mode = "pipe_p"

    cfg_par = config.get("parallel") if isinstance(config.get("parallel"), dict) else {}
    result = {
        "metric": "transformer_compute_s",
        "description": "scheduler.prepare + denoise loop only; excludes model load and encoder/decoder",
        "parallel_mode": parallel_mode,
        "seq_p_size": args.seq_p_size if parallel_mode == "seq_p" else 1,
        "pipe_p_size": int(config.get("pp_size", 0) or cfg_par.get("pipe_p_size", 1)),
        "tensor_p_size": int(args.tensor_p_size or 0) or (
            cfg_par.get("tensor_p_size", 1) if parallel_mode == "tensor_p" else 1
        ),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "model_load_s_excluded": model_load_s,
        "warmup_iters": args.warmup,
        "measure_iters": args.measure_iters,
        "transformer_compute_samples_s": measure_samples,
        "transformer_compute_s": avg_transformer_compute_s,
        "config_json": args.config_json,
    }

    if is_main_process() and args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if parallel_mode == "tensor_p":
            p_label = f"tensor_p={result['tensor_p_size']}"
        elif parallel_mode == "seq_p":
            p_label = f"seq_p={args.seq_p_size}"
        elif parallel_mode == "pipe_p":
            p_label = f"pipe_p={result.get('pipe_p_size', 2)}"
        else:
            p_label = "P=1"
        print(
            f"{p_label}: transformer_compute_s={avg_transformer_compute_s} "
            f"(samples={measure_samples})"
        )
        print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
