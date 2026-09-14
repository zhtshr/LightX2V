#!/usr/bin/env python3
"""Real dist_infer multi-GPU stage latency (not rank0-only Exp-3b).

Matches LightX2V monolithic SP serving behavior:
  - T5 encoder: every rank runs full infer (redundant, no seq_p)
  - VAE enc/dec: vae_parallel=True by default when parallel is a dict
  - DiT denoise: Ulysses seq_p

Reports per-stage wall latency as max(rank elapsed) after barrier+cuda sync.
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
from lightx2v.disagg.utils import (
    load_wan_text_encoder,
    load_wan_transformer,
    load_wan_vae_decoder,
    load_wan_vae_encoder,
    read_image_input,
    set_config,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def _log(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _max_elapsed(local_s: float, device: torch.device) -> float:
    """Pipeline-visible stage latency = slowest rank."""
    if not dist.is_initialized():
        return local_s
    t = torch.tensor([local_s], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _init_distributed(seq_p_size: int, config: dict[str, Any], vae_parallel: bool) -> None:
    if seq_p_size <= 1:
        config["parallel"] = False
        return
    config["parallel"] = {
        "seq_p_size": seq_p_size,
        "seq_p_attn_type": "ulysses",
        "vae_parallel": bool(vae_parallel),
    }
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _time_stage(fn, device: torch.device) -> tuple[Any, float]:
    _barrier()
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    local = time.perf_counter() - t0
    _barrier()
    return out, _max_elapsed(local, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_baseline.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    parser.add_argument(
        "--prompt",
        default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_p_size", type=int, required=True)
    parser.add_argument("--vae_parallel", type=int, default=1, help="1=True (real default), 0=False")
    parser.add_argument("--baseline_json", default="")
    parser.add_argument(
        "--output_json",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_real_sp_stage_latency_seqp1.json",
    )
    args = parser.parse_args()

    config = set_config(
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=args.config_json,
    )
    seed_all(args.seed)
    vae_parallel = bool(args.vae_parallel)
    _init_distributed(args.seq_p_size, config, vae_parallel=vae_parallel)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"{AI_DEVICE}:{rank}" if dist.is_initialized() else AI_DEVICE)
    text_len = int(config.get("text_len", 512))

    _log(
        f"[real-sp] P={args.seq_p_size} world={world} vae_parallel={vae_parallel} "
        f"seq_parallel={config.get('seq_parallel')} cpu_offload={config.get('cpu_offload')}"
    )

    t_load0 = time.perf_counter()
    text_encoder = load_wan_text_encoder(config)[0]
    vae_encoder = load_wan_vae_encoder(config)
    vae_decoder = load_wan_vae_decoder(config)
    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)
    load_s = time.perf_counter() - t_load0
    _log(f"[real-sp] model load {load_s:.1f}s (excluded)")

    img, _ = read_image_input(args.image_path)
    latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)

    # Warmup lightly
    def _warmup() -> None:
        _ = text_encoder.infer([args.prompt])
        _ = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)

    _barrier()
    _warmup()
    _barrier()

    # --- text encoder: ALL ranks (real dist_infer) ---
    def _run_text() -> torch.Tensor:
        context = text_encoder.infer([args.prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
        return context.to(device=device, dtype=GET_DTYPE())

    context, text_s = _time_stage(_run_text, device)
    _log(f"  text_encoder: {text_s:.3f}s")

    # --- VAE encode: all ranks, vae_parallel if enabled ---
    def _run_vae_enc() -> torch.Tensor:
        vae_out = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
        return vae_out.to(device=device, dtype=GET_DTYPE())

    vae_out, vae_enc_s = _time_stage(_run_vae_enc, device)
    _log(f"  vae_encoder: {vae_enc_s:.3f}s")

    enc_s = text_s + vae_enc_s
    inputs = {
        "text_encoder_output": {"context": context, "context_null": None},
        "image_encoder_output": {"clip_encoder_out": None, "vae_encoder_out": vae_out},
    }

    # --- DiT denoise ---
    def _run_denoise() -> torch.Tensor:
        scheduler.prepare(
            seed=args.seed,
            latent_shape=latent_shape,
            image_encoder_output=inputs["image_encoder_output"],
        )
        for step_index in range(scheduler.infer_steps):
            scheduler.step_pre(step_index=step_index)
            model.infer(inputs)
            scheduler.step_post()
        return scheduler.latents

    # one-step warmup for small P
    if args.seq_p_size <= 4:
        scheduler.prepare(
            seed=args.seed,
            latent_shape=latent_shape,
            image_encoder_output=inputs["image_encoder_output"],
        )
        scheduler.step_pre(step_index=0)
        model.infer(inputs)
        scheduler.step_post()
        _barrier()

    latents, denoise_s = _time_stage(_run_denoise, device)
    _log(f"  dit_denoise: {denoise_s:.3f}s ({denoise_s / max(int(config.get('infer_steps', 1)), 1):.3f}s/step)")

    # --- VAE decode ---
    def _run_vae_dec() -> Any:
        gen = vae_decoder.decode(latents.to(GET_DTYPE()))
        # rank0-only comfy convert to avoid extra host work skew; still decode on all ranks
        if is_main_process():
            _ = wan_vae_to_comfy(gen)
        return gen

    _, vae_dec_s = _time_stage(_run_vae_dec, device)
    _log(f"  vae_decoder: {vae_dec_s:.3f}s")

    e2e = enc_s + denoise_s + vae_dec_s
    stages = {
        "text_encoder": text_s,
        "vae_encoder": vae_enc_s,
        "encoder_total": enc_s,
        "denoise": denoise_s,
        "decoder": vae_dec_s,
        "e2e_sum": e2e,
    }

    eff: dict[str, Any] = {}
    if args.baseline_json and Path(args.baseline_json).is_file():
        base = json.loads(Path(args.baseline_json).read_text())
        b = base["stages_s"]
        p = float(args.seq_p_size)

        def _eta(key: str) -> float | None:
            if key not in b or key not in stages or stages[key] <= 0:
                return None
            return b[key] / (stages[key] * p)

        eff = {
            "text_encoder": _eta("text_encoder"),
            "vae_encoder": _eta("vae_encoder"),
            "encoder_total": _eta("encoder_total"),
            "denoise": _eta("denoise"),
            "decoder": _eta("decoder"),
            "e2e_sum": _eta("e2e_sum"),
        }

    result = {
        "metric": "real_dist_infer_sp_stage_latency",
        "description": (
            "Real monolithic SP: all ranks run T5; VAE uses vae_parallel; DiT Ulysses. "
            "Latency = max(rank elapsed) after barrier+cuda sync."
        ),
        "seq_p_size": args.seq_p_size,
        "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "vae_parallel": vae_parallel,
        "cpu_offload": config.get("cpu_offload"),
        "offload_granularity": config.get("offload_granularity"),
        "infer_steps": config.get("infer_steps"),
        "target_hw": [config.get("target_height"), config.get("target_width")],
        "model_load_s_excluded": load_s,
        "stages_s": stages,
        "stage_fraction": {k: (v / e2e if e2e > 0 else None) for k, v in stages.items() if k != "e2e_sum"},
        "parallel_efficiency_vs_p1": eff,
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        _log(f"[real-sp] wrote {out}")
        _log(json.dumps(stages, indent=2))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
