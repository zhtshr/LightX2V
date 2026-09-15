#!/usr/bin/env python3
"""Profile BAGEL T2I three phases: encode(prefill), transformer(flow), decode(vae)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from argparse import Namespace
from pathlib import Path

import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.models.runners.bagel.bagel_runner import BagelRunner
from lightx2v.models.runners.bagel.t2i_utils import get_bagel_latent_downsample
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import seed_all


def gb(x: float) -> float:
    return round(x / 1024**3, 3)


def file_size_gb(path: Path) -> float:
    return gb(path.stat().st_size) if path.exists() else 0.0


def cuda_mem() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    return {
        "allocated_gb": gb(torch.cuda.memory_allocated()),
        "reserved_gb": gb(torch.cuda.memory_reserved()),
        "max_allocated_gb": gb(torch.cuda.max_memory_allocated()),
    }


def reset_cuda_mem_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def weight_bytes_on_device(module) -> int:
    total = 0
    if hasattr(module, "state_dict"):
        for t in module.state_dict().values():
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
    if hasattr(module, "_modules"):
        for child in module._modules.values():
            if child is not None and hasattr(child, "state_dict"):
                for t in child.state_dict().values():
                    if torch.is_tensor(t):
                        total += t.numel() * t.element_size()
    return total


def estimate_kv_cache_gb(past_key_values, num_layers: int, hidden_size: int, num_kv_heads: int, head_dim: int, seq_len: int) -> float:
    """Rough KV size from cached tensors if present."""
    if past_key_values is None:
        return 0.0
    total = 0
    for i in range(num_layers):
        k = past_key_values.key_cache[i]
        v = past_key_values.value_cache[i]
        if k is not None:
            total += k.numel() * k.element_size()
        if v is not None:
            total += v.numel() * v.element_size()
    if total == 0 and seq_len > 0:
        # fallback estimate: 2 * layers * seq * kv_heads * head_dim * 2 bytes
        total = 2 * num_layers * seq_len * num_kv_heads * head_dim * 2
    return gb(total)


def build_runner(model_path: str, config_json: str, cpu_offload: bool, infer_steps: int) -> BagelRunner:
    args = Namespace(
        model_cls="bagel",
        task="t2i",
        model_path=model_path,
        config_json=config_json,
        seed=42,
        support_tasks=[],
        parallel=False,
        num_iterations=None,
        cpu_offload=cpu_offload,
        offload_granularity="block",
    )
    config = set_config(args)
    config["infer_steps"] = infer_steps
    config["inference_hyper"] = dict(config["inference_hyper"])
    config["inference_hyper"]["num_timesteps"] = infer_steps
    runner = BagelRunner(config)
    return runner


def profile_t2i(model_path: str, config_json: str, cpu_offload: bool, infer_steps: int, aspect_ratio: str) -> dict:
    seed_all(42)
    runner = build_runner(model_path, config_json, cpu_offload, infer_steps)

    model_dir = Path(model_path)
    weight_files = {
        "ema_safetensors_gb": file_size_gb(model_dir / "ema.safetensors"),
        "ae_safetensors_gb": file_size_gb(model_dir / "ae.safetensors"),
    }

    # --- load weights ---
    reset_cuda_mem_stats()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    runner.init_modules()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    load_mem = cuda_mem()

    model = runner.model
    llm = model.llm_config
    num_layers = llm["num_hidden_layers"]
    hidden_size = llm["hidden_size"]
    num_kv_heads = llm["num_key_value_heads"]
    head_dim = hidden_size // llm["num_attention_heads"]

    gpu_weight_breakdown = {
        "pre_post_on_gpu_gb": gb(weight_bytes_on_device(model.pre_weight) + weight_bytes_on_device(model.post_weight)),
        "transformer_blocks_on_gpu_gb": gb(weight_bytes_on_device(model.transformer_weights)),
        "vae_on_gpu_gb": gb(sum(p.numel() * p.element_size() for p in runner.vae_decoder.vae_model.parameters())),
    }

    input_info = init_empty_input_info("t2i")
    update_input_info_from_dict(
        input_info,
        {
            "prompt": "A small cabin beside a lake at sunrise, cinematic lighting",
            "aspect_ratio": aspect_ratio,
            "save_result_path": "save_results/bagel_profile_t2i.png",
            "seed": 42,
        },
    )
    runner.input_info = input_info
    runner._refresh_scheduler_from_config()
    image_shape = runner.set_t2i_image_shapes()
    h, w = image_shape
    down = get_bagel_latent_downsample(runner.config)
    num_image_tokens = (h // down) * (w // down)

    # --- phase 1: encode / prefill ---
    reset_cuda_mem_stats()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    runner.inputs, runner.scheduler = model.prepare_inputs(input_info, runner.scheduler, vae_model=None)
    torch.cuda.synchronize()
    encode_s = time.perf_counter() - t1
    encode_mem = cuda_mem()

    kv_seq = int(runner.inputs.gen_context["kv_lens"][0])
    kv_gb = estimate_kv_cache_gb(runner.inputs.gen_context["past_key_values"], num_layers, hidden_size, num_kv_heads, head_dim, kv_seq)

    # --- phase 2: transformer / flow denoise ---
    total_steps = runner.model.scheduler.infer_steps - 1
    step_times = []
    reset_cuda_mem_stats()
    t2 = time.perf_counter()
    torch.cuda.synchronize()
    for step_index in range(total_steps):
        st = time.perf_counter()
        torch.cuda.synchronize()
        runner.model.scheduler.step_pre(step_index=step_index)
        runner.model.infer(runner.inputs)
        runner.model.scheduler.step_post()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - st)
    transformer_s = time.perf_counter() - t2
    transformer_mem = cuda_mem()

    latents = runner.model.scheduler.latents
    generator = runner.model.scheduler.generator

    # --- phase 3: decode ---
    decode_info = {
        "packed_seqlens": runner.inputs.generation_input["packed_seqlens"],
        "image_shape": image_shape,
        "latent_downsample": down,
        "latent_channel": runner.config["vae_config"]["z_channels"],
        "latent_patch_size": runner.config["latent_patch_size"],
    }
    reset_cuda_mem_stats()
    t3 = time.perf_counter()
    torch.cuda.synchronize()
    images = runner.run_vae_decoder(latents, decode_info)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t3
    decode_mem = cuda_mem()

    # roofline hints
    # A10 ~31 TFLOPS bf16, ~600 GB/s HBM; PCIe gen4 x16 ~25 GB/s effective with offload
    a10_tflops = 31.0
    a10_bw_gbs = 600.0
    pcie_gbs = 25.0

    # rough FLOPs per denoise step: ~2 * params * tokens (matmul dominated)
    active_params_b = 7e9  # 7B active MoT
    tokens_per_step = num_image_tokens + 2 + kv_seq
    flops_per_step = 2 * active_params_b * tokens_per_step
    compute_time_floor_s = flops_per_step / (a10_tflops * 1e12)
    mem_time_offload_s = weight_files["ema_safetensors_gb"] / pcie_gbs * num_layers / num_layers  # one full model read per step approx

    def bound_label(measured_s: float, phase: str) -> str:
        if cpu_offload and phase in ("encode", "transformer"):
            return "memory-bound (CPU-GPU weight streaming + KV/activation traffic)"
        if phase == "decode":
            return "memory-bound (VAE conv bandwidth-dominated)"
        if measured_s > 0 and compute_time_floor_s / measured_s < 0.3:
            return "memory-bound"
        return "compute-bound (attention+MLP on long image token seq)"

    result = {
        "task": "t2i",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_offload": cpu_offload,
        "resolution": list(image_shape),
        "num_image_tokens": num_image_tokens,
        "kv_seq_len": kv_seq,
        "infer_steps": infer_steps,
        "denoise_steps": total_steps,
        "weight_files_gb": weight_files,
        "weight_files_total_gb": round(weight_files["ema_safetensors_gb"] + weight_files["ae_safetensors_gb"], 3),
        "gpu_resident_weights_after_load_gb": gpu_weight_breakdown,
        "kv_cache_est_gb": kv_gb,
        "phases": {
            "load": {"latency_s": round(load_s, 3), **{f"mem_{k}": v for k, v in load_mem.items()}},
            "encode_prefill": {
                "latency_s": round(encode_s, 3),
                "bound": bound_label(encode_s, "encode"),
                **{f"mem_{k}": v for k, v in encode_mem.items()},
            },
            "transformer_flow": {
                "latency_s": round(transformer_s, 3),
                "latency_per_step_s": round(transformer_s / max(total_steps, 1), 3),
                "step_times_s": {
                    "first": round(step_times[0], 3) if step_times else None,
                    "median": round(sorted(step_times)[len(step_times) // 2], 3) if step_times else None,
                    "last": round(step_times[-1], 3) if step_times else None,
                },
                "bound": bound_label(transformer_s / max(total_steps, 1), "transformer"),
                "rough_compute_floor_per_step_s": round(compute_time_floor_s, 4),
                "rough_pcie_weight_read_per_step_s": round(mem_time_offload_s, 3) if cpu_offload else 0,
                **{f"mem_{k}": v for k, v in transformer_mem.items()},
            },
            "decode_vae": {
                "latency_s": round(decode_s, 3),
                "bound": bound_label(decode_s, "decode"),
                **{f"mem_{k}": v for k, v in decode_mem.items()},
            },
        },
        "total_pipeline_s": round(encode_s + transformer_s + decode_s, 3),
        "phase_latency_pct": {
            "encode": round(100 * encode_s / max(encode_s + transformer_s + decode_s, 1e-6), 1),
            "transformer": round(100 * transformer_s / max(encode_s + transformer_s + decode_s, 1e-6), 1),
            "decode": round(100 * decode_s / max(encode_s + transformer_s + decode_s, 1e-6), 1),
        },
        "output_saved": Path("save_results/bagel_profile_t2i.png").exists(),
    }

    del images, latents, generator, runner
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/ByteDance-Seed/BAGEL-7B-MoT")
    parser.add_argument("--config_json", default="configs/bagel/bagel_t2i.json")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--aspect_ratio", default="1:1")
    parser.add_argument("--cpu_offload", action="store_true", default=True)
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--out", default="save_results/bagel_phase_profile_t2i.json")
    args = parser.parse_args()

    cpu_offload = args.cpu_offload and not args.no_cpu_offload
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    result = profile_t2i(args.model_path, args.config_json, cpu_offload, args.infer_steps, args.aspect_ratio)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
