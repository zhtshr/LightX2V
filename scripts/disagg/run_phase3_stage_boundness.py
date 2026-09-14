#!/usr/bin/env python3
"""Characterize stage boundness: SM util vs memory-controller util.

Hypothesis for disagg:
  - denoise (DiT): compute-bound → high SM%, lower relative mem%
  - encoder / decoder: memory-bound → high mem% relative to SM%

Samples nvidia-smi utilization.gpu (SM proxy) and utilization.memory
(memory controller) at high frequency during each stage on a single GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

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
from lightx2v.utils.utils import seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _query_sm_mem(physical_gpu: int) -> tuple[float, float]:
    cmd = [
        "nvidia-smi",
        f"-i={physical_gpu}",
        "--query-gpu=utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    text = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    sm_s, mem_s = [x.strip() for x in text.split(",")]
    return float(sm_s), float(mem_s)


class SmMemSampler:
    def __init__(self, physical_gpu: int, interval_s: float = 0.1):
        self.physical_gpu = int(physical_gpu)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float, float]] = []  # ts, sm, mem

    def start(self) -> None:
        self.stop()
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float, float]]:
        if self._thread is None:
            return list(self.samples)
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        return list(self.samples)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sm, mem = _query_sm_mem(self.physical_gpu)
                self.samples.append((time.time(), sm, mem))
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def _summarize(samples: list[tuple[float, float, float]], wall_s: float) -> dict[str, Any]:
    if not samples:
        return {"n_samples": 0, "wall_s": wall_s}
    sms = [s[1] for s in samples]
    mems = [s[2] for s in samples]
    # Drop leading/trailing near-idle samples (<5% both) to reduce stage-edge bias
    core = [(sm, mem) for sm, mem in zip(sms, mems) if sm >= 5.0 or mem >= 5.0]
    if len(core) < max(3, len(samples) // 10):
        core = list(zip(sms, mems))
    c_sm = [x[0] for x in core]
    c_mem = [x[1] for x in core]
    mean_sm = statistics.fmean(c_sm)
    mean_mem = statistics.fmean(c_mem)
    # Boundness score: mem/sm — >1 suggests memory-controller pressure dominates SM
    ratio = mean_mem / mean_sm if mean_sm > 1e-6 else float("inf")
    if mean_sm >= 70 and ratio <= 0.85:
        verdict = "compute_bound"
    elif mean_mem >= 40 and ratio >= 0.90:
        verdict = "memory_bound"
    elif mean_sm >= 50 and mean_mem >= 50:
        verdict = "mixed_high_util"
    else:
        verdict = "inconclusive"
    return {
        "wall_s": wall_s,
        "n_samples": len(samples),
        "n_core_samples": len(core),
        "sm_util_pct": {
            "mean": mean_sm,
            "p50": statistics.median(c_sm),
            "p90": sorted(c_sm)[int(0.9 * (len(c_sm) - 1))],
            "max": max(c_sm),
        },
        "mem_util_pct": {
            "mean": mean_mem,
            "p50": statistics.median(c_mem),
            "p90": sorted(c_mem)[int(0.9 * (len(c_mem) - 1))],
            "max": max(c_mem),
        },
        "mem_over_sm_ratio": ratio,
        "verdict": verdict,
    }


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
    parser.add_argument("--physical_gpu", type=int, default=0)
    parser.add_argument("--sample_interval_s", type=float, default=0.1)
    parser.add_argument("--encoder_repeats", type=int, default=4, help="Repeat encoder to collect enough samples")
    parser.add_argument("--decoder_repeats", type=int, default=3)
    parser.add_argument(
        "--output_json",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_stage_boundness.json",
    )
    parser.add_argument(
        "--output_md",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_stage_boundness.md",
    )
    args = parser.parse_args()

    if args.physical_gpu in (1, 3):
        raise RuntimeError(f"Refusing unreliable GPU {args.physical_gpu}")

    os_environ_note = f"CUDA_VISIBLE_DEVICES should map logical cuda:0 → physical {args.physical_gpu}"
    config = set_config(
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=args.config_json,
    )
    config["parallel"] = False
    seed_all(args.seed)

    print(f"[boundness] {os_environ_note}")
    print(
        f"[boundness] cpu_offload={config.get('cpu_offload')} "
        f"t5_offload={config.get('t5_cpu_offload')} vae_offload={config.get('vae_cpu_offload')}"
    )

    text_encoder = load_wan_text_encoder(config)[0]
    vae_encoder = load_wan_vae_encoder(config)
    vae_decoder = load_wan_vae_decoder(config)
    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)

    sampler = SmMemSampler(args.physical_gpu, interval_s=args.sample_interval_s)
    text_len = int(config.get("text_len", 512))
    img, _ = read_image_input(args.image_path)
    latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)

    results: dict[str, Any] = {
        "metric": "stage_boundness_sm_vs_mem",
        "physical_gpu": args.physical_gpu,
        "config": {
            "cpu_offload": config.get("cpu_offload"),
            "offload_granularity": config.get("offload_granularity"),
            "t5_cpu_offload": config.get("t5_cpu_offload"),
            "vae_cpu_offload": config.get("vae_cpu_offload"),
            "target_hw": [config.get("target_height"), config.get("target_width")],
            "infer_steps": config.get("infer_steps"),
        },
        "method": (
            "nvidia-smi utilization.gpu (SM) vs utilization.memory (DRAM controller). "
            "verdict: compute_bound if SM high & mem/sm<=0.85; memory_bound if mem high & mem/sm>=0.90."
        ),
        "stages": {},
    }

    # --- encoder ---
    print("[boundness] measuring encoder...")
    # warmup once
    _ = text_encoder.infer([args.prompt])
    _ = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
    _sync()
    sampler.start()
    t0 = time.perf_counter()
    context = None
    vae_out = None
    for _ in range(args.encoder_repeats):
        context = text_encoder.infer([args.prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
        vae_out = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
    _sync()
    enc_wall = time.perf_counter() - t0
    enc_samples = sampler.stop()
    results["stages"]["encoder"] = _summarize(enc_samples, enc_wall)
    results["stages"]["encoder"]["repeats"] = args.encoder_repeats
    results["stages"]["encoder"]["per_repeat_s"] = enc_wall / max(args.encoder_repeats, 1)
    print(
        f"  encoder: SM={results['stages']['encoder']['sm_util_pct']['mean']:.1f}% "
        f"MEM={results['stages']['encoder']['mem_util_pct']['mean']:.1f}% "
        f"ratio={results['stages']['encoder']['mem_over_sm_ratio']:.2f} "
        f"→ {results['stages']['encoder']['verdict']}"
    )

    assert context is not None and vae_out is not None
    context = context.to(dtype=GET_DTYPE())
    vae_out = vae_out.to(dtype=GET_DTYPE())
    inputs = {
        "text_encoder_output": {"context": context, "context_null": None},
        "image_encoder_output": {"clip_encoder_out": None, "vae_encoder_out": vae_out},
    }

    # --- denoise ---
    print("[boundness] measuring denoise (includes block offload if enabled)...")
    # short warmup step path via prepare only
    scheduler.prepare(seed=args.seed, latent_shape=latent_shape, image_encoder_output=inputs["image_encoder_output"])
    _sync()
    sampler.start()
    t0 = time.perf_counter()
    scheduler.prepare(seed=args.seed, latent_shape=latent_shape, image_encoder_output=inputs["image_encoder_output"])
    for step_index in range(scheduler.infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()
    _sync()
    den_wall = time.perf_counter() - t0
    den_samples = sampler.stop()
    results["stages"]["denoise"] = _summarize(den_samples, den_wall)
    if config.get("cpu_offload"):
        results["stages"]["denoise"]["note"] = (
            "block offload streams weights over PCIe/HBM; elevates mem util beyond pure GEMM traffic"
        )
    print(
        f"  denoise: SM={results['stages']['denoise']['sm_util_pct']['mean']:.1f}% "
        f"MEM={results['stages']['denoise']['mem_util_pct']['mean']:.1f}% "
        f"ratio={results['stages']['denoise']['mem_over_sm_ratio']:.2f} "
        f"→ {results['stages']['denoise']['verdict']}"
    )

    latents = scheduler.latents

    # --- decoder ---
    print("[boundness] measuring decoder...")
    _ = vae_decoder.decode(latents.to(GET_DTYPE()))
    _sync()
    sampler.start()
    t0 = time.perf_counter()
    for _ in range(args.decoder_repeats):
        gen_video = vae_decoder.decode(latents.to(GET_DTYPE()))
        _ = wan_vae_to_comfy(gen_video)
    _sync()
    dec_wall = time.perf_counter() - t0
    dec_samples = sampler.stop()
    results["stages"]["decoder"] = _summarize(dec_samples, dec_wall)
    results["stages"]["decoder"]["repeats"] = args.decoder_repeats
    results["stages"]["decoder"]["per_repeat_s"] = dec_wall / max(args.decoder_repeats, 1)
    print(
        f"  decoder: SM={results['stages']['decoder']['sm_util_pct']['mean']:.1f}% "
        f"MEM={results['stages']['decoder']['mem_util_pct']['mean']:.1f}% "
        f"ratio={results['stages']['decoder']['mem_over_sm_ratio']:.2f} "
        f"→ {results['stages']['decoder']['verdict']}"
    )

    # Cross-stage contrast
    enc = results["stages"]["encoder"]
    den = results["stages"]["denoise"]
    dec = results["stages"]["decoder"]
    results["contrast"] = {
        "denoise_sm_minus_encoder_sm": den["sm_util_pct"]["mean"] - enc["sm_util_pct"]["mean"],
        "encoder_mem_sm_ratio_vs_denoise": enc["mem_over_sm_ratio"] / max(den["mem_over_sm_ratio"], 1e-6),
        "decoder_mem_sm_ratio_vs_denoise": dec["mem_over_sm_ratio"] / max(den["mem_over_sm_ratio"], 1e-6),
        "hypothesis_supported": (
            den["verdict"] in ("compute_bound", "mixed_high_util")
            and enc["mem_over_sm_ratio"] > den["mem_over_sm_ratio"]
            and dec["mem_over_sm_ratio"] > den["mem_over_sm_ratio"]
        ),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    md = f"""# Stage boundness: SM vs memory controller util

GPU physical `{args.physical_gpu}` (A10). Metric: `utilization.gpu` (SM%) vs `utilization.memory` (DRAM ctrl%).

| Stage | wall (s) | SM mean% | MEM mean% | MEM/SM | verdict |
|---|---:|---:|---:|---:|---|
| encoder | {enc['wall_s']:.2f} | {enc['sm_util_pct']['mean']:.1f} | {enc['mem_util_pct']['mean']:.1f} | {enc['mem_over_sm_ratio']:.2f} | **{enc['verdict']}** |
| denoise | {den['wall_s']:.2f} | {den['sm_util_pct']['mean']:.1f} | {den['mem_util_pct']['mean']:.1f} | {den['mem_over_sm_ratio']:.2f} | **{den['verdict']}** |
| decoder | {dec['wall_s']:.2f} | {dec['sm_util_pct']['mean']:.1f} | {dec['mem_util_pct']['mean']:.1f} | {dec['mem_over_sm_ratio']:.2f} | **{dec['verdict']}** |

- cpu_offload={config.get('cpu_offload')}, t5_offload={config.get('t5_cpu_offload')}, vae_offload={config.get('vae_cpu_offload')}
- hypothesis_supported={results['contrast']['hypothesis_supported']}
- encoder MEM/SM relative to denoise: **{results['contrast']['encoder_mem_sm_ratio_vs_denoise']:.2f}×**
- decoder MEM/SM relative to denoise: **{results['contrast']['decoder_mem_sm_ratio_vs_denoise']:.2f}×**

## Reading

- **Higher MEM/SM** ⇒ more memory-controller pressure per unit SM (memory-leaning).
- **Denoise** with block offload also streams weights (raises MEM%); even then, compare ratios to enc/dec.
- For disagg: enc/dec favor **batching** (amortize bandwidth); denoise favors **compute parallel (SP)**.
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(f"[boundness] wrote {out}")
    print(f"[boundness] wrote {args.output_md}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
