#!/usr/bin/env python3
"""Enc/Dec boundness vs hardware: SM% & MEM% across batch sizes (no DiT)."""

from __future__ import annotations

import argparse
import json
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
    load_wan_vae_decoder,
    load_wan_vae_encoder,
    read_image_input,
    set_config,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _query(gpu: int) -> tuple[float, float]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"-i={gpu}",
            "--query-gpu=utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    a, b = [x.strip() for x in out.split(",")]
    return float(a), float(b)


class Sampler:
    def __init__(self, gpu: int, interval: float = 0.08):
        self.gpu = gpu
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []

    def start(self) -> None:
        self.stop()
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> list[tuple[float, float]]:
        if self._t is None:
            return list(self.samples)
        self._stop.set()
        self._t.join(timeout=3)
        self._t = None
        return list(self.samples)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(_query(self.gpu))
            except Exception:
                pass
            self._stop.wait(self.interval)


def _summ(samples: list[tuple[float, float]], wall: float, n: int) -> dict[str, Any]:
    core = [(s, m) for s, m in samples if s >= 5 or m >= 5] or list(samples)
    sms = [x[0] for x in core]
    mems = [x[1] for x in core]
    mean_sm = statistics.fmean(sms) if sms else 0.0
    mean_mem = statistics.fmean(mems) if mems else 0.0
    ratio = mean_mem / mean_sm if mean_sm > 1e-6 else float("inf")
    # vs hardware: mem-bound if DRAM ctrl near-saturated while SM not fully ahead
    if mean_mem >= 75 and ratio >= 0.85:
        vs_hw = "memory_bound"
    elif mean_sm >= 85 and ratio <= 0.75:
        vs_hw = "compute_bound"
    elif mean_sm >= 70 and mean_mem >= 55:
        vs_hw = "mixed"
    else:
        vs_hw = "inconclusive"
    return {
        "batch": n,
        "wall_s": wall,
        "ms_per_sample": wall / max(n, 1) * 1000.0,
        "sm_mean": mean_sm,
        "mem_mean": mean_mem,
        "mem_over_sm": ratio,
        "vs_hardware": vs_hw,
        "n_samples": len(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_baseline.json")
    ap.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    ap.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    ap.add_argument("--prompt", default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.")
    ap.add_argument("--physical_gpu", type=int, default=0)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--output_json",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_encdec_boundness_vs_hw.json",
    )
    ap.add_argument(
        "--output_md",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_encdec_boundness_vs_hw.md",
    )
    args = ap.parse_args()
    if args.physical_gpu in (1, 3):
        raise SystemExit(f"refuse GPU {args.physical_gpu}")

    cfg = set_config(model_path=args.model_path, task="i2v", model_cls="wan2.2_moe", config_path=args.config_json)
    cfg["parallel"] = False
    seed_all(42)

    text_encoder = load_wan_text_encoder(cfg)[0]
    vae_encoder = load_wan_vae_encoder(cfg)
    vae_decoder = load_wan_vae_decoder(cfg)
    text_len = int(cfg.get("text_len", 512))
    img, _ = read_image_input(args.image_path)
    latent_shape, lh, lw = compute_latent_shape_from_image(cfg, img)
    # representative latents for decode
    latents = torch.randn(tuple(latent_shape), device=torch.device(AI_DEVICE), dtype=GET_DTYPE())

    sampler = Sampler(args.physical_gpu)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows_enc: list[dict[str, Any]] = []
    rows_dec: list[dict[str, Any]] = []

    # warmup
    _ = text_encoder.infer([args.prompt])
    _ = get_vae_encoder_output(vae_encoder, cfg, img, lh, lw)
    _ = vae_decoder.decode(latents)
    _sync()

    for b in batches:
        prompts = [args.prompt] * b
        # encoder: 1 text batch + b vae encodes (matches phase2 style)
        for _ in range(1):
            pass
        sampler.start()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            ctx = text_encoder.infer(prompts)
            _ = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in ctx])
            for _i in range(b):
                get_vae_encoder_output(vae_encoder, cfg, img, lh, lw)
        _sync()
        wall = time.perf_counter() - t0
        s = _summ(sampler.stop(), wall, b * args.repeats)
        s["batch_size"] = b
        s["stage"] = "encoder"
        rows_enc.append(s)
        print(f"enc B={b}: SM={s['sm_mean']:.1f} MEM={s['mem_mean']:.1f} ratio={s['mem_over_sm']:.2f} → {s['vs_hardware']} ({s['ms_per_sample']:.1f} ms/sample)")

        sampler.start()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            for _i in range(b):
                gv = vae_decoder.decode(latents)
                _ = wan_vae_to_comfy(gv)
        _sync()
        wall = time.perf_counter() - t0
        s = _summ(sampler.stop(), wall, b * args.repeats)
        s["batch_size"] = b
        s["stage"] = "decoder"
        rows_dec.append(s)
        print(f"dec B={b}: SM={s['sm_mean']:.1f} MEM={s['mem_mean']:.1f} ratio={s['mem_over_sm']:.2f} → {s['vs_hardware']} ({s['ms_per_sample']:.1f} ms/sample)")

    def _conclude(rows: list[dict[str, Any]], name: str) -> str:
        # Prefer B=1 hardware reading; also note if larger B pushes toward memory
        b1 = next(r for r in rows if r["batch_size"] == 1)
        bmax = rows[-1]
        if b1["vs_hardware"] == "compute_bound" and bmax["mem_mean"] > b1["mem_mean"] + 10:
            return (
                f"{name}: at B=1 **compute-bound vs A10** (SM saturated, DRAM ctrl not); "
                f"larger B raises MEM ({b1['mem_mean']:.0f}%→{bmax['mem_mean']:.0f}%) — more bandwidth-sensitive under batching"
            )
        if b1["vs_hardware"] == "memory_bound":
            return f"{name}: **memory-bound vs A10** (DRAM ctrl high)"
        if b1["vs_hardware"] == "compute_bound":
            return f"{name}: **compute-bound vs A10** at serving B=1 (SM≈{b1['sm_mean']:.0f}%, MEM≈{b1['mem_mean']:.0f}%)"
        return f"{name}: {b1['vs_hardware']} (SM={b1['sm_mean']:.0f} MEM={b1['mem_mean']:.0f})"

    out = {
        "hardware": "NVIDIA A10 24GB (~600 GB/s HBM)",
        "metric": "nvidia-smi utilization.gpu (SM) vs utilization.memory (DRAM controller)",
        "note": (
            "vs_hardware uses util proxies, not full roofline FLOPs/byte. "
            "A10 ridge point is high; many kernels can be AI-memory-bound while still showing high SM%."
        ),
        "encoder": rows_enc,
        "decoder": rows_dec,
        "conclusion_encoder": _conclude(rows_enc, "Encoder"),
        "conclusion_decoder": _conclude(rows_dec, "Decoder"),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    def _tbl(rows: list[dict[str, Any]]) -> str:
        lines = ["| B | SM% | MEM% | MEM/SM | ms/sample | vs hardware |", "|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            lines.append(
                f"| {r['batch_size']} | {r['sm_mean']:.1f} | {r['mem_mean']:.1f} | {r['mem_over_sm']:.2f} | "
                f"{r['ms_per_sample']:.1f} | {r['vs_hardware']} |"
            )
        return "\n".join(lines)

    md = f"""# Enc/Dec vs A10 hardware: compute- or memory-bound?

## Encoder
{_tbl(rows_enc)}

{out['conclusion_encoder']}

## Decoder
{_tbl(rows_dec)}

{out['conclusion_decoder']}

## How to read
- **Compute-bound vs hardware**: SM util near ceiling, DRAM-controller util clearly lower.
- **Memory-bound vs hardware**: DRAM-controller util high (≥~75%) and MEM/SM high.
- Relative to denoise (prior run MEM/SM≈0.39), enc/dec are still **more memory-pressure heavy**, but at B=1 on A10 they typically **hit the SM wall first**.
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
