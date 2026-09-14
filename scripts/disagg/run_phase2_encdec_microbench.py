#!/usr/bin/env python3
"""Phase 2: encoder / VAE decode batch micro-benchmark (single GPU)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from lightx2v.disagg.examples.wan_i2v import compute_latent_shape_from_image, get_vae_encoder_output
from lightx2v.disagg.utils import (
    load_wan_text_encoder,
    load_wan_vae_decoder,
    load_wan_vae_encoder,
    read_image_input,
    set_config,
)
from lightx2v.utils.utils import wan_vae_to_comfy
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


def _bench_encoder(
    text_encoder,
    vae_encoder,
    config,
    img: torch.Tensor,
    latent_h: int,
    latent_w: int,
    prompt: str,
    text_len: int,
    batch_size: int,
    warmup: int,
) -> tuple[float, float, float]:
    prompts = [prompt] * batch_size

    def _run_once() -> tuple[float, float, float]:
        torch.cuda.synchronize()
        text_start = time.perf_counter()
        context = text_encoder.infer(prompts)
        torch.cuda.synchronize()
        text_s = time.perf_counter() - text_start
        _ = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])

        torch.cuda.synchronize()
        vae_start = time.perf_counter()
        for _ in range(batch_size):
            get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
        torch.cuda.synchronize()
        vae_s = time.perf_counter() - vae_start
        return text_s, vae_s

    for _ in range(warmup):
        _run_once()

    text_s, vae_s = _run_once()
    return text_s + vae_s, text_s, vae_s


def _bench_decode(vae_decoder, latents: torch.Tensor, batch_size: int, warmup: int) -> float:
    sample = latents.to(GET_DTYPE())
    for _ in range(warmup):
        for _ in range(batch_size):
            _ = vae_decoder.decode(sample)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(batch_size):
        gen_video = vae_decoder.decode(sample)
        _ = wan_vae_to_comfy(gen_video)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/single_node/wan22_i2v_distill_controller.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    parser.add_argument("--prompt", default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.")
    parser.add_argument("--batch_sizes", default="1,2,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output_dir", default="/root/zht/LightX2V/save_results/optimization_study")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = set_config(
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=args.config_json,
    )
    text_len = int(config.get("text_len", 512))

    print("Loading encoder/decoder models (one-time)...")
    load_start = time.perf_counter()
    text_encoder = load_wan_text_encoder(config)[0]
    vae_encoder = load_wan_vae_encoder(config)
    vae_decoder = load_wan_vae_decoder(config)
    load_end = time.perf_counter()
    print(f"Model load time (excluded from bench): {load_end - load_start:.3f}s")

    img, _ = read_image_input(args.image_path)
    latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)
    # Representative denoised latents for decode throughput (shape matches scheduler.latents).
    latents = torch.randn(tuple(latent_shape), device=torch.device(AI_DEVICE), dtype=GET_DTYPE())

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    rows: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        enc_total_s, enc_text_s, enc_vae_s = _bench_encoder(
            text_encoder, vae_encoder, config, img, latent_h, latent_w, args.prompt, text_len, batch_size, args.warmup
        )
        dec_s = _bench_decode(vae_decoder, latents, batch_size, args.warmup)
        rows.append(
            {
                "batch_size": batch_size,
                "encoder_total_s": enc_total_s,
                "encoder_text_s": enc_text_s,
                "encoder_vae_s": enc_vae_s,
                "encoder_ms_per_sample": enc_total_s / batch_size * 1000.0,
                "decoder_total_s": dec_s,
                "decoder_ms_per_sample": dec_s / batch_size * 1000.0,
            }
        )
        print(
            f"B={batch_size}: encoder={enc_total_s:.3f}s ({enc_total_s/batch_size*1000:.1f} ms/sample, "
            f"text={enc_text_s:.3f}s vae={enc_vae_s:.3f}s), "
            f"decoder={dec_s:.3f}s ({dec_s/batch_size*1000:.1f} ms/sample)"
        )

    csv_path = out_dir / "phase2_encdec_microbench.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_load_s_excluded": load_end - load_start,
        "rows": rows,
    }
    json_path = out_dir / "phase2_encdec_microbench.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Amdahl summary vs Phase 0 anchor (B=1 disagg N=1).
    t_enc_b1 = next((r["encoder_total_s"] for r in rows if r["batch_size"] == 1), None)
    t_dec_b1 = next((r["decoder_total_s"] for r in rows if r["batch_size"] == 1), None)
    t_trans = 74.5
    t_e2e = 91.0
    lines = [
        "# Phase 2 — Encode/Decode Batch Microbench (Measured)",
        "",
        "Single GPU; model load excluded. Encoder = T5 batch infer + VAE encode × B (sequential).",
        "",
        "| B | encoder_total_s | enc ms/sample | decoder_total_s | dec ms/sample |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['encoder_total_s']:.3f} | {row['encoder_ms_per_sample']:.1f} | "
            f"{row['decoder_total_s']:.3f} | {row['decoder_ms_per_sample']:.1f} |"
        )
    if t_enc_b1 and t_dec_b1:
        lines.extend(["", "## Amdahl upper bound vs Phase 0 E2E (~91s)", ""])
        lines.append("| B | enc E2E saving | dec E2E saving | combined |")
        lines.append("|---:|---:|---:|---:|")
        for row in rows:
            b = int(row["batch_size"])
            enc_save = t_enc_b1 * (1 - 1 / b) / t_e2e * 100
            dec_save = t_dec_b1 * (1 - 1 / b) / t_e2e * 100
            lines.append(f"| {b} | {enc_save:.1f}% | {dec_save:.1f}% | {enc_save + dec_save:.1f}% |")
    md_path = out_dir / "phase2_encdec_microbench.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}, {json_path}, {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
