#!/usr/bin/env python3
"""Paper-ready arithmetic intensity (FLOPs/byte) for Encoder / Decoder / DiT.

Workload: Wan2.2-MoE I2V distill, 480x832x81, single GPU (A10).
Denoise step counts: 1, 4, 8, 50.

Notes for paper:
  - Enc/Dec FLOPs & bytes do NOT depend on denoise steps.
  - DiT FLOPs & HBM bytes both scale ~linearly with steps → per-step AI ≈ constant.
  - What changes with steps is FLOPs/time share and E2E effective AI.
"""

from __future__ import annotations

import argparse
import json
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


# ---------- hardware ridge (A10) ----------
A10_BF16_TFLOPS = 125.0
A10_INT8_TOPS = 250.0
A10_BW_GBS = 600.0
RIDGE_BF16 = A10_BF16_TFLOPS * 1e12 / (A10_BW_GBS * 1e9)  # ~208.3 FLOP/byte
RIDGE_INT8 = A10_INT8_TOPS * 1e12 / (A10_BW_GBS * 1e9)  # ~416.7 Op/byte


def _gemm_flops(m: float, n: float, k: float) -> float:
    return 2.0 * m * n * k


def _gemm_bytes(m: float, n: float, k: float, w_b: float, a_b: float = 2.0) -> float:
    # X[m,k], W[k,n], Y[m,n] — each touched once (lower-bound traffic)
    return m * k * a_b + k * n * w_b + m * n * a_b


def _conv3d_flops(cin: float, cout: float, vol: float, k: int = 3) -> float:
    return 2.0 * cin * cout * (k**3) * vol


def _conv3d_bytes(cin: float, cout: float, vol: float, k: int = 3, w_b: float = 2.0, a_b: float = 2.0) -> float:
    return cin * cout * (k**3) * w_b + (cin + cout) * vol * a_b


def _verdict(ai: float, ridge: float = RIDGE_BF16) -> str:
    return "compute_bound" if ai >= ridge else "memory_bound"


def estimate_t5_encoder(S: int = 512, D: int = 4096, FFN: int = 10240, L: int = 24, w_b: float = 2.0, a_b: float = 2.0) -> dict[str, Any]:
    """UMT5-XXL text encoder (gated-GELU FFN)."""
    flops = 0.0
    bytes_ = 0.0
    for _ in range(L):
        # QKVO
        for _op in range(4):
            flops += _gemm_flops(S, D, D)
            bytes_ += _gemm_bytes(S, D, D, w_b, a_b)
        # flash-attn style: 4*S*S*D FLOPs; score matrix stays on-chip → act traffic ~ QKV+O
        flops += 4.0 * S * S * D
        bytes_ += (3 * S * D + S * D) * a_b
        # gated FFN: wi0, wi1, wo
        flops += _gemm_flops(S, FFN, D) + _gemm_flops(S, FFN, D) + _gemm_flops(S, D, FFN)
        bytes_ += _gemm_bytes(S, FFN, D, w_b, a_b) + _gemm_bytes(S, FFN, D, w_b, a_b) + _gemm_bytes(S, D, FFN, w_b, a_b)
    ai = flops / bytes_
    return {
        "name": "T5/UMT5 encoder",
        "flops": flops,
        "bytes": bytes_,
        "ai_flop_per_byte": ai,
        "tflop": flops / 1e12,
        "gbyte": bytes_ / 1e9,
        "vs_a10_ridge_bf16": ai / RIDGE_BF16,
        "verdict": _verdict(ai),
    }


def estimate_vae_tower(
    *,
    T: int = 81,
    H: int = 480,
    W: int = 832,
    dim: int = 96,
    z_dim: int = 16,
    encode: bool = True,
    w_b: float = 2.0,
    a_b: float = 2.0,
) -> dict[str, Any]:
    """Approximate Wan VAE enc/dec FLOPs+bytes via resolution pyramid.

    Encoder: dim_mult=[1,2,4,4], 2 ResBlocks/level, spatial /8, temporal ~ /4.
    Decoder: mirror upsample path (same order of magnitude).
    """
    flops = 0.0
    bytes_ = 0.0
    # track dominant-op AI samples (activation-amortized)
    op_ais: list[float] = []

    def add_conv(cin: int, cout: int, t: int, h: int, w: int) -> None:
        nonlocal flops, bytes_
        vol = float(t * h * w)
        f = _conv3d_flops(cin, cout, vol)
        b = _conv3d_bytes(cin, cout, vol, w_b=w_b, a_b=a_b)
        flops += f
        bytes_ += b
        op_ais.append(f / b)

    if encode:
        t, h, w = T, H, W
        add_conv(3, dim, t, h, w)
        cin = dim
        # levels: keep / downsample
        levels = [dim, dim * 2, dim * 4, dim * 4]  # after first block channels
        temperal_down = [True, True, False]
        for i, cout in enumerate(levels):
            for _ in range(2):  # 2 resblocks × 2 convs (approx as cin→cout then cout→cout)
                add_conv(cin, cout, t, h, w)
                add_conv(cout, cout, t, h, w)
                cin = cout
            if i < 3:
                h //= 2
                w //= 2
                if i < len(temperal_down) and temperal_down[i]:
                    t = (t + 1) // 2
                add_conv(cin, cin, t, h, w)  # resample
        # mid + head
        for _ in range(2):
            add_conv(cin, cin, t, h, w)
            add_conv(cin, cin, t, h, w)
        add_conv(cin, z_dim * 2, t, h, w)
        name = "VAE encoder"
    else:
        # decoder starts from latent ~ (21, 60, 104), z_dim → dim*4
        t, h, w = (T - 1) // 4 + 1, H // 8, W // 8
        cin = z_dim
        add_conv(cin, dim * 4, t, h, w)
        cin = dim * 4
        levels = [dim * 4, dim * 4, dim * 2, dim]  # upsample toward RGB
        temperal_up = [False, True, True]
        for i, cout in enumerate(levels):
            for _ in range(2):
                add_conv(cin, cout, t, h, w)
                add_conv(cout, cout, t, h, w)
                cin = cout
            if i < 3:
                h *= 2
                w *= 2
                if i < len(temperal_up) and temperal_up[i]:
                    t = t * 2 - 1  # rough inverse of causal temporal down
                add_conv(cin, cin, t, h, w)
        for _ in range(2):
            add_conv(cin, cin, t, h, w)
            add_conv(cin, cin, t, h, w)
        add_conv(cin, 3, t, h, w)
        name = "VAE decoder"

    ai = flops / bytes_
    return {
        "name": name,
        "flops": flops,
        "bytes": bytes_,
        "ai_flop_per_byte": ai,
        "tflop": flops / 1e12,
        "gbyte": bytes_ / 1e9,
        "vs_a10_ridge_bf16": ai / RIDGE_BF16,
        "verdict": _verdict(ai),
        "median_op_ai": float(sorted(op_ais)[len(op_ais) // 2]) if op_ais else None,
        "min_op_ai": float(min(op_ais)) if op_ais else None,
        "max_op_ai": float(max(op_ais)) if op_ais else None,
    }


def estimate_dit_one_step(
    *,
    S: int = 32760,
    D: int = 5120,
    FFN: int = 13824,
    L: int = 40,
    TEXT: int = 512,
    w_b: float = 1.0,  # int8
    a_b: float = 2.0,  # bf16 acts
) -> dict[str, Any]:
    """One Wan DiT forward (one MoE expert active)."""
    flops = 0.0
    bytes_ = 0.0
    for _ in range(L):
        # self QKVO
        for _op in range(4):
            flops += _gemm_flops(S, D, D)
            bytes_ += _gemm_bytes(S, D, D, w_b, a_b)
        flops += 4.0 * S * S * D
        bytes_ += (3 * S * D + S * D) * a_b
        # cross: Q,O on seq; K,V on text
        for m, n, k in [(S, D, D), (S, D, D), (TEXT, D, D), (TEXT, D, D)]:
            flops += _gemm_flops(m, n, k)
            bytes_ += _gemm_bytes(m, n, k, w_b, a_b)
        flops += 4.0 * S * TEXT * D
        bytes_ += (S * D + TEXT * D + S * D) * a_b  # Q, K/V-ish, O (lower bound)
        # FFN
        flops += _gemm_flops(S, FFN, D) + _gemm_flops(S, D, FFN)
        bytes_ += _gemm_bytes(S, FFN, D, w_b, a_b) + _gemm_bytes(S, D, FFN, w_b, a_b)
    # head proj approx
    flops += _gemm_flops(S, D, D)
    bytes_ += _gemm_bytes(S, D, D, w_b, a_b)

    ai = flops / bytes_
    return {
        "name": "DiT one step",
        "flops": flops,
        "bytes": bytes_,
        "ai_flop_per_byte": ai,
        "tflop": flops / 1e12,
        "gbyte": bytes_ / 1e9,
        "vs_a10_ridge_bf16": ai / RIDGE_BF16,
        "vs_a10_ridge_int8": ai / RIDGE_INT8,
        "verdict": _verdict(ai, RIDGE_BF16),
    }


def build_theory(steps_list: list[int]) -> dict[str, Any]:
    t5 = estimate_t5_encoder()
    vae_e = estimate_vae_tower(encode=True)
    vae_d = estimate_vae_tower(encode=False)
    dit1 = estimate_dit_one_step()

    encoder_flops = t5["flops"] + vae_e["flops"]
    encoder_bytes = t5["bytes"] + vae_e["bytes"]
    encoder_ai = encoder_flops / encoder_bytes

    decoder_flops = vae_d["flops"]
    decoder_bytes = vae_d["bytes"]
    decoder_ai = decoder_flops / decoder_bytes

    by_steps: dict[str, Any] = {}
    for n in steps_list:
        dit_f = dit1["flops"] * n
        dit_b = dit1["bytes"] * n
        dit_ai = dit_f / dit_b  # == dit1 AI
        total_f = encoder_flops + decoder_flops + dit_f
        total_b = encoder_bytes + decoder_bytes + dit_b
        by_steps[str(n)] = {
            "steps": n,
            "encoder": {
                "tflop": encoder_flops / 1e12,
                "gbyte": encoder_bytes / 1e9,
                "ai_flop_per_byte": encoder_ai,
                "flops_share": encoder_flops / total_f,
                "verdict": _verdict(encoder_ai),
            },
            "decoder": {
                "tflop": decoder_flops / 1e12,
                "gbyte": decoder_bytes / 1e9,
                "ai_flop_per_byte": decoder_ai,
                "flops_share": decoder_flops / total_f,
                "verdict": _verdict(decoder_ai),
            },
            "dit": {
                "tflop": dit_f / 1e12,
                "gbyte": dit_b / 1e9,
                "ai_flop_per_byte": dit_ai,
                "flops_share": dit_f / total_f,
                "verdict": _verdict(dit_ai),
            },
            "e2e": {
                "tflop": total_f / 1e12,
                "gbyte": total_b / 1e9,
                "ai_flop_per_byte": total_f / total_b,
                "verdict": _verdict(total_f / total_b),
            },
        }

    return {
        "hardware": {
            "gpu": "NVIDIA A10",
            "peak_bf16_tflops": A10_BF16_TFLOPS,
            "peak_int8_tops": A10_INT8_TOPS,
            "hbm_bw_gbs": A10_BW_GBS,
            "ridge_bf16_flop_per_byte": RIDGE_BF16,
            "ridge_int8_op_per_byte": RIDGE_INT8,
        },
        "assumptions": {
            "resolution": "480x832x81",
            "dit_seq_len": 32760,
            "dit_weight_dtype": "int8",
            "act_dtype": "bf16",
            "t5_weight_dtype": "bf16",
            "vae_weight_dtype": "bf16",
            "bytes_model": "weights once + activations in/out once per GEMM/conv (roofline lower-bound)",
            "note_steps": (
                "Encoder/Decoder AI independent of denoise steps. "
                "DiT AI per-step constant; total DiT FLOPs/bytes ∝ steps."
            ),
        },
        "components": {
            "t5": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in t5.items()},
            "vae_encoder": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in vae_e.items()},
            "vae_decoder": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in vae_d.items()},
            "dit_one_step": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dit1.items()},
            "encoder_combined": {
                "ai_flop_per_byte": encoder_ai,
                "tflop": encoder_flops / 1e12,
                "gbyte": encoder_bytes / 1e9,
                "verdict": _verdict(encoder_ai),
            },
        },
        "by_steps": by_steps,
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _round_floats(obj: Any, nd: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


def measure_wall_times(
    *,
    config_json: str,
    model_path: str,
    image_path: str,
    prompt: str,
    seed: int,
    steps_list: list[int],
    enc_repeats: int,
    dec_repeats: int,
) -> dict[str, Any]:
    config = set_config(
        model_path=model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=config_json,
    )
    config["parallel"] = False
    seed_all(seed)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log("[measure] loading models...")
    t_load0 = time.perf_counter()
    text_encoder = load_wan_text_encoder(config)[0]
    vae_encoder = load_wan_vae_encoder(config)
    vae_decoder = load_wan_vae_decoder(config)
    model = load_wan_transformer(config)
    load_s = time.perf_counter() - t_load0
    _log(f"[measure] load done in {load_s:.1f}s")

    text_len = int(config.get("text_len", 512))
    img, _ = read_image_input(image_path)
    latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)

    # --- encoder ---
    _log("[measure] encoder warmup + timed")
    _ = text_encoder.infer([prompt])
    _ = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
    _sync()
    t0 = time.perf_counter()
    context = None
    vae_out = None
    for _ in range(enc_repeats):
        context = text_encoder.infer([prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
        vae_out = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
    _sync()
    enc_wall = (time.perf_counter() - t0) / max(enc_repeats, 1)
    assert context is not None and vae_out is not None
    context = context.to(dtype=GET_DTYPE())
    vae_out = vae_out.to(dtype=GET_DTYPE())
    inputs = {
        "text_encoder_output": {"context": context, "context_null": None},
        "image_encoder_output": {"clip_encoder_out": None, "vae_encoder_out": vae_out},
    }
    _log(f"  encoder: {enc_wall:.3f}s")

    denoise_by_steps: dict[str, Any] = {}
    last_latents = None
    for n in steps_list:
        _log(f"[measure] denoise steps={n} (scheduler.infer_steps will be {n}) ...")
        cfg = dict(config)
        cfg["infer_steps"] = int(n)
        # Keep distill list length consistent with N for MoE boundary logic.
        if n == 4:
            cfg["denoising_step_list"] = [1000, 750, 500, 250]
            cfg["boundary_step_index"] = 2
        else:
            # Uniform timesteps; boundary at mid for high/low expert switch.
            cfg["denoising_step_list"] = [int(1000 * (n - i) / n) for i in range(n)]
            cfg["boundary_step_index"] = max(1, n // 2)
        scheduler = WanScheduler(cfg)
        assert scheduler.infer_steps == n, (scheduler.infer_steps, n)
        model.set_scheduler(scheduler)
        # Warmup one step for short runs; skip for N=50 to save ~18s.
        scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=inputs["image_encoder_output"])
        if n <= 8:
            scheduler.step_pre(step_index=0)
            model.infer(inputs)
            scheduler.step_post()
            _sync()
            scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=inputs["image_encoder_output"])
        _sync()
        t0 = time.perf_counter()
        scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=inputs["image_encoder_output"])
        for step_index in range(scheduler.infer_steps):
            scheduler.step_pre(step_index=step_index)
            model.infer(inputs)
            scheduler.step_post()
            if n >= 50 and (step_index + 1) % 10 == 0:
                _log(f"  ... denoise@{n} progress {step_index + 1}/{n}")
        _sync()
        den_s = time.perf_counter() - t0
        last_latents = scheduler.latents
        denoise_by_steps[str(n)] = {
            "wall_s": den_s,
            "per_step_s": den_s / n,
        }
        _log(f"  denoise@{n}: {den_s:.3f}s ({den_s / n:.3f}s/step)")

    assert last_latents is not None
    _log("[measure] decoder warmup + timed")
    _ = vae_decoder.decode(last_latents.to(GET_DTYPE()))
    _sync()
    t0 = time.perf_counter()
    for _ in range(dec_repeats):
        gen_video = vae_decoder.decode(last_latents.to(GET_DTYPE()))
        _ = wan_vae_to_comfy(gen_video)
    _sync()
    dec_wall = (time.perf_counter() - t0) / max(dec_repeats, 1)
    _log(f"  decoder: {dec_wall:.3f}s")

    by_steps: dict[str, Any] = {}
    for n in steps_list:
        den = denoise_by_steps[str(n)]["wall_s"]
        e2e = enc_wall + den + dec_wall
        by_steps[str(n)] = {
            "encoder_s": enc_wall,
            "decoder_s": dec_wall,
            "dit_s": den,
            "dit_per_step_s": denoise_by_steps[str(n)]["per_step_s"],
            "e2e_s": e2e,
            "time_share": {
                "encoder": enc_wall / e2e,
                "decoder": dec_wall / e2e,
                "dit": den / e2e,
            },
        }

    return {
        "model_load_s": load_s,
        "config": {
            "cpu_offload": config.get("cpu_offload"),
            "offload_granularity": config.get("offload_granularity"),
            "target_hw": [config.get("target_height"), config.get("target_width")],
        },
        "by_steps": by_steps,
    }


def write_markdown(theory: dict[str, Any], measured: dict[str, Any] | None, out_md: Path) -> None:
    hw = theory["hardware"]
    ridge = hw["ridge_bf16_flop_per_byte"]
    comps = theory["components"]
    lines: list[str] = []
    lines.append("# Stage arithmetic intensity (paper table)")
    lines.append("")
    lines.append(
        f"Workload: Wan2.2-MoE I2V distill, 480×832×81, single A10. "
        f"Roofline ridge (BF16) = {hw['peak_bf16_tflops']} TFLOPS / {hw['hbm_bw_gbs']} GB/s "
        f"= **{ridge:.1f} FLOP/byte**."
    )
    lines.append("")
    lines.append("## Table 1 — Per-stage algorithmic AI (independent of denoise steps)")
    lines.append("")
    lines.append("| Stage | TFLOP | Traffic (GB) | AI (FLOP/byte) | vs ridge 208 | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    enc = comps["encoder_combined"]
    lines.append(
        f"| Encoder (T5+VAE) | {enc['tflop']:.3f} | {enc['gbyte']:.3f} | "
        f"**{enc['ai_flop_per_byte']:.1f}** | {enc['ai_flop_per_byte']/ridge:.2f}× | {enc['verdict']} |"
    )
    t5 = comps["t5"]
    lines.append(
        f"|  └ T5 only | {t5['tflop']:.3f} | {t5['gbyte']:.3f} | "
        f"{t5['ai_flop_per_byte']:.1f} | {t5['ai_flop_per_byte']/ridge:.2f}× | {t5['verdict']} |"
    )
    ve = comps["vae_encoder"]
    lines.append(
        f"|  └ VAE enc only | {ve['tflop']:.3f} | {ve['gbyte']:.3f} | "
        f"{ve['ai_flop_per_byte']:.1f} | {ve['ai_flop_per_byte']/ridge:.2f}× | {ve['verdict']} |"
    )
    vd = comps["vae_decoder"]
    lines.append(
        f"| Decoder (VAE) | {vd['tflop']:.3f} | {vd['gbyte']:.3f} | "
        f"**{vd['ai_flop_per_byte']:.1f}** | {vd['ai_flop_per_byte']/ridge:.2f}× | {vd['verdict']} |"
    )
    d1 = comps["dit_one_step"]
    lines.append(
        f"| DiT (per step) | {d1['tflop']:.2f} | {d1['gbyte']:.2f} | "
        f"**{d1['ai_flop_per_byte']:.1f}** | {d1['ai_flop_per_byte']/ridge:.1f}× | {d1['verdict']} |"
    )
    lines.append("")
    lines.append(
        "DiT AI does **not** change with step count under the HBM traffic model "
        "(FLOPs and bytes both ∝ steps). Enc/Dec are also step-invariant."
    )
    lines.append("")
    lines.append("## Table 2 — FLOPs share & E2E effective AI vs steps")
    lines.append("")
    lines.append("| Steps | Enc TFLOP (share) | Dec TFLOP (share) | DiT TFLOP (share) | E2E AI (FLOP/byte) |")
    lines.append("|---:|---|---|---|---:|")
    for n, row in theory["by_steps"].items():
        e, d, t, ee = row["encoder"], row["decoder"], row["dit"], row["e2e"]
        lines.append(
            f"| {n} | {e['tflop']:.3f} ({100*e['flops_share']:.1f}%) | "
            f"{d['tflop']:.3f} ({100*d['flops_share']:.1f}%) | "
            f"{t['tflop']:.2f} ({100*t['flops_share']:.1f}%) | "
            f"**{ee['ai_flop_per_byte']:.1f}** |"
        )
    lines.append("")

    if measured is not None:
        lines.append("## Table 3 — Measured wall time (single A10, block offload on DiT)")
        lines.append("")
        cfg = measured.get("config", {})
        lines.append(
            f"Config: `cpu_offload={cfg.get('cpu_offload')}`, "
            f"`offload_granularity={cfg.get('offload_granularity')}`."
        )
        lines.append("")
        lines.append("| Steps | Enc (s) | Dec (s) | DiT (s) | DiT/step (s) | E2E (s) | Enc% | Dec% | DiT% |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n, row in measured["by_steps"].items():
            ts = row["time_share"]
            lines.append(
                f"| {n} | {row['encoder_s']:.2f} | {row['decoder_s']:.2f} | {row['dit_s']:.2f} | "
                f"{row['dit_per_step_s']:.2f} | {row['e2e_s']:.2f} | "
                f"{100*ts['encoder']:.1f} | {100*ts['decoder']:.1f} | {100*ts['dit']:.1f} |"
            )
        lines.append("")
        lines.append("## Table 4 — Achieved TFLOPS (theory FLOPs / measured time)")
        lines.append("")
        lines.append("| Steps | Enc TFLOPS | Dec TFLOPS | DiT TFLOPS | vs A10 BF16 peak |")
        lines.append("|---:|---:|---:|---:|---:|")
        for n, mrow in measured["by_steps"].items():
            trow = theory["by_steps"][str(n)]
            enc_t = trow["encoder"]["tflop"] / mrow["encoder_s"]
            dec_t = trow["decoder"]["tflop"] / mrow["decoder_s"]
            dit_t = trow["dit"]["tflop"] / mrow["dit_s"]
            lines.append(
                f"| {n} | {enc_t:.1f} | {dec_t:.1f} | {dit_t:.1f} | "
                f"DiT {100*dit_t/A10_BF16_TFLOPS:.0f}% of 125 |"
            )
        lines.append("")

    lines.append("## Takeaways for paper")
    lines.append("")
    lines.append(
        f"1. **Encoder AI ≈ {enc['ai_flop_per_byte']:.0f}**, **Decoder AI ≈ {vd['ai_flop_per_byte']:.0f}**, "
        f"**DiT AI ≈ {d1['ai_flop_per_byte']:.0f}** FLOP/byte (all ≫ A10 ridge {ridge:.0f}) → all compute-leaning."
    )
    lines.append(
        f"2. DiT AI is **~{d1['ai_flop_per_byte']/enc['ai_flop_per_byte']:.1f}×** Enc and "
        f"**~{d1['ai_flop_per_byte']/vd['ai_flop_per_byte']:.1f}×** Dec → strongest compute intensity; "
        "Enc/Dec still compute-side vs hardware but closer to the ridge."
    )
    lines.append(
        "3. Changing denoise steps (1/4/8/50) does **not** change per-stage AI; it changes **DiT FLOPs/time share** and E2E effective AI (pulled toward DiT as N grows)."
    )
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--steps", default="1,4,8,50")
    parser.add_argument("--mode", choices=["theory", "measure", "both"], default="both")
    parser.add_argument("--enc_repeats", type=int, default=2)
    parser.add_argument("--dec_repeats", type=int, default=2)
    parser.add_argument(
        "--output_json",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_stage_arith_intensity.json",
    )
    parser.add_argument(
        "--output_md",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_stage_arith_intensity.md",
    )
    args = parser.parse_args()

    steps_list = [int(x) for x in args.steps.split(",") if x.strip()]
    theory = build_theory(steps_list)
    measured = None
    if args.mode in ("measure", "both"):
        measured = measure_wall_times(
            config_json=args.config_json,
            model_path=args.model_path,
            image_path=args.image_path,
            prompt=args.prompt,
            seed=args.seed,
            steps_list=steps_list,
            enc_repeats=args.enc_repeats,
            dec_repeats=args.dec_repeats,
        )

    out = {
        "metric": "stage_arithmetic_intensity",
        "theory": _round_floats(theory, 6),
        "measured": _round_floats(measured, 4) if measured is not None else None,
    }
    # attach achieved tflops into measured
    if measured is not None:
        ach: dict[str, Any] = {}
        for n, mrow in measured["by_steps"].items():
            trow = theory["by_steps"][str(n)]
            ach[str(n)] = {
                "encoder_tflops": trow["encoder"]["tflop"] / mrow["encoder_s"],
                "decoder_tflops": trow["decoder"]["tflop"] / mrow["decoder_s"],
                "dit_tflops": trow["dit"]["tflop"] / mrow["dit_s"],
            }
        out["achieved_tflops"] = _round_floats(ach, 2)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    write_markdown(theory, measured, out_md)
    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
