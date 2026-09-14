#!/usr/bin/env python3
"""Wan2.2 TP=2 six-phase dual-tenant pipeline overlap benchmark.

Run (ensure ``tp_norm_p2p`` — small norm AR via P2P, not NCCL; row COMM AR stays NCCL):
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_wan22_tp2_phase_pipeline_bench.py \\
    --config_json configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json \\
    --inputs_cache save_results/optimization_study/phase1_encoder_inputs_256x256.pt \\
    --output_json save_results/optimization_study/wan22_256_tp2_phase_pipeline.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rps(n: float, t: float | None) -> float | None:
    return n / t if t and t > 0 else None


def _cuda_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_p_size", type=int, default=2)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=0)
    parser.add_argument("--seed_a", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    parser.add_argument("--tp_norm_p2p", action="store_true", default=True)
    parser.add_argument("--no_tp_norm_p2p", action="store_false", dest="tp_norm_p2p")
    parser.add_argument("--tp_comm_p2p", action="store_true", default=False,
                        help="Experimental: replace row COMM NCCL AR with IPC P2P copy")
    parser.add_argument("--tp_chunked_ar", action="store_true", default=False,
                        help="Split COMM all-reduce into chunks interleaved with peer COMP steps")
    args = parser.parse_args()

    here = Path(__file__).parent
    p1 = _load_module("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load_module("overlap", here / "run_phase3_dual_overlap_bench.py")
    phase = _load_module("phase_pipe", here / "tp_phase_pipeline.py")

    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        seed=args.seed_a,
        seq_p_size=1,
        tensor_p_size=args.tensor_p_size,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )
    config = p1._load_config(ns)
    if args.tp_norm_p2p:
        config["tp_norm_p2p"] = True
    seed_all(args.seed_a)
    p1._init_distributed(config)
    phase.configure_phase_pipeline(
        use_comm_p2p=bool(args.tp_comm_p2p),
        use_chunked_ar=bool(args.tp_chunked_ar),
    )

    payload_a = p1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=ns.prompt,
        image_path=ns.image_path,
        seed=args.seed_a,
        force=False,
        task="i2v",
        negative_prompt=ns.negative_prompt,
    )
    payload_a = p1._prepare_payload_on_device(payload_a)
    payload_b = p1._prepare_payload_on_device({
        "seed": args.seed_b,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    scheduler_a = WanScheduler(config)
    scheduler_b = WanScheduler(config)
    model.set_scheduler(scheduler_a)
    tenant_a = ov.TenantCtx("A", scheduler_a, payload_a["inputs"])
    tenant_b = ov.TenantCtx("B", scheduler_b, payload_b["inputs"])

    measure_steps = args.measure_steps or scheduler_a.infer_steps
    phase_kwargs = dict(
        bind_tenant=ov._bind_tenant,
        ensure_block=ov._ensure_block,
        preload_blocks=ov._preload_blocks,
        pre_infer_tenant=ov._pre_infer_tenant,
        finish_step_tenant=ov._finish_step_tenant,
        capture_ti_snap=ov._capture_ti_snap,
        time_fn=ov._time_fn,
    )

    if args.warmup > 0:
        for _ in range(args.warmup):
            ov._run_denoise_serial(model, scheduler_a, payload_a)
        _cuda_gc()

    err: str | None = None
    dual_b2b_s: float | None = None
    dual_ar_s: float | None = None
    dual_phase_overlap_s: float | None = None
    dual_phase_serial_s: float | None = None
    phase_overlap_stats: dict[str, Any] = {}
    phase_serial_stats: dict[str, Any] = {}

    try:
        _cuda_gc()
        dual_phase_overlap_s, phase_overlap_stats = phase.run_dual_phase_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps, **phase_kwargs,
        )
        _cuda_gc()
        dual_phase_serial_s, phase_serial_stats = phase.run_dual_phase_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps, **phase_kwargs,
        )
        _cuda_gc()
        dual_ar_s, _ = ov._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps,
        )
        _cuda_gc()
        dual_b2b_s = ov._time_fn(lambda: (
            ov._run_denoise_serial(model, scheduler_a, payload_a),
            ov._run_denoise_serial(model, scheduler_b, payload_b),
        ))
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if is_main_process():
            import traceback
            traceback.print_exc()

    result = {
        "config_json": args.config_json,
        "tensor_p_size": args.tensor_p_size,
        "tp_norm_p2p": bool(config.get("tp_norm_p2p", False)),
        "tp_comm_p2p": bool(args.tp_comm_p2p),
        "tp_chunked_ar": bool(args.tp_chunked_ar),
        "measure_steps": measure_steps,
        "dual_back_to_back_s": dual_b2b_s,
        "dual_ar_overlap_s": dual_ar_s,
        "dual_phase_overlap_s": dual_phase_overlap_s,
        "dual_phase_serial_s": dual_phase_serial_s,
        "phase_overlap_stats": phase_overlap_stats,
        "phase_serial_stats": phase_serial_stats,
        "error": err,
        "throughput": {
            "dual_back_to_back_rps": _rps(2, dual_b2b_s),
            "dual_ar_overlap_rps": _rps(2, dual_ar_s),
            "dual_phase_overlap_rps": _rps(2, dual_phase_overlap_s),
            "dual_phase_serial_rps": _rps(2, dual_phase_serial_s),
        },
        "speedup": {
            "phase_overlap_vs_phase_serial": (
                dual_phase_serial_s / dual_phase_overlap_s
                if dual_phase_serial_s and dual_phase_overlap_s and dual_phase_overlap_s > 0
                else None
            ),
            "phase_overlap_vs_back_to_back": (
                dual_b2b_s / dual_phase_overlap_s
                if dual_b2b_s and dual_phase_overlap_s and dual_phase_overlap_s > 0
                else None
            ),
            "phase_overlap_vs_ar_overlap": (
                dual_ar_s / dual_phase_overlap_s
                if dual_ar_s and dual_phase_overlap_s and dual_phase_overlap_s > 0
                else None
            ),
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        md = out.with_suffix(".md")
        tp = result["throughput"]
        sp = result["speedup"]
        lines = [
            "# Wan TP=2 — 6-Phase Pipeline Dual Overlap",
            "",
            f"tp_norm_p2p: **{result['tp_norm_p2p']}** | tp_comm_p2p: **{result['tp_comm_p2p']}** | tp_chunked_ar: **{result['tp_chunked_ar']}** | measure_steps: **{measure_steps}**",
            "",
            "| mode | wall (s) | req/s |",
            "|---|---:|---:|",
        ]
        for label, key_s, key_r in (
            ("dual back-to-back", "dual_back_to_back_s", "dual_back_to_back_rps"),
            ("dual ar-overlap", "dual_ar_overlap_s", "dual_ar_overlap_rps"),
            ("dual phase-serial", "dual_phase_serial_s", "dual_phase_serial_rps"),
            ("dual **phase-overlap**", "dual_phase_overlap_s", "dual_phase_overlap_rps"),
        ):
            lines.append(
                f"| {label} | {result.get(key_s) or '—'} | {tp.get(key_r) or '—'} |"
            )
        lines.extend([
            "",
            f"- phase-overlap vs phase-serial: **{sp.get('phase_overlap_vs_phase_serial', '—')}**",
            f"- phase-overlap vs b2b: **{sp.get('phase_overlap_vs_back_to_back', '—')}**",
            f"- phase-overlap vs ar-overlap: **{sp.get('phase_overlap_vs_ar_overlap', '—')}**",
            "",
        ])
        if err:
            lines.append(f"**Error:** {err}")
        md.write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")
        print(f"wrote {md}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
