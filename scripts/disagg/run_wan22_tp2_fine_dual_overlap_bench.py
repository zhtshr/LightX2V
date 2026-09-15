#!/usr/bin/env python3
"""Wan2.2 TP=2 dual-tenant fine-grained all_reduce overlap benchmark.

Per-AR async all_reduce + pump AR-free cross_ffn slices on the peer tenant.
Compares dual b2b vs ar-overlap (SP-style) vs fine-overlap throughput.

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_wan22_tp2_fine_dual_overlap_bench.py \\
    --config_json configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json \\
    --inputs_cache save_results/optimization_study/phase1_encoder_inputs_256x256.pt \\
    --output_json save_results/optimization_study/wan22_256_tp2_fine_dual_overlap.json
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
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tp1_single_s(study_dir: Path, override: float | None, config: dict | None = None) -> float | None:
    if override is not None and override > 0:
        return override
    if config is not None:
        h, w = config.get("target_height"), config.get("target_width")
        if h == 256 and w == 256:
            lat_path = study_dir / "wan22_256_tp1_latency.json"
            if lat_path.is_file():
                return float(json.loads(lat_path.read_text())["transformer_compute_s"])
        if h == 480 and w == 832:
            for name in ("wan22_distill_tp1_real.json", "wan22_distill_tp1_offload.json"):
                lat_path = study_dir / name
                if lat_path.is_file():
                    return float(json.loads(lat_path.read_text())["transformer_compute_s"])
    lat_path = study_dir / "wan22_256_tp1_latency.json"
    if lat_path.is_file():
        return float(json.loads(lat_path.read_text())["transformer_compute_s"])
    return None


def _resolution_label(config: dict) -> str:
    return f"{config.get('target_height', '?')}x{config.get('target_width', '?')}"


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
    parser.add_argument("--task", default="i2v", choices=("i2v", "t2v"))
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument(
        "--inputs_cache",
        default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs_256x256.pt",
    )
    parser.add_argument("--tp1_single_s", type=float, default=0.0)
    parser.add_argument("--seed_a", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=0)
    parser.add_argument("--tp_norm_p2p", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    if args.tensor_p_size < 2:
        raise SystemExit("tensor_p_size must be >= 2")

    here = Path(__file__).parent
    p1 = _load_module("phase1_bench", here / "run_phase1_transformer_bench.py")
    ov = _load_module("overlap_bench", here / "run_phase3_dual_overlap_bench.py")
    fine = _load_module("tp_fine_overlap", here / "tp_fine_overlap.py")
    micro = _load_module("tp_micro_overlap", here / "tp_micro_overlap.py")
    study_dir = Path("/root/zht/LightX2V/save_results/optimization_study")

    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        seed=args.seed_a,
        seq_p_size=1,
        tensor_p_size=args.tensor_p_size,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt=(
            "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量"
        ),
    )
    config = p1._load_config(ns)
    if args.tp_norm_p2p:
        config["tp_norm_p2p"] = True
    seed_all(args.seed_a)
    p1._init_distributed(config)

    payload_a = p1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=ns.prompt,
        image_path=ns.image_path,
        seed=args.seed_a,
        force=False,
        task=args.task,
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
    resolution = _resolution_label(config)
    tp1_single_s = _load_tp1_single_s(study_dir, args.tp1_single_s or None, config)
    tp1_parallel_ceiling_rps = _rps(2, tp1_single_s) if tp1_single_s else None

    for _ in range(args.warmup):
        ov._run_denoise_serial(model, scheduler_a, payload_a)

    single_s = ov._time_fn(lambda: ov._run_denoise_serial(model, scheduler_a, payload_a))

    err: str | None = None
    dual_fine_serial_s: float | None = None
    dual_fine_overlap_s: float | None = None
    dual_ar_overlap_s: float | None = None
    dual_b2b_s: float | None = None
    dual_micro_serial_s: float | None = None
    dual_micro_overlap_s: float | None = None
    micro_serial_stats: dict[str, Any] = {}
    micro_overlap_stats: dict[str, Any] = {}
    fine_serial_stats: dict[str, Any] = {}
    fine_overlap_stats: dict[str, Any] = {}
    ar_overlap_stats: dict[str, Any] = {}

    fine_kwargs = dict(
        bind_tenant=ov._bind_tenant,
        run_self_attn_block=ov._run_self_attn_block,
        preload_blocks=ov._preload_blocks,
        pre_infer_tenant=ov._pre_infer_tenant,
        finish_step_tenant=ov._finish_step_tenant,
        time_fn=ov._time_fn,
    )

    micro_kwargs = dict(
        bind_tenant=ov._bind_tenant,
        ensure_block=ov._ensure_block,
        preload_blocks=ov._preload_blocks,
        pre_infer_tenant=ov._pre_infer_tenant,
        finish_step_tenant=ov._finish_step_tenant,
        capture_ti_snap=ov._capture_ti_snap,
        time_fn=ov._time_fn,
    )

    try:
        _cuda_gc()
        dual_ar_overlap_s, ar_overlap_stats = ov._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps,
        )
        _cuda_gc()
        dual_micro_overlap_s, micro_overlap_stats = micro.run_dual_micro_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps, **micro_kwargs,
        )
        _cuda_gc()
        dual_micro_serial_s, micro_serial_stats = micro.run_dual_micro_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps, **micro_kwargs,
        )
        _cuda_gc()
        dual_fine_overlap_s, fine_overlap_stats = fine.run_dual_fine_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps, **fine_kwargs,
        )
        _cuda_gc()
        dual_fine_serial_s, fine_serial_stats = fine.run_dual_fine_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps, **fine_kwargs,
        )
        _cuda_gc()
        dual_b2b_s = ov._time_fn(lambda: (
            ov._run_denoise_serial(model, scheduler_a, payload_a),
            ov._run_denoise_serial(model, scheduler_b, payload_b),
        ))
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if is_main_process():
            print(f"fine dual pipeline failed: {exc}")
            import traceback
            traceback.print_exc()

    speedup_micro_vs_serial = (
        dual_micro_serial_s / dual_micro_overlap_s
        if dual_micro_serial_s and dual_micro_overlap_s and dual_micro_overlap_s > 0
        else None
    )
    speedup_micro_vs_b2b = (
        dual_b2b_s / dual_micro_overlap_s
        if dual_b2b_s and dual_micro_overlap_s and dual_micro_overlap_s > 0
        else None
    )
    speedup_micro_vs_ar = (
        dual_ar_overlap_s / dual_micro_overlap_s
        if dual_ar_overlap_s and dual_micro_overlap_s and dual_micro_overlap_s > 0
        else None
    )
    speedup_fine_vs_serial = (
        dual_fine_serial_s / dual_fine_overlap_s
        if dual_fine_serial_s and dual_fine_overlap_s and dual_fine_overlap_s > 0
        else None
    )
    speedup_fine_vs_b2b = (
        dual_b2b_s / dual_fine_overlap_s
        if dual_b2b_s and dual_fine_overlap_s and dual_fine_overlap_s > 0
        else None
    )
    speedup_fine_vs_ar = (
        dual_ar_overlap_s / dual_fine_overlap_s
        if dual_ar_overlap_s and dual_fine_overlap_s and dual_fine_overlap_s > 0
        else None
    )

    fine_overlap_rps = _rps(2, dual_fine_overlap_s)
    pct_parallel = (
        100.0 * fine_overlap_rps / tp1_parallel_ceiling_rps
        if fine_overlap_rps and tp1_parallel_ceiling_rps and tp1_parallel_ceiling_rps > 0
        else None
    )

    result = {
        "resolution": resolution,
        "target_video_length": config.get("target_video_length"),
        "tensor_p_size": args.tensor_p_size,
        "tp_norm_p2p": bool(config.get("tp_norm_p2p", False)),
        "measure_steps": measure_steps,
        "tp1_single_s": tp1_single_s,
        "single_transformer_s": single_s,
        "dual_back_to_back_s": dual_b2b_s,
        "dual_fine_serial_s": dual_fine_serial_s,
        "dual_fine_overlap_s": dual_fine_overlap_s,
        "dual_micro_serial_s": dual_micro_serial_s,
        "dual_micro_overlap_s": dual_micro_overlap_s,
        "micro_serial_stats": micro_serial_stats,
        "micro_overlap_stats": micro_overlap_stats,
        "fine_overlap_stats": fine_overlap_stats,
        "dual_ar_overlap_s": dual_ar_overlap_s,
        "fine_serial_stats": fine_serial_stats,
        "ar_overlap_stats": ar_overlap_stats,
        "error": err,
        "throughput": {
            "single_rps": _rps(1, single_s),
            "tp1_parallel_ceiling_rps": tp1_parallel_ceiling_rps,
            "dual_back_to_back_rps": _rps(2, dual_b2b_s),
            "dual_fine_serial_rps": _rps(2, dual_fine_serial_s),
            "dual_fine_overlap_rps": fine_overlap_rps,
            "dual_ar_overlap_rps": _rps(2, dual_ar_overlap_s),
            "dual_micro_serial_rps": _rps(2, dual_micro_serial_s),
            "dual_micro_overlap_rps": _rps(2, dual_micro_overlap_s),
            "pct_of_2x_single_card": pct_parallel,
        },
        "speedup": {
            "fine_overlap_vs_fine_serial": speedup_fine_vs_serial,
            "fine_overlap_vs_back_to_back": speedup_fine_vs_b2b,
            "fine_overlap_vs_ar_overlap": speedup_fine_vs_ar,
            "micro_overlap_vs_micro_serial": speedup_micro_vs_serial,
            "micro_overlap_vs_back_to_back": speedup_micro_vs_b2b,
            "micro_overlap_vs_ar_overlap": speedup_micro_vs_ar,
        },
        "interpretation": {
            "tp1_parallel_ceiling_rps": "2 / tp1_single_s — two independent single-GPU cards",
            "dual_fine_overlap": "Per-AR async + pump cross_ffn AR-free slices (seg0 pre-cross_o)",
            "dual_ar_overlap": "SP-style OnceCompute cross_ffn during peer self_attn AR",
            "dual_micro_overlap": "Scheme-1 micro-phase: comp₁-comm₁-comp₂-comm₂ alignment per block section",
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")

        md_path = out.with_suffix(".md")
        tp = result["throughput"]
        sp = result["speedup"]
        ceiling = tp.get("tp1_parallel_ceiling_rps") or 0

        def _pct(rps: float | None) -> str:
            if rps is None or not ceiling:
                return "—"
            return f"{100 * rps / ceiling:.1f}%"

        def _fmt_sp(key: str) -> str:
            v = sp.get(key)
            return f"{v:.3f}×" if v is not None else "—"

        lines = [
            f"# Wan2.2-Distill {resolution} TP=2 — Fine-Grained Dual Overlap",
            "",
            f"TP=1 single: **{tp1_single_s:.3f}s** | 2×单卡并行 ceiling: **{ceiling:.4f} req/s**"
            if tp1_single_s else f"2×单卡并行 ceiling: **{ceiling:.4f} req/s**",
            "",
            "## Throughput (2 requests)",
            "",
            "| mode | wall (s) | req/s | vs 2×单卡 |",
            "|---|---:|---:|---:|",
        ]
        for label, key_s, key_rps in (
            ("dual back-to-back", "dual_back_to_back_s", "dual_back_to_back_rps"),
            ("dual fine-serial", "dual_fine_serial_s", "dual_fine_serial_rps"),
            ("dual **fine-overlap**", "dual_fine_overlap_s", "dual_fine_overlap_rps"),
            ("dual ar-overlap (SP)", "dual_ar_overlap_s", "dual_ar_overlap_rps"),
            ("dual micro-serial", "dual_micro_serial_s", "dual_micro_serial_rps"),
            ("dual **micro-overlap**", "dual_micro_overlap_s", "dual_micro_overlap_rps"),
        ):
            wall = result.get(key_s)
            rps = tp.get(key_rps)
            if wall is None or rps is None:
                lines.append(f"| {label} | — | — | — |")
            else:
                lines.append(f"| {label} | {wall:.3f} | {rps:.4f} | {_pct(rps)} |")

        fos = fine_overlap_stats or {}
        lines.extend([
            "",
            "## Speedup",
            "",
            f"- fine-overlap vs fine-serial: **{_fmt_sp('fine_overlap_vs_fine_serial')}**",
            f"- fine-overlap vs back-to-back: **{_fmt_sp('fine_overlap_vs_back_to_back')}**",
            f"- fine-overlap vs ar-overlap: **{_fmt_sp('fine_overlap_vs_ar_overlap')}**",
            f"- micro-overlap vs micro-serial: **{_fmt_sp('micro_overlap_vs_micro_serial')}**",
            f"- micro-overlap vs back-to-back: **{_fmt_sp('micro_overlap_vs_back_to_back')}**",
            f"- micro-overlap vs ar-overlap: **{_fmt_sp('micro_overlap_vs_ar_overlap')}**",
            "",
            "## Fine overlap comm stats",
            "",
            f"- all_reduce calls: {fos.get('all_reduce_calls', '—')}",
            f"- overlap windows: {fos.get('all_reduce_overlap_windows', '—')}",
            f"- pump segments run: {fos.get('pump_segments_run', '—')}",
        ])
        mos = micro_overlap_stats or {}
        lines.extend([
            "",
            "## Micro overlap stats",
            "",
            f"- all_reduce calls: {mos.get('all_reduce_calls', '—')}",
            f"- overlap windows: {mos.get('all_reduce_overlap_windows', '—')}",
            f"- peer comp steps pumped: {mos.get('peer_comp_steps', '—')}",
        ])
        if err:
            lines.extend(["", f"**Error:** {err}"])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {md_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
