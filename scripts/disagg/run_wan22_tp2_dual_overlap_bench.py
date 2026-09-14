#!/usr/bin/env python3
"""Wan2.2 TP=2 dual-tenant all_reduce overlap benchmark (256x256 validation).

Patches ``dist.all_reduce`` during tenant self_attn; overlaps other tenant cross_ffn.
Compares dual back-to-back vs ar-serial vs ar-overlap; reports throughput vs TP=1 ceiling.

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_wan22_tp2_dual_overlap_bench.py \\
    --config_json configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json \\
    --inputs_cache save_results/optimization_study/phase1_encoder_inputs_256x256.pt \\
    --output_json save_results/optimization_study/wan22_256_tp2_dual_overlap.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1_module():
    bench_path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1_bench", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {bench_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_overlap_module():
    bench_path = Path(__file__).with_name("run_phase3_dual_overlap_bench.py")
    spec = importlib.util.spec_from_file_location("overlap_bench", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {bench_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tp1_single_s(study_dir: Path, override: float | None) -> float | None:
    if override is not None and override > 0:
        return override
    lat_path = study_dir / "wan22_256_tp1_latency.json"
    if lat_path.is_file():
        return float(json.loads(lat_path.read_text())["transformer_compute_s"])
    return None


def _rps(n: float, t: float | None) -> float | None:
    return n / t if t and t > 0 else None


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
    parser.add_argument(
        "--tp1_single_s",
        type=float,
        default=0.0,
        help="TP=1 single-request latency for ceiling (0 = load from wan22_256_tp1_latency.json)",
    )
    parser.add_argument(
        "--comm_wall_ratio",
        type=float,
        default=0.473,
        help="NCCL all_reduce fraction of wall (from tp2 overhead profile)",
    )
    parser.add_argument("--seed_a", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=0, help="0 = all infer_steps")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    if args.tensor_p_size < 2:
        raise SystemExit("tensor_p_size must be >= 2 for dual-tenant TP overlap bench")

    p1 = _load_phase1_module()
    ov = _load_overlap_module()
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
    tp1_single_s = _load_tp1_single_s(study_dir, args.tp1_single_s or None)

    for _ in range(args.warmup):
        ov._run_denoise_serial(model, scheduler_a, payload_a)

    single_s = ov._time_fn(lambda: ov._run_denoise_serial(model, scheduler_a, payload_a))

    dual_b2b_s = ov._time_fn(lambda: (
        ov._run_denoise_serial(model, scheduler_a, payload_a),
        ov._run_denoise_serial(model, scheduler_b, payload_b),
    ))

    def _dual_decomposed_b2b() -> None:
        ov._run_single_decomposed_once(
            model, tenant_a, payload_a, patch_a2a=True, steps=measure_steps,
        )
        ov._run_single_decomposed_once(
            model, tenant_b, payload_b, patch_a2a=True, steps=measure_steps,
        )

    dual_decomposed_b2b_s = ov._time_fn(_dual_decomposed_b2b)

    err: str | None = None
    dual_ar_serial_s: float | None = None
    dual_ar_overlap_s: float | None = None
    dual_layer_serial_s: float | None = None
    dual_layer_overlap_s: float | None = None
    layer_overlap_error = (
        "skipped: threaded layer overlap deadlocks on TP (all_reduce in both self_attn and cross_ffn)"
    )
    ar_serial_stats: dict[str, Any] = {}
    ar_overlap_stats: dict[str, Any] = {}
    one_step_ar_serial_s: float | None = None
    one_step_ar_overlap_s: float | None = None
    one_step_layer_serial_s: float | None = None
    one_step_layer_overlap_s: float | None = None
    pair_overlap_timings: list[dict[str, float]] = []

    try:
        dual_ar_serial_s, ar_serial_stats = ov._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps,
        )
        dual_ar_overlap_s, ar_overlap_stats = ov._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps,
        )
        dual_layer_serial_s, _ = ov._run_dual_layer_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps,
        )
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if is_main_process():
            print(f"dual ar pipeline failed: {exc}")

    # Layer-thread overlap is unsafe for TP: both self_attn and cross_ffn issue all_reduce;
    # concurrent NCCL from two threads deadlocks (see layer_overlap_error note in output).

    if err is None:
        try:
            one_step_ar_serial_s, ar_one_serial = ov._run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b, overlap=False, steps=1,
            )
            one_step_ar_overlap_s, ar_one_overlap = ov._run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b, overlap=True, steps=1,
            )
            ar_serial_stats = {**ar_serial_stats, "one_step": ar_one_serial}
            ar_overlap_stats = {**ar_overlap_stats, "one_step": ar_one_overlap}
        except Exception as exc:  # noqa: BLE001
            if err is None:
                err = str(exc)

    fair_dual_baseline_s = dual_decomposed_b2b_s or dual_b2b_s
    pure_compute_dual_s = single_s * 2.0 * (1.0 - args.comm_wall_ratio)
    tp1_ceiling_rps = _rps(2, tp1_single_s * 2) if tp1_single_s else None

    speedup_ar_vs_serial = (
        dual_ar_serial_s / dual_ar_overlap_s
        if dual_ar_serial_s and dual_ar_overlap_s and dual_ar_overlap_s > 0
        else None
    )
    speedup_ar_vs_b2b = (
        dual_b2b_s / dual_ar_overlap_s
        if dual_b2b_s and dual_ar_overlap_s and dual_ar_overlap_s > 0
        else None
    )
    speedup_ar_vs_fair = (
        fair_dual_baseline_s / dual_ar_overlap_s
        if fair_dual_baseline_s and dual_ar_overlap_s and dual_ar_overlap_s > 0
        else None
    )
    one_step_speedup = (
        one_step_ar_serial_s / one_step_ar_overlap_s
        if one_step_ar_serial_s and one_step_ar_overlap_s and one_step_ar_overlap_s > 0
        else None
    )

    speedup_layer_vs_serial = (
        dual_layer_serial_s / dual_layer_overlap_s
        if dual_layer_serial_s and dual_layer_overlap_s and dual_layer_overlap_s > 0
        else None
    )
    speedup_layer_vs_b2b = (
        dual_b2b_s / dual_layer_overlap_s
        if dual_b2b_s and dual_layer_overlap_s and dual_layer_overlap_s > 0
        else None
    )
    one_step_layer_speedup = (
        one_step_layer_serial_s / one_step_layer_overlap_s
        if one_step_layer_serial_s and one_step_layer_overlap_s and one_step_layer_overlap_s > 0
        else None
    )
    one_step_pair_stats = ov._summarize_pair_timings(pair_overlap_timings) if pair_overlap_timings else {}

    dual_ar_overlap_rps = _rps(2, dual_ar_overlap_s)
    dual_layer_overlap_rps = _rps(2, dual_layer_overlap_s)
    best_dual_s = min(
        s for s in (dual_b2b_s, dual_ar_overlap_s, dual_layer_overlap_s) if s
    )
    best_dual_rps = _rps(2, best_dual_s)
    pct_tp1_ceiling = (
        100.0 * best_dual_rps / tp1_ceiling_rps
        if best_dual_rps and tp1_ceiling_rps and tp1_ceiling_rps > 0
        else None
    )

    result = {
        "resolution": "256x256",
        "tensor_p_size": args.tensor_p_size,
        "task": args.task,
        "model_cls": args.model_cls,
        "config_json": args.config_json,
        "measure_steps": measure_steps,
        "comm_wall_ratio": args.comm_wall_ratio,
        "tp1_single_s": tp1_single_s,
        "single_transformer_s": single_s,
        "dual_back_to_back_s": dual_b2b_s,
        "dual_decomposed_back_to_back_s": dual_decomposed_b2b_s,
        "fair_dual_baseline_s": fair_dual_baseline_s,
        "dual_ar_serial_s": dual_ar_serial_s,
        "dual_ar_overlap_s": dual_ar_overlap_s,
        "dual_layer_serial_s": dual_layer_serial_s,
        "dual_layer_overlap_s": dual_layer_overlap_s,
        "one_step_ar_serial_s": one_step_ar_serial_s,
        "one_step_ar_overlap_s": one_step_ar_overlap_s,
        "one_step_layer_serial_s": one_step_layer_serial_s,
        "one_step_layer_overlap_s": one_step_layer_overlap_s,
        "one_step_pair_overlap_stats": one_step_pair_stats,
        "ar_serial_stats": ar_serial_stats,
        "ar_overlap_stats": ar_overlap_stats,
        "error": err,
        "layer_overlap_error": layer_overlap_error,
        "throughput": {
            "single_rps": _rps(1, single_s),
            "tp1_ceiling_rps": tp1_ceiling_rps,
            "dual_back_to_back_rps": _rps(2, dual_b2b_s),
            "dual_decomposed_back_to_back_rps": _rps(2, dual_decomposed_b2b_s),
            "dual_ar_serial_rps": _rps(2, dual_ar_serial_s),
            "dual_ar_overlap_rps": dual_ar_overlap_rps,
            "dual_layer_serial_rps": _rps(2, dual_layer_serial_s),
            "dual_layer_overlap_rps": dual_layer_overlap_rps,
            "best_dual_rps": best_dual_rps,
            "pure_compute_ceiling_rps": _rps(2, pure_compute_dual_s),
            "pct_of_tp1_ceiling": pct_tp1_ceiling,
        },
        "speedup": {
            "ar_overlap_vs_ar_serial": speedup_ar_vs_serial,
            "ar_overlap_vs_back_to_back": speedup_ar_vs_b2b,
            "ar_overlap_vs_fair_dual_baseline": speedup_ar_vs_fair,
            "one_step_ar_overlap_vs_serial": one_step_speedup,
            "layer_overlap_vs_layer_serial": speedup_layer_vs_serial,
            "layer_overlap_vs_back_to_back": speedup_layer_vs_b2b,
            "one_step_layer_overlap_vs_serial": one_step_layer_speedup,
        },
        "interpretation": {
            "tp1_ceiling_rps": "2 / (2 * tp1_single_s) — two independent TP=1 cards",
            "dual_ar_serial": "Decomposed dual-tenant; all_reduce synchronous (patch on, overlap off)",
            "dual_ar_overlap": "Per-AR async + other tenant cross_ffn (SP-style; poor for dense TP AR)",
            "dual_layer_overlap": "Layer-granularity: self_attn (comm stream) || cross_ffn (compute stream)",
            "dual_back_to_back": "Two model.infer() — production baseline",
            "meaningful": "best_dual_rps ≈ tp1_ceiling_rps when overlap hides NCCL",
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
        lines = [
            "# Wan2.2-Distill 256×256 TP=2 — Dual-Request all_reduce Overlap",
            "",
            f"TP=1 single baseline: **{tp1_single_s:.3f}s** → ceiling **{tp.get('tp1_ceiling_rps', 0):.4f} req/s**",
            "",
            "## Throughput (2 requests)",
            "",
            "| mode | wall (s) | req/s | vs TP=1 ceiling |",
            "|---|---:|---:|---:|",
        ]
        for label, key_s, key_rps in (
            ("single (TP=2)", "single_transformer_s", "single_rps"),
            ("dual back-to-back", "dual_back_to_back_s", "dual_back_to_back_rps"),
            ("dual ar-serial", "dual_ar_serial_s", "dual_ar_serial_rps"),
            ("dual ar-overlap", "dual_ar_overlap_s", "dual_ar_overlap_rps"),
            ("dual layer-serial", "dual_layer_serial_s", "dual_layer_serial_rps"),
            ("dual **layer-overlap**", "dual_layer_overlap_s", "dual_layer_overlap_rps"),
        ):
            wall = result.get(key_s)
            rps = tp.get(key_rps)
            if wall is None or rps is None:
                lines.append(f"| {label} | — | — | — |")
            else:
                pct = f"{100 * rps / tp1_ceiling_rps:.1f}%" if tp1_ceiling_rps else "—"
                lines.append(f"| {label} | {wall:.3f} | {rps:.4f} | {pct} |")

        def _fmt_sp(key: str) -> str:
            v = sp.get(key)
            return f"{v:.3f}×" if v is not None else "—"

        lines.extend([
            "",
            "## Speedup",
            "",
            f"- ar-overlap vs ar-serial: **{_fmt_sp('ar_overlap_vs_ar_serial')}**",
            f"- layer-overlap vs layer-serial: **{_fmt_sp('layer_overlap_vs_layer_serial')}**",
            f"- layer-overlap vs back-to-back: **{_fmt_sp('layer_overlap_vs_back_to_back')}**",
            f"- one-step layer-overlap vs serial: **{_fmt_sp('one_step_layer_overlap_vs_serial')}**",
            "",
            f"**Best dual throughput:** {tp.get('best_dual_rps') or 0:.4f} req/s "
            f"({tp.get('pct_of_tp1_ceiling') or 0:.1f}% of TP=1 ceiling)",
            "",
            "## Comm stats (ar-overlap run)",
            "",
        ])
        aos = ar_overlap_stats or {}
        lines.append(
            f"- all_reduce calls: {aos.get('all_reduce_calls', '—')} "
            f"({aos.get('all_reduce_overlap_windows', '—')} overlap windows)"
        )
        if err:
            lines.extend(["", f"**Error:** {err}"])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {md_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
