#!/usr/bin/env python3
"""256×256 TP=2 end-to-end suite: all validated dual-card optimization modes.

Runs single-request baseline, dual back-to-back, and every overlap strategy that
has been implemented and deemed worth measuring (excludes known-bad variants:
chunked AR, comm P2P, independent NCCL group, defer-attn, etc.).

Phase-overlap uses production-tuned flags: tp_norm_p2p + stream AR wait + row_ready event.

  CUDA_VISIBLE_DEVICES=0,1 PROFILING_DEBUG_LEVEL=0 \\
    torchrun --standalone --nproc_per_node=2 \\
      scripts/disagg/run_wan22_256_tp2_e2e_suite.py \\
      --output_json save_results/optimization_study/wan22_256_tp2_e2e_suite.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
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
        torch.cuda.synchronize()


def _load_tp1_single_s(study_dir: Path) -> float | None:
    lat_path = study_dir / "wan22_256_tp1_latency.json"
    if lat_path.is_file():
        return float(json.loads(lat_path.read_text())["transformer_compute_s"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_json",
        default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json",
    )
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt",
    )
    parser.add_argument("--output_json", default="save_results/optimization_study/wan22_256_tp2_e2e_suite.json")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=0)
    parser.add_argument("--seed_a", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    args = parser.parse_args()

    here = Path(__file__).parent
    study_dir = Path("/root/zht/LightX2V/save_results/optimization_study")
    p1 = _load_module("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load_module("overlap", here / "run_phase3_dual_overlap_bench.py")
    fine = _load_module("fine", here / "tp_fine_overlap.py")
    micro = _load_module("micro", here / "tp_micro_overlap.py")
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
        tensor_p_size=2,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt=(
            "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量"
        ),
    )
    config = p1._load_config(ns)
    config["tp_norm_p2p"] = True
    seed_all(args.seed_a)
    p1._init_distributed(config)

    phase.configure_phase_pipeline(
        use_stream_ar_wait=True,
        use_row_ready_event=True,
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
    tp1_single_s = _load_tp1_single_s(study_dir)
    tp1_ceiling_rps = _rps(2, tp1_single_s)

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
    phase_kwargs = dict(
        bind_tenant=ov._bind_tenant,
        ensure_block=ov._ensure_block,
        preload_blocks=ov._preload_blocks,
        pre_infer_tenant=ov._pre_infer_tenant,
        finish_step_tenant=ov._finish_step_tenant,
        capture_ti_snap=ov._capture_ti_snap,
        time_fn=ov._time_fn,
    )

    modes: list[dict[str, Any]] = []
    err: str | None = None

    def _run_mode(mode_id: str, label: str, fn) -> None:
        nonlocal err
        wall_s: float | None = None
        stats: dict[str, Any] = {}
        mode_err: str | None = None
        try:
            _cuda_gc()
            out = fn()
            if isinstance(out, tuple):
                wall_s, stats = out[0], (out[1] if len(out) > 1 else {})
            else:
                wall_s = out
        except Exception as exc:  # noqa: BLE001
            mode_err = str(exc)
            err = err or mode_err
            if is_main_process():
                import traceback
                print(f"[{mode_id}] FAILED: {exc}")
                traceback.print_exc()
        modes.append({
            "id": mode_id,
            "label": label,
            "wall_s": wall_s,
            "req_s": 1 if "single" in mode_id else 2,
            "rps": _rps(1 if "single" in mode_id else 2, wall_s),
            "vs_tp1_ceiling_pct": (
                100.0 * _rps(2, wall_s) / tp1_ceiling_rps
                if "single" not in mode_id and wall_s and tp1_ceiling_rps
                else None
            ),
            "stats": stats,
            "error": mode_err,
        })

    for _ in range(args.warmup):
        ov._run_denoise_serial(model, scheduler_a, payload_a)
    _cuda_gc()

    # Single-request baselines
    _run_mode(
        "tp2_single",
        "TP=2 single request (transformer only)",
        lambda: ov._time_fn(lambda: ov._run_denoise_serial(model, scheduler_a, payload_a)),
    )
    _run_mode(
        "dual_b2b",
        "Dual back-to-back (2× serial infer, no overlap)",
        lambda: ov._time_fn(lambda: (
            ov._run_denoise_serial(model, scheduler_a, payload_a),
            ov._run_denoise_serial(model, scheduler_b, payload_b),
        )),
    )

    # AR / fine / micro overlap families
    _run_mode(
        "dual_ar_overlap",
        "Dual AR-overlap (SP-style cross_ffn during peer self_attn AR)",
        lambda: ov._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps,
        ),
    )
    _run_mode(
        "dual_fine_serial",
        "Dual fine-serial (per-AR pump, overlap off)",
        lambda: fine.run_dual_fine_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps, **fine_kwargs,
        ),
    )
    _run_mode(
        "dual_fine_overlap",
        "Dual fine-overlap (per-AR async + pump cross_ffn)",
        lambda: fine.run_dual_fine_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps, **fine_kwargs,
        ),
    )
    _run_mode(
        "dual_micro_serial",
        "Dual micro-serial (comp-comm micro-phase, overlap off)",
        lambda: micro.run_dual_micro_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps, **micro_kwargs,
        ),
    )
    _run_mode(
        "dual_micro_overlap",
        "Dual micro-overlap (comp-comm micro-phase alignment)",
        lambda: micro.run_dual_micro_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps, **micro_kwargs,
        ),
    )

    # 6-phase pipeline (production flags)
    _run_mode(
        "dual_phase_serial",
        "Dual 6-phase serial (COMM then COMP per pair)",
        lambda: phase.run_dual_phase_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=False, steps=measure_steps, **phase_kwargs,
        ),
    )
    _run_mode(
        "dual_phase_overlap",
        "Dual 6-phase overlap (best: P2P norm + row_ready + stream AR wait)",
        lambda: phase.run_dual_phase_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=True, steps=measure_steps, **phase_kwargs,
        ),
    )

    dual_modes = [m for m in modes if m["id"] != "tp2_single" and m["rps"]]
    best_dual = max(dual_modes, key=lambda m: m["rps"] or 0.0) if dual_modes else None
    b2b = next((m for m in modes if m["id"] == "dual_b2b"), None)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": "256x256",
        "config_json": args.config_json,
        "tensor_p_size": 2,
        "tp_norm_p2p": True,
        "phase_flags": {
            "use_stream_ar_wait": True,
            "use_row_ready_event": True,
        },
        "profiling_debug_level": os.environ.get("PROFILING_DEBUG_LEVEL"),
        "measure_steps": measure_steps,
        "tp1_single_s_cached": tp1_single_s,
        "tp1_parallel_ceiling_rps": tp1_ceiling_rps,
        "modes": modes,
        "summary": {
            "best_dual_mode": best_dual["id"] if best_dual else None,
            "best_dual_rps": best_dual["rps"] if best_dual else None,
            "dual_b2b_rps": b2b["rps"] if b2b else None,
            "best_vs_b2b_pct": (
                100.0 * best_dual["rps"] / b2b["rps"]
                if best_dual and b2b and best_dual["rps"] and b2b["rps"]
                else None
            ),
            "best_vs_tp1_ceiling_pct": best_dual.get("vs_tp1_ceiling_pct") if best_dual else None,
        },
        "error": err,
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

        def _pct(v: float | None) -> str:
            if v is None or not tp1_ceiling_rps:
                return "—"
            return f"{v:.1f}%"

        lines = [
            "# Wan2.2 256×256 TP=2 — End-to-End Optimization Suite",
            "",
            f"measure_steps: **{measure_steps}** | tp_norm_p2p: **True** | "
            f"phase: stream_ar_wait + row_ready_event",
            "",
            f"TP=1 single (cached): **{tp1_single_s:.3f}s** | "
            f"2×单卡 ceiling: **{tp1_ceiling_rps:.4f} req/s**"
            if tp1_single_s else f"2×单卡 ceiling: **{tp1_ceiling_rps:.4f} req/s**",
            "",
            "## All modes (transformer denoise only)",
            "",
            "| mode | wall (s) | req/s | vs 2×TP=1 | vs b2b |",
            "|---|---:|---:|---:|---:|",
        ]
        b2b_rps = b2b["rps"] if b2b else None
        for m in modes:
            wall = m["wall_s"]
            rps = m["rps"]
            if wall is None or rps is None:
                lines.append(f"| {m['label']} | — | — | — | — |")
                continue
            vs_b2b = f"{100 * rps / b2b_rps:.1f}%" if b2b_rps and m["id"] != "tp2_single" else "—"
            lines.append(
                f"| {m['label']} | {wall:.3f} | {rps:.4f} | {_pct(m.get('vs_tp1_ceiling_pct'))} | {vs_b2b} |"
            )

        sm = result["summary"]
        lines.extend([
            "",
            "## Summary",
            "",
            f"- **Best dual mode:** {sm.get('best_dual_mode', '—')} "
            f"({sm.get('best_dual_rps', 0):.4f} req/s)"
            if sm.get("best_dual_rps") else "- **Best dual mode:** —",
            f"- Best vs dual b2b: **{sm.get('best_vs_b2b_pct', 0):.1f}%**"
            if sm.get("best_vs_b2b_pct") else "- Best vs dual b2b: —",
            f"- Best vs 2×TP=1 ceiling: **{sm.get('best_vs_tp1_ceiling_pct', 0):.1f}%**"
            if sm.get("best_vs_tp1_ceiling_pct") else "- Best vs 2×TP=1 ceiling: —",
        ])
        if err:
            lines.extend(["", f"**Errors:** {err}"])

        md = out.with_suffix(".md")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")
        print(f"wrote {md}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
