#!/usr/bin/env python3
"""Sweep serialization-fix variants for TP=2 NCCL row AR overlap (production comm).

Each variant toggles one hypothesized cause:
  A  use_stream_ar_wait   — comm_stream.synchronize() instead of work.wait()
  B  use_row_ready_event — compute→comm cuda event + record_stream before NCCL
  C  use_stream_fence    — fence default stream into orch streams at step start
  D  no_sync in _handle_ar (always on in pipeline; baseline includes it)
  E  NCCL_LAUNCH_MODE=PARALLEL (must be set before dist init; run via env)

Metrics per variant (A2B1 L{layer}):
  comm_alone_ms, comp_alone_ms, ideal_max_ms, overlap_pair_ms, overlap/ideal

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_serialization_fix_sweep.py \\
    --output_json save_results/optimization_study/tp2_serialization_fix_sweep.json

  NCCL_LAUNCH_MODE=PARALLEL torchrun ...  # or use --include_nccl_parallel subprocess
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.profiler import no_sync_profiling
from lightx2v.utils.utils import is_main_process, seed_all


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sync_orch(orch: Any) -> None:
    orch.comm_stream.synchronize()
    orch.compute_stream.synchronize()


def _sync_all() -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _apply_configure(phase: Any, cfg: dict[str, Any]) -> None:
    phase.configure_phase_pipeline(
        use_comm_p2p=bool(cfg.get("use_comm_p2p", False)),
        use_chunked_ar=bool(cfg.get("use_chunked_ar", False)),
        use_stream_ar_wait=bool(cfg.get("use_stream_ar_wait", False)),
        use_stream_fence=bool(cfg.get("use_stream_fence", False)),
        use_row_ready_event=bool(cfg.get("use_row_ready_event", False)),
    )


def _measure_comm_alone(phase: Any, model: Any, ta: Any, ov: Any, layer: int) -> float:
    ta.pipe_states = {}
    phase.run_tenant_phase(
        model, ta, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    )
    _sync_all()
    t0 = time.perf_counter()
    phase.run_tenant_phase(
        model, ta, layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    )
    _sync_all()
    return (time.perf_counter() - t0) * 1000.0


def _measure_comp_alone(phase: Any, model: Any, tb: Any, ov: Any, layer: int) -> float:
    tb.pipe_states = {}
    _sync_all()
    t0 = time.perf_counter()
    phase.run_tenant_phase(
        model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    )
    _sync_all()
    return (time.perf_counter() - t0) * 1000.0


def _measure_a2b1_overlap(
    phase: Any,
    model: Any,
    ta: Any,
    tb: Any,
    ov: Any,
    orch: Any,
    layer: int,
) -> float:
    comm_t, comm_p, comp_t, comp_p = ta, 2, tb, 1

    def prep(tenant: Any, ph: int) -> None:
        tenant.pipe_states = {}
        for p in range(1, ph):
            phase.run_tenant_phase(
                model, tenant, layer, p,
                ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
            )

    prep(comm_t, comm_p)
    prep(comp_t, comp_p)
    prep(comm_t, comm_p)

    orch.enabled = True
    orch.set_peer(phase._make_peer_runner(
        model, comp_t, layer, comp_p,
        ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    ))
    _sync_orch(orch)
    t0 = time.perf_counter()
    with no_sync_profiling(enabled=True):
        phase.run_tenant_phase(
            model, comm_t, layer, comm_p,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
        orch.drain_peer()
    orch.enabled = False
    orch.set_peer(None)
    _sync_orch(orch)
    return (time.perf_counter() - t0) * 1000.0


def _run_variant(
    phase: Any,
    model: Any,
    ta: Any,
    tb: Any,
    ov: Any,
    orch: Any,
    layer: int,
    reps: int,
) -> dict[str, float]:
    comm_samples: list[float] = []
    comp_samples: list[float] = []
    overlap_samples: list[float] = []
    for _ in range(reps):
        comm_samples.append(_measure_comm_alone(phase, model, ta, ov, layer))
        comp_samples.append(_measure_comp_alone(phase, model, tb, ov, layer))
        overlap_samples.append(_measure_a2b1_overlap(phase, model, ta, tb, ov, orch, layer))

    comm_m = statistics.median(comm_samples)
    comp_m = statistics.median(comp_samples)
    overlap_m = statistics.median(overlap_samples)
    ideal = max(comm_m, comp_m)
    return {
        "comm_alone_ms": comm_m,
        "comp_alone_ms": comp_m,
        "ideal_max_ms": ideal,
        "overlap_pair_ms": overlap_m,
        "overlap_vs_ideal_ratio": overlap_m / max(ideal, 0.001),
        "overlap_vs_ideal_pct": 100.0 * (overlap_m - ideal) / max(ideal, 0.001),
        "gap_ms": overlap_m - ideal,
    }


VARIANTS: list[dict[str, Any]] = [
    {"id": "baseline", "label": "baseline (no_sync in _handle_ar)", "configure": {}},
    {"id": "A_stream_ar_wait", "label": "A: comm_stream sync vs work.wait", "configure": {"use_stream_ar_wait": True}},
    {"id": "B_row_ready_event", "label": "B: row_ready cuda event + record_stream", "configure": {"use_row_ready_event": True}},
    {"id": "C_stream_fence", "label": "C: default→orch stream fence", "configure": {"use_stream_fence": True}},
    {"id": "A+B", "label": "A+B combined", "configure": {"use_stream_ar_wait": True, "use_row_ready_event": True}},
    {"id": "A+C", "label": "A+C combined", "configure": {"use_stream_ar_wait": True, "use_stream_fence": True}},
    {"id": "B+C", "label": "B+C combined", "configure": {"use_row_ready_event": True, "use_stream_fence": True}},
    {"id": "A+B+C", "label": "A+B+C all fixes", "configure": {
        "use_stream_ar_wait": True, "use_row_ready_event": True, "use_stream_fence": True,
    }},
]


def _setup_model(args: argparse.Namespace, p1: Any, ov: Any) -> tuple[Any, Any, Any, Any, Any]:
    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="t",
        seed=42,
        seq_p_size=1,
        tensor_p_size=2,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="",
    )
    config = p1._load_config(ns)
    config["tp_norm_p2p"] = True
    seed_all(42)
    p1._init_distributed(config)

    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(
            config=config,
            cache_path=Path(args.inputs_cache),
            prompt=ns.prompt,
            image_path=ns.image_path,
            seed=42,
            force=False,
            task="i2v",
            negative_prompt="",
        )
    )
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    sa, sb = WanScheduler(config), WanScheduler(config)
    ta = ov.TenantCtx("A", sa, payload_a["inputs"])
    tb = ov.TenantCtx("B", sb, payload_b["inputs"])
    for t, p, s in ((ta, payload_a, sa), (tb, payload_b, sb)):
        t.scheduler.prepare(
            seed=int(p["seed"]),
            latent_shape=p["latent_shape"],
            image_encoder_output=p["image_encoder_output"],
        )
        t.inputs = p["inputs"]
        s.step_pre(step_index=0)
        ov._pre_infer_tenant(model, t)
    return model, ta, tb, ov, config


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("ph", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--variant_id", default="", help="run single variant only")
    parser.add_argument("--output_json", default="save_results/optimization_study/tp2_serialization_fix_sweep.json")
    args = parser.parse_args()

    model, ta, tb, ov, _config = _setup_model(args, p1, ov)
    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = phase._get_orch(device)

    # warmup baseline
    _apply_configure(phase, {})
    _measure_a2b1_overlap(phase, model, ta, tb, ov, orch, args.layer)

    variants = VARIANTS
    if args.variant_id:
        variants = [v for v in VARIANTS if v["id"] == args.variant_id]
        if not variants:
            raise SystemExit(f"unknown variant_id={args.variant_id!r}")

    results: list[dict[str, Any]] = []
    baseline_overlap: float | None = None

    for var in variants:
        _apply_configure(phase, var["configure"])
        metrics = _run_variant(phase, model, ta, tb, ov, orch, args.layer, args.reps)
        if baseline_overlap is None:
            baseline_overlap = metrics["overlap_pair_ms"]
        rec = {
            "id": var["id"],
            "label": var["label"],
            "configure": var["configure"],
            **metrics,
            "delta_overlap_vs_baseline_pct": 100.0 * (metrics["overlap_pair_ms"] - baseline_overlap) / baseline_overlap,
            "nccl_env": {
                "NCCL_LAUNCH_MODE": os.environ.get("NCCL_LAUNCH_MODE"),
                "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
            },
        }
        results.append(rec)
        if is_main_process():
            print(
                f"[{var['id']}] overlap={metrics['overlap_pair_ms']:.2f}ms "
                f"ideal={metrics['ideal_max_ms']:.2f}ms "
                f"ratio={metrics['overlap_vs_ideal_ratio']:.2f}x "
                f"comm={metrics['comm_alone_ms']:.2f} comp={metrics['comp_alone_ms']:.2f} "
                f"vs_base={rec['delta_overlap_vs_baseline_pct']:+.1f}%"
            )

    out = {
        "baseline_overlap_ms": baseline_overlap,
        "layer_a2b1": args.layer,
        "reps": args.reps,
        "policy": {"row_comm_ar": "NCCL", "norm_ar": "P2P"},
        "variants": results,
    }

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {path}")
        print("结论文档: save_results/nsys/tp2_overlap_root_cause.md §6")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
