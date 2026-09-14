#!/usr/bin/env python3
"""Sweep unexcluded overlap root-cause fixes; compare 1-step phase-overlap throughput.

Each variant is measured against the same baseline (post sync-fix). Reports:
  - one_step_phase_overlap_ms, overlap_vs_serial_pct
  - projected dual-tenant req/s (2 / (overlap_ms/1000) per step, 4-step scale)
  - A2B1 L10 pair overlap_wall_ms (micro)

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_overlap_source_sweep.py \\
    --output_json save_results/optimization_study/tp2_overlap_source_sweep.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
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


def _load(name: str, path: Path):
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


def _wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _measure_phase_overlap(
    phase: Any,
    model: Any,
    ta: Any,
    tb: Any,
    ov: Any,
    prep_step: Any,
    *,
    reps: int,
) -> tuple[float, float]:
    serial_samples: list[float] = []
    overlap_samples: list[float] = []
    for _ in range(reps):
        prep_step()
        serial_samples.append(_wall_ms(lambda: phase.phase_pipeline_main_blocks(
            model, ta, tb,
            ov._bind_tenant, ov._ensure_block, ov._preload_blocks, ov._capture_ti_snap,
            overlap=False,
        )))
        prep_step()
        with no_sync_profiling(enabled=True):
            overlap_samples.append(_wall_ms(lambda: phase.phase_pipeline_main_blocks(
                model, ta, tb,
                ov._bind_tenant, ov._ensure_block, ov._preload_blocks, ov._capture_ti_snap,
                overlap=True,
            )))
    return statistics.median(serial_samples), statistics.median(overlap_samples)


def _measure_a2b1_pair(
    phase: Any,
    model: Any,
    ta: Any,
    tb: Any,
    ov: Any,
    orch: Any,
    layer: int,
) -> float:
    """A2B1 overlap pair wall (comm_p2 ∥ comp_p1), stream-scoped timing."""
    comm_t, comm_p, comp_t, comp_p = ta, 2, tb, 1

    def _prep_through(tenant: Any, ph: int) -> None:
        tenant.pipe_states = {}
        for p in range(1, ph):
            phase.run_tenant_phase(
                model, tenant, layer, p,
                ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
            )

    _prep_through(comm_t, comm_p)
    _prep_through(comp_t, comp_p)
    _prep_through(comm_t, comm_p)

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


def _apply_variant(phase: Any, cfg: dict[str, Any]) -> None:
    phase.configure_phase_pipeline(
        use_comm_p2p=cfg.get("use_comm_p2p", False),
        use_chunked_ar=bool(cfg.get("use_chunked_ar", False)),
        use_stream_ar_wait=bool(cfg.get("use_stream_ar_wait", False)),
        use_stream_fence=bool(cfg.get("use_stream_fence", False)),
        use_row_ready_event=bool(cfg.get("use_row_ready_event", False)),
    )


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("ph", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--output_json", default="save_results/optimization_study/tp2_overlap_source_sweep.json")
    args = parser.parse_args()

    variants: list[dict[str, Any]] = [
        {"id": "baseline", "label": "NCCL row AR + norm P2P (default)", "configure": {}},
        {"id": "nccl_comm", "label": "same as baseline", "configure": {"use_comm_p2p": False}},
        {"id": "G_chunked_ar", "label": "G chunked AR", "configure": {"use_chunked_ar": True}},
        {"id": "G_comm_p2p", "label": "G P2P COMM AR (no NCCL)", "configure": {"use_comm_p2p": True}},
        {"id": "G_chunked_p2p", "label": "G chunked + P2P COMM", "configure": {"use_chunked_ar": True, "use_comm_p2p": True}},
        {"id": "F_stream_ar_wait", "label": "F stream-scoped AR wait", "configure": {"use_stream_ar_wait": True}},
        {"id": "F_row_ready_event", "label": "F row_ready event before NCCL", "configure": {"use_row_ready_event": True}},
        {"id": "F_stream_fence", "label": "F default→orch stream fence", "configure": {"use_stream_fence": True}},
        {"id": "F_stream_wait_fence", "label": "F stream wait + fence", "configure": {"use_stream_ar_wait": True, "use_stream_fence": True}},
        {"id": "F_all_serialization", "label": "F A+B+C serialization fixes", "configure": {
            "use_stream_ar_wait": True, "use_row_ready_event": True, "use_stream_fence": True,
        }},
        {"id": "G_best_combo", "label": "G chunked+P2P + F stream wait", "configure": {
            "use_chunked_ar": True, "use_comm_p2p": True, "use_stream_ar_wait": True,
        }},
    ]

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

    def prep_step() -> None:
        sa.step_pre(step_index=0)
        sb.step_pre(step_index=0)
        ov._pre_infer_tenant(model, ta)
        ov._pre_infer_tenant(model, tb)

    # warmup
    _apply_variant(phase, {})
    prep_step()
    phase.phase_pipeline_main_blocks(
        model, ta, tb,
        ov._bind_tenant, ov._ensure_block, ov._preload_blocks, ov._capture_ti_snap,
        overlap=True,
    )

    device = torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0")
    orch = phase._get_orch(device)

    results: list[dict[str, Any]] = []
    baseline_overlap_ms: float | None = None

    for var in variants:
        _apply_variant(phase, var["configure"])
        serial_ms, overlap_ms = _measure_phase_overlap(
            phase, model, ta, tb, ov, prep_step, reps=args.reps,
        )
        a2b1_ms = _measure_a2b1_pair(phase, model, ta, tb, ov, orch, args.layer)
        if baseline_overlap_ms is None:
            baseline_overlap_ms = overlap_ms
        delta_pct = 100.0 * (overlap_ms - baseline_overlap_ms) / baseline_overlap_ms
        vs_serial_pct = 100.0 * (overlap_ms - serial_ms) / serial_ms
        dual_rps_4step = 2.0 / (overlap_ms * 4.0 / 1000.0) if overlap_ms > 0 else None
        rec = {
            "id": var["id"],
            "label": var["label"],
            "configure": var["configure"],
            "one_step_phase_serial_ms": serial_ms,
            "one_step_phase_overlap_ms": overlap_ms,
            "overlap_vs_serial_pct": vs_serial_pct,
            "a2b1_pair_overlap_ms": a2b1_ms,
            "delta_vs_baseline_overlap_pct": delta_pct,
            "projected_4step_dual_rps": dual_rps_4step,
            "nccl_env": {
                "NCCL_LAUNCH_MODE": os.environ.get("NCCL_LAUNCH_MODE"),
                "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
            },
        }
        results.append(rec)
        if is_main_process():
            print(f"[{var['id']}] overlap={overlap_ms:.1f}ms serial={serial_ms:.1f}ms "
                  f"A2B1={a2b1_ms:.1f}ms vs_base={delta_pct:+.2f}% vs_serial={vs_serial_pct:+.2f}%")

    out = {
        "baseline_overlap_ms": baseline_overlap_ms,
        "layer_a2b1": args.layer,
        "reps": args.reps,
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
