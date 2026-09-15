#!/usr/bin/env python3
"""Sweep independent NCCL communicator + CUDA graph overlap strategies (A2B1 L10).

Run:
  PYTHONPATH=$PWD PROFILING_DEBUG_LEVEL=0 torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_overlap_comm_opt_sweep.py \\
    --output_json save_results/nsys/tp2_comm_opt_sweep.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all

from scripts.disagg.tp_overlap_cuda_graph import OverlapOnceCudaGraph


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _sync() -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _cuda_event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _prep_through(phase_mod, model, tenant, layer, through, ov) -> None:
    tenant.pipe_states = {}
    for ph in range(1, through):
        phase_mod.run_tenant_phase(
            model, tenant, layer, ph, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )


def _warmup_phases(phase_mod, model, tenant, layer, ov) -> None:
    for ph in range(1, 7):
        tenant.pipe_states = {}
        for prep in range(1, ph):
            phase_mod.run_tenant_phase(
                model, tenant, layer, prep, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
            )
        phase_mod.run_tenant_phase(
            model, tenant, layer, ph, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
    torch.cuda.synchronize()


def _measure_overlap(
    phase_mod,
    model,
    ov,
    orch,
    *,
    layer: int,
    comm_t,
    comp_t,
    use_cuda_graph_once: bool = False,
    capture_staggered: bool = False,
) -> dict[str, Any]:
    _prep_through(phase_mod, model, comm_t, layer, 2, ov)
    orch.enabled = False
    orch.set_peer(None)
    _prep_through(phase_mod, model, comp_t, layer, 1, ov)
    comp_t.pipe_states = {}

    once_graph = OverlapOnceCudaGraph()
    graph_meta: dict[str, Any] = {}

    def _once() -> None:
        orch.enabled = True
        orch.set_peer(phase_mod._make_peer_runner(
            model, comp_t, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        phase_mod.run_tenant_phase(
            model, comm_t, layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
        orch.drain_peer()
        orch.enabled = False
        orch.set_peer(None)

    if capture_staggered:
        orch.enabled = True
        orch.set_peer(phase_mod._make_peer_runner(
            model, comp_t, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        graph_meta = orch.capture_peer_pump_tail_graph()
        orch.enabled = False
        orch.set_peer(None)
        _sync()

    overlap_ms: float
    if use_cuda_graph_once:
        ok = once_graph.try_capture(_once, warmup_iters=2)
        graph_meta["overlap_once_graph"] = {
            "captured": ok,
            "error": once_graph.capture_error,
        }
        if ok:
            overlap_ms = _cuda_event_ms(once_graph.replay)
        else:
            overlap_ms = _cuda_event_ms(_once)
    else:
        overlap_ms = _cuda_event_ms(_once)

    return {"overlap_ms": overlap_ms, "graph": graph_meta}


def _measure_comm_comp(phase_mod, model, tenant_a, tenant_b, layer, ov, orch) -> dict[str, float]:
    _prep_through(phase_mod, model, tenant_a, layer, 2, ov)
    orch.enabled = False
    orch.set_peer(None)
    _sync()
    t0 = time.perf_counter()
    phase_mod.run_tenant_phase(
        model, tenant_a, layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    )
    _sync()
    comm_ms = (time.perf_counter() - t0) * 1000.0

    _prep_through(phase_mod, model, tenant_b, layer, 1, ov)
    tenant_b.pipe_states = {}
    _sync()
    t0 = time.perf_counter()
    phase_mod.run_tenant_phase(
        model, tenant_b, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    )
    _sync()
    comp_ms = (time.perf_counter() - t0) * 1000.0
    return {"comm_alone_ms": comm_ms, "comp_alone_ms": comp_ms}


VARIANTS: list[tuple[str, dict[str, bool]]] = [
    ("baseline", {}),
    ("row_ar_nccl", {"use_row_ar_nccl_group": True}),
    ("high_pri_comm", {"use_high_priority_comm_stream": True}),
    ("staggered_pump", {"use_staggered_pump": True}),
    ("cuda_graph_staggered", {"use_cuda_graph_staggered_pump": True}),
    ("row_ar_nccl+cuda_graph", {
        "use_row_ar_nccl_group": True,
        "use_cuda_graph_staggered_pump": True,
    }),
    ("row_ar_nccl+high_pri", {
        "use_row_ar_nccl_group": True,
        "use_high_priority_comm_stream": True,
    }),
    ("all_three", {
        "use_row_ar_nccl_group": True,
        "use_high_priority_comm_stream": True,
        "use_cuda_graph_staggered_pump": True,
    }),
]


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load("overlap", here / "run_phase3_dual_overlap_bench.py")
    phase_mod = _load("phase_pipe", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--try_overlap_once_graph", action="store_true", default=False)
    parser.add_argument("--output_json", default="save_results/nsys/tp2_comm_opt_sweep.json")
    args = parser.parse_args()

    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="test",
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

    payload_a = p1._prepare_inputs_cache(
        config=config, cache_path=Path(args.inputs_cache), prompt=ns.prompt,
        image_path=ns.image_path, seed=42, force=False, task="i2v", negative_prompt="",
    )
    payload_a = p1._prepare_payload_on_device(payload_a)
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
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
    for t, p, s in ((tenant_a, payload_a, scheduler_a), (tenant_b, payload_b, scheduler_b)):
        t.scheduler.prepare(
            seed=int(p["seed"]),
            latent_shape=p["latent_shape"],
            image_encoder_output=p["image_encoder_output"],
        )
        t.inputs = p["inputs"]
        s.step_pre(step_index=0)
        ov._pre_infer_tenant(model, t)

    wan, ti = ov._bind_tenant(model, tenant_a)
    ov._preload_blocks(tenant_a, wan, ti, len(wan.transformer_weights.blocks))

    layer = args.layer
    _warmup_phases(phase_mod, model, tenant_a, layer, ov)
    _warmup_phases(phase_mod, model, tenant_b, layer, ov)

    orch = phase_mod._get_orch(torch.device(f"cuda:{dist.get_rank()}"))
    results: list[dict[str, Any]] = []

    for name, cfg in VARIANTS:
        phase_mod.configure_phase_pipeline(
            use_comm_p2p=False,
            use_stream_ar_wait=True,
            use_row_ready_event=True,
            use_row_ar_nccl_group=bool(cfg.get("use_row_ar_nccl_group", False)),
            use_high_priority_comm_stream=bool(cfg.get("use_high_priority_comm_stream", False)),
            use_cuda_graph_staggered_pump=bool(cfg.get("use_cuda_graph_staggered_pump", False)),
            use_staggered_pump=bool(cfg.get("use_staggered_pump", False)),
        )
        if cfg.get("use_row_ar_nccl_group"):
            from lightx2v.common.ops.tp_row_ar_nccl import get_row_ar_nccl_group
            wan0, _ = ov._bind_tenant(model, tenant_a)
            blk = wan0.transformer_weights.blocks[layer]
            get_row_ar_nccl_group(blk.compute_phases[0].self_attn_o.tp_group)
            _sync()
        _sync()
        baselines = _measure_comm_comp(phase_mod, model, tenant_a, tenant_b, layer, ov, orch)
        ov_row = _measure_overlap(
            phase_mod, model, ov, orch,
            layer=layer, comm_t=tenant_a, comp_t=tenant_b,
            capture_staggered=bool(cfg.get("use_cuda_graph_staggered_pump", False)),
        )
        once_row: dict[str, Any] = {}
        if args.try_overlap_once_graph and name == "baseline":
            once_row = _measure_overlap(
                phase_mod, model, ov, orch,
                layer=layer, comm_t=tenant_a, comp_t=tenant_b,
                use_cuda_graph_once=True,
            )

        ideal = max(baselines["comm_alone_ms"], baselines["comp_alone_ms"])
        entry = {
            "variant": name,
            "configure": cfg,
            **baselines,
            "ideal_max_ms": ideal,
            **ov_row,
            "overlap_minus_ideal_ms": ov_row["overlap_ms"] - ideal,
        }
        if once_row:
            entry["overlap_once_graph_ms"] = once_row["overlap_ms"]
            entry["overlap_once_graph_meta"] = once_row.get("graph", {})
        results.append(entry)
        if is_main_process():
            print(
                f"{name}: overlap={ov_row['overlap_ms']:.2f} ms "
                f"(ideal {ideal:.2f}, delta {entry['overlap_minus_ideal_ms']:+.2f}) "
                f"graph={ov_row.get('graph', {})}"
            )

    out = {"layer": layer, "pair": "A2B1", "variants": results}
    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
