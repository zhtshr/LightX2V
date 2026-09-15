#!/usr/bin/env python3
"""Nsight capture: COMP-alone vs COMM+COMP overlap (A2B1 @ layer 10).

Run (rank0 GPU timeline):
  nsys profile -o save_results/nsys/tp2_overlap \\
    --trace=cuda,nvtx,cublas,osrt --cudabacktrace=sync \\
    -c cudaProfilerApi --force-overwrite=true \\
    torchrun --standalone --nproc_per_node=2 \\
      scripts/disagg/run_tp2_overlap_nsys.py
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
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _cuda_event_ms(fn) -> float:
    """GPU-scoped timing without host sync inside the timed region."""
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


def _run_comp_alone(phase_mod, model, tenant, layer, comp_p, ov) -> float:
    _prep_through(phase_mod, model, tenant, layer, comp_p, ov)
    nvtx.range_push(f"COMP_alone_p{comp_p}")
    ms = _wall_ms(lambda: phase_mod.run_tenant_phase(
        model, tenant, layer, comp_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    ))
    nvtx.range_pop()
    return ms


def _run_comm_alone(phase_mod, model, tenant, layer, comm_p, ov) -> float:
    _prep_through(phase_mod, model, tenant, layer, comm_p, ov)
    orch = phase_mod._get_orch(torch.device(f"cuda:{dist.get_rank()}"))
    orch.enabled = False
    orch.set_peer(None)
    nvtx.range_push(f"COMM_alone_p{comm_p}")
    ms = _wall_ms(lambda: phase_mod.run_tenant_phase(
        model, tenant, layer, comm_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    ))
    nvtx.range_pop()
    return ms


def _run_overlap_pair(
    phase_mod, model, ov, orch, *, layer, comm_t, comm_p, comp_t, comp_p,
) -> float:
    _prep_through(phase_mod, model, comm_t, layer, comm_p, ov)
    orch.enabled = False
    orch.set_peer(None)
    _prep_through(phase_mod, model, comp_t, layer, comp_p, ov)
    comp_t.pipe_states = {}

    prof_ms: dict[str, float] = {}

    def _once() -> None:
        orch.enabled = True
        orch.set_peer(phase_mod._make_peer_runner(
            model, comp_t, layer, comp_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        phase_mod.run_tenant_phase(
            model, comm_t, layer, comm_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
        orch.drain_peer()
        orch.enabled = False
        orch.set_peer(None)

    nvtx.range_push(f"OVERLAP_comm_p{comm_p}_comp_p{comp_p}")
    ms = _cuda_event_ms(_once)
    nvtx.range_pop()
    return ms


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
    parser.add_argument("--tp_norm_p2p", action="store_true", default=True)
    parser.add_argument("--no_tp_norm_p2p", action="store_false", dest="tp_norm_p2p")
    parser.add_argument("--use_stream_ar_wait", action="store_true", default=True)
    parser.add_argument("--no_stream_ar_wait", action="store_false", dest="use_stream_ar_wait")
    parser.add_argument("--use_row_ready_event", action="store_true", default=False)
    parser.add_argument(
        "--use_chunked_ar",
        action="store_true",
        default=False,
        help="Interleave row NCCL chunks with peer pump micro-steps",
    )
    parser.add_argument(
        "--use_defer_attn_post_ar",
        action="store_true",
        default=False,
        help="Run attn/o_linear after row NCCL completes (reduce BW serial tail)",
    )
    parser.add_argument(
        "--use_delay_nccl_until_prep",
        action="store_true",
        default=False,
        help="Finish q/k prep before launching row NCCL; overlap attn+o with NCCL",
    )
    parser.add_argument(
        "--use_row_ar_nccl_group",
        action="store_true",
        default=False,
        help="Use dedicated NCCL process group for row AR",
    )
    parser.add_argument(
        "--use_staggered_pump",
        action="store_true",
        default=False,
        help="Prep q/k before NCCL; overlap attn+o with row AR (reduces BW serial tail)",
    )
    parser.add_argument(
        "--use_cuda_graph_staggered_pump",
        action="store_true",
        default=False,
        help="Staggered pump + CUDA graph replay for attn+o tail (experimental)",
    )
    parser.add_argument("--output_json", default="save_results/nsys/tp2_overlap_wall.json")
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
    if args.tp_norm_p2p:
        config["tp_norm_p2p"] = True
    seed_all(42)
    p1._init_distributed(config)

    phase_mod.configure_phase_pipeline(
        use_comm_p2p=False,
        use_chunked_ar=args.use_chunked_ar,
        use_stream_ar_wait=args.use_stream_ar_wait,
        use_row_ready_event=args.use_row_ready_event,
        use_defer_attn_post_ar=args.use_defer_attn_post_ar,
        use_delay_nccl_until_prep=args.use_delay_nccl_until_prep,
        use_row_ar_nccl_group=args.use_row_ar_nccl_group,
        use_staggered_pump=args.use_staggered_pump,
        use_cuda_graph_staggered_pump=args.use_cuda_graph_staggered_pump,
    )

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

    # CPU sync across ranks before capture
    if dist.is_initialized():
        dist.barrier()

    torch.cuda.profiler.start()
    nvtx.range_push("tp2_overlap_capture")
    comm_ms = _run_comm_alone(phase_mod, model, tenant_a, layer, 2, ov)
    comp_ms = _run_comp_alone(phase_mod, model, tenant_b, layer, 1, ov)
    overlap_ms = _run_overlap_pair(
        phase_mod, model, ov, orch,
        layer=layer, comm_t=tenant_a, comm_p=2, comp_t=tenant_b, comp_p=1,
    )
    nvtx.range_pop()
    torch.cuda.profiler.stop()

    if dist.is_initialized():
        dist.barrier()

    out = {
        "layer": layer,
        "pair": "A2B1",
        "tp_norm_p2p": bool(config.get("tp_norm_p2p", False)),
        "use_stream_ar_wait": args.use_stream_ar_wait,
        "use_row_ready_event": args.use_row_ready_event,
        "use_chunked_ar": args.use_chunked_ar,
        "use_defer_attn_post_ar": args.use_defer_attn_post_ar,
        "use_delay_nccl_until_prep": args.use_delay_nccl_until_prep,
        "use_row_ar_nccl_group": args.use_row_ar_nccl_group,
        "use_staggered_pump": args.use_staggered_pump,
        "use_cuda_graph_staggered_pump": args.use_cuda_graph_staggered_pump,
        "comm_alone_ms": comm_ms,
        "comp_alone_ms": comp_ms,
        "serial_sum_ms": comm_ms + comp_ms,
        "ideal_max_ms": max(comm_ms, comp_ms),
        "overlap_ms": overlap_ms,
        "overlap_minus_ideal_ms": overlap_ms - max(comm_ms, comp_ms),
        "overlap_minus_serial_ms": overlap_ms - (comm_ms + comp_ms),
        "rank": dist.get_rank() if dist.is_initialized() else 0,
    }

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        print(f"wrote {path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
