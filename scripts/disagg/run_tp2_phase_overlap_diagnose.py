#!/usr/bin/env python3
"""Diagnose 6-phase pipeline: COMM purity, stream overlap, scheduling overhead."""

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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Evt:
    def __init__(self) -> None:
        self._s: dict[str, torch.cuda.Event] = {}

    def start(self, k: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._s[k] = e

    def stop(self, k: str) -> float:
        if k not in self._s:
            return 0.0
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        s = self._s.pop(k)
        torch.cuda.synchronize()
        return s.elapsed_time(end)


def _cuda_overlap_microbench(device: torch.device, group: Any) -> dict[str, float]:
    """Synthetic: async NCCL AR on comm_stream vs GEMM on compute_stream."""
    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    ar_buf = torch.randn(8192, 5120, device=device, dtype=torch.float16)
    gemm = torch.randn(4096, 4096, device=device, dtype=torch.float16)

    def _one(serial: bool) -> float:
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        if serial:
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=group, async_op=False)
            for _ in range(3):
                torch.matmul(gemm, gemm)
        else:
            with torch.cuda.stream(comm_stream):
                work = dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=group, async_op=True)
            with torch.cuda.stream(compute_stream):
                for _ in range(3):
                    torch.matmul(gemm, gemm)
            if work is not None:
                work.wait()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1)

    serial_ms = _one(True)
    overlap_ms = _one(False)
    return {
        "synthetic_ar_gemm_serial_ms": serial_ms,
        "synthetic_ar_gemm_overlap_ms": overlap_ms,
        "synthetic_overlap_ratio": serial_ms / overlap_ms if overlap_ms > 0 else 0.0,
    }


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load("overlap", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("phase_pipe", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output_json", default="save_results/optimization_study/wan22_256_tp2_phase_diagnose.json")
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
    seed_all(42)
    p1._init_distributed(config)

    device = torch.device(f"cuda:{dist.get_rank()}")
    group = dist.group.WORLD
    out: dict[str, Any] = {"layer": args.layer, "rank": dist.get_rank()}
    out["synthetic"] = _cuda_overlap_microbench(device, group)

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

    k = args.layer
    tenant_a.pipe_states = {}
    tenant_b.pipe_states = {}
    phase.run_tenant_phase(model, tenant_a, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)

    ev = _Evt()

    def _time_phase(tenant: Any, block: int, ph: int, tag: str) -> dict[str, float]:
        torch.cuda.synchronize()
        t_wall0 = time.perf_counter()
        ev.start(tag)
        phase.run_tenant_phase(model, tenant, block, ph, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        cuda_ms = ev.stop(tag)
        wall_ms = (time.perf_counter() - t_wall0) * 1000.0
        return {"cuda_ms": cuda_ms, "wall_ms": wall_ms, "python_overhead_ms": wall_ms - cuda_ms}

    # reset scratch for B
    tenant_b.pipe_states = {}

    t_comm2 = _time_phase(tenant_a, k, 2, "comm_p2")
    tenant_a.pipe_states = {}
    phase.run_tenant_phase(model, tenant_a, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    t_comp1 = _time_phase(tenant_b, k, 1, "comp_p1")

    # serial pair A2B1
    tenant_a.pipe_states = {}
    tenant_b.pipe_states = {}
    phase.run_tenant_phase(model, tenant_a, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    phase.run_tenant_phase(model, tenant_a, k, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    phase.run_tenant_phase(model, tenant_b, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    torch.cuda.synchronize()
    pair_serial_wall_ms = (time.perf_counter() - t0) * 1000.0

    # overlap pair A2B1 with cuda events on comm vs comp portion
    tenant_a.pipe_states = {}
    tenant_b.pipe_states = {}
    phase.run_tenant_phase(model, tenant_a, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    orch = phase._get_orch(device)

    def _comp() -> None:
        ev.start("overlap_comp")
        phase.run_tenant_phase(model, tenant_b, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        ev.stop("overlap_comp")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    orch.enabled = True
    orch.reset_comp(_comp)
    ev.start("overlap_comm_phase")
    phase.run_tenant_phase(model, tenant_a, k, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    overlap_comm_cuda_ms = ev.stop("overlap_comm_phase")
    if not orch.comp_done:
        _comp()
    orch.enabled = False
    torch.cuda.synchronize()
    pair_overlap_wall_ms = (time.perf_counter() - t0) * 1000.0

    import scripts.disagg.tp_micro_overlap as micro

    # sa_o breakdown: O matmul only vs AR only (profile segment split)
    tenant_a.pipe_states = {}
    phase.run_tenant_phase(model, tenant_a, k, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    wan, ti = ov._bind_tenant(model, tenant_a)
    block = ov._ensure_block(tenant_a, wan, ti, k)
    st = phase._state(tenant_a, k)
    runner = micro.TenantMicroRunner.for_self(k)
    runner.scratch = st.self_scratch
    runner.step_idx = phase._step_index(runner, "sa_o")
    attn_out = runner.scratch["attn_out"]
    pf = block.compute_phases[0]
    orch2 = phase._get_orch(device)
    orig_mm = orch2._orig["mm_apply"]

    torch.cuda.synchronize()
    ev.start("sa_o_matmul")
    partial = orig_mm(pf.self_attn_o, attn_out)
    sa_o_matmul_cuda_ms = ev.stop("sa_o_matmul")

    torch.cuda.synchronize()
    ev.start("sa_o_ar_only")
    with torch.cuda.stream(orch2.comm_stream):
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=pf.self_attn_o.tp_group, async_op=False)
    sa_o_ar_cuda_ms = ev.stop("sa_o_ar_only")

    out["phase_timings"] = {
        "comm_phase2": t_comm2,
        "comp_phase1": t_comp1,
        "pair_A2B1_serial_wall_ms": pair_serial_wall_ms,
        "pair_A2B1_overlap_wall_ms": pair_overlap_wall_ms,
        "profile_sum_serial_ms": t_comm2["cuda_ms"] + t_comp1["cuda_ms"],
        "pair_serial_minus_profile_ms": pair_serial_wall_ms - (t_comm2["cuda_ms"] + t_comp1["cuda_ms"]),
        "overlap_comm_phase_cuda_ms": overlap_comm_cuda_ms,
        "sa_o_breakdown": {
            "o_matmul_cuda_ms": sa_o_matmul_cuda_ms,
            "ar_only_cuda_ms": sa_o_ar_cuda_ms,
            "phase2_is_pure_comm": False,
            "note": "phase2 runs sa_o step = matmul + AR bundled in MMWeightTP hook",
        },
    }

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = out
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(json.dumps(merged, indent=2))
        print(f"wrote {path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
