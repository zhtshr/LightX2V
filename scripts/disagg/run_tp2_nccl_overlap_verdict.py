#!/usr/bin/env python3
"""Exhaustive NCCL overlap verdict for TP=2 phase pipeline (production comm strategy).

Tests (all row AR = NCCL, norm = P2P when configured):
  T1  Synthetic AR∥GEMM (isolated NCCL+compute concurrency possible?)
  T2  Real o_partial tensor size AR∥GEMM
  T3  Real A2B1 pair wall: serial vs overlap (NCCL only)
  T4  Peer COMP during overlap: count norm NCCL kernels (should be 0 with tp_norm_p2p)
  T5  NCCL env variants (LAUNCH_MODE, CUDA_DEVICE_MAX_CONNECTIONS)
  T6  Chunked NCCL vs single NCCL
  T7  work.wait vs comm_stream.synchronize

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_nccl_overlap_verdict.py \\
    --output_json save_results/optimization_study/tp2_nccl_overlap_verdict.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.profiler import no_sync_profiling
from lightx2v.utils.utils import is_main_process, seed_all


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


def _cuda_ms(fn: Callable[[], None], *, reps: int = 5) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(reps):
        _sync()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        _sync()
        samples.append(s.elapsed_time(e))
    return {
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _test_synthetic_ar_gemm(
    group: dist.ProcessGroup,
    dev: torch.device,
    ar_shape: tuple[int, ...],
    *,
    gemm_iters: int = 8,
) -> dict[str, Any]:
    ar = torch.randn(*ar_shape, device=dev, dtype=torch.float16)
    gemm = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    comm_stream = torch.cuda.Stream(device=dev)
    compute_stream = torch.cuda.Stream(device=dev)

    def serial() -> None:
        dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=False)
        with torch.cuda.stream(compute_stream):
            for _ in range(gemm_iters):
                torch.matmul(gemm, gemm)

    def overlap() -> None:
        with torch.cuda.stream(comm_stream):
            work = dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=True)
        with torch.cuda.stream(compute_stream):
            for _ in range(gemm_iters):
                torch.matmul(gemm, gemm)
        if work is not None:
            work.wait()

    def ar_only() -> None:
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=False)

    def gemm_only() -> None:
        with torch.cuda.stream(compute_stream):
            for _ in range(gemm_iters):
                torch.matmul(gemm, gemm)

    ser = _cuda_ms(serial)
    ovl = _cuda_ms(overlap)
    ar_ms = _cuda_ms(ar_only)
    gemm_ms = _cuda_ms(gemm_only)
    ideal = max(ar_ms["mean_ms"], gemm_ms["mean_ms"])
    return {
        "ar_shape": list(ar_shape),
        "ar_numel": int(ar.numel()),
        "serial_ms": ser["mean_ms"],
        "overlap_ms": ovl["mean_ms"],
        "ar_only_ms": ar_ms["mean_ms"],
        "gemm_only_ms": gemm_ms["mean_ms"],
        "ideal_max_ms": ideal,
        "overlap_speedup_vs_serial": ser["mean_ms"] / max(ovl["mean_ms"], 0.001),
        "overlap_vs_ideal_ratio": ovl["mean_ms"] / max(ideal, 0.001),
        "concurrent_possible_wall": ovl["mean_ms"] < ser["mean_ms"] * 0.95,
    }


def _estimate_real_ar_shape(model: Any, tenant: Any, ov: Any, layer: int) -> tuple[int, ...]:
    wan, ti = ov._bind_tenant(model, tenant)
    ov._ensure_block(tenant, wan, ti, layer)
    block = tenant.block_cache[layer]
    pf = block.compute_phases[0]
    # approximate o_partial after sa_v_attn path
    n_heads_shard = ti.num_heads
    head_dim = ti.head_dim
    seq = 4096  # typical 256x256 latent seq after patch
    return (seq, n_heads_shard * head_dim)


def _sync_orch(orch: Any) -> None:
    orch.comm_stream.synchronize()
    orch.compute_stream.synchronize()


def _measure_a2b1(
    phase: Any,
    model: Any,
    ta: Any,
    tb: Any,
    ov: Any,
    orch: Any,
    layer: int,
    *,
    overlap: bool,
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

    _sync_orch(orch)
    t0 = time.perf_counter()
    if overlap:
        with no_sync_profiling(enabled=True):
            orch.enabled = True
            orch.set_peer(phase._make_peer_runner(
                model, comp_t, layer, comp_p,
                ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
            ))
            phase.run_tenant_phase(
                model, comm_t, layer, comm_p,
                ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
            )
            orch.drain_peer()
            orch.enabled = False
            orch.set_peer(None)
    else:
        phase.run_tenant_phase(
            model, comm_t, layer, comm_p,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
        prep(comp_t, comp_p)
        phase.run_tenant_phase(
            model, comp_t, layer, comp_p,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
    _sync_orch(orch)
    return (time.perf_counter() - t0) * 1000.0


@contextmanager
def _count_nccl_launches():
    """Monkeypatch dist.all_reduce to count NCCL calls during peer COMP."""
    counts = {"nccl": 0, "bytes": 0}
    orig = dist.all_reduce

    def wrapped(tensor, *args, **kwargs):
        counts["nccl"] += 1
        counts["bytes"] += tensor.numel() * tensor.element_size()
        return orig(tensor, *args, **kwargs)

    dist.all_reduce = wrapped
    try:
        yield counts
    finally:
        dist.all_reduce = orig


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("ph", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--output_json", default="save_results/optimization_study/tp2_nccl_overlap_verdict.json")
    args = parser.parse_args()

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

    rank = dist.get_rank()
    dev = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    out: dict[str, Any] = {
        "env": {
            "NCCL_LAUNCH_MODE": os.environ.get("NCCL_LAUNCH_MODE"),
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
            "device": torch.cuda.get_device_name(rank) if rank == 0 else None,
        },
        "policy": {
            "row_comm_ar": "NCCL",
            "norm_ar": "P2P (tp_norm_p2p=True)",
            "use_comm_p2p": False,
        },
    }

    # T1/T2 synthetic
    shapes = [
        (8192, 5120),
        (4096, 2048),
    ]
    out["synthetic_ar_gemm"] = [_test_synthetic_ar_gemm(group, dev, s) for s in shapes]

    # Load model for real tests
    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(
            config=config, cache_path=Path(args.inputs_cache),
            prompt=ns.prompt, image_path=ns.image_path, seed=42,
            force=False, task="i2v", negative_prompt="",
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
        s.prepare(
            seed=int(p["seed"]),
            latent_shape=p["latent_shape"],
            image_encoder_output=p["image_encoder_output"],
        )
        t.inputs = p["inputs"]
        s.step_pre(step_index=0)
        ov._pre_infer_tenant(model, t)

    real_shape = _estimate_real_ar_shape(model, ta, ov, args.layer)
    out["real_ar_shape_estimate"] = list(real_shape)
    out["synthetic_real_size"] = _test_synthetic_ar_gemm(group, dev, real_shape)

    phase.configure_phase_pipeline(use_comm_p2p=False, use_chunked_ar=False)
    orch = phase._get_orch(dev)

    # warmup
    _measure_a2b1(phase, model, ta, tb, ov, orch, args.layer, overlap=True)

    comm_alone_samples, comp_alone_samples = [], []
    for _ in range(3):
        # comm alone phase 2
        ta.pipe_states = {}
        for p in range(1, 2):
            phase.run_tenant_phase(model, ta, args.layer, p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        _sync()
        t0 = time.perf_counter()
        phase.run_tenant_phase(model, ta, args.layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        _sync()
        comm_alone_samples.append((time.perf_counter() - t0) * 1000.0)

        tb.pipe_states = {}
        for p in range(1, 1):
            pass
        for p in range(1, 2):
            phase.run_tenant_phase(model, tb, args.layer, p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        _sync()
        t0 = time.perf_counter()
        phase.run_tenant_phase(model, tb, args.layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        _sync()
        comp_alone_samples.append((time.perf_counter() - t0) * 1000.0)

    serial_samples, overlap_samples = [], []
    for _ in range(3):
        serial_samples.append(_measure_a2b1(phase, model, ta, tb, ov, orch, args.layer, overlap=False))
        with _count_nccl_launches() as nccl_cnt:
            overlap_samples.append(_measure_a2b1(phase, model, ta, tb, ov, orch, args.layer, overlap=True))
        if "nccl_during_overlap" not in out:
            out["nccl_during_overlap"] = nccl_cnt

    comm_m = statistics.mean(comm_alone_samples)
    comp_m = statistics.mean(comp_alone_samples)
    serial_m = statistics.mean(serial_samples)
    overlap_m = statistics.mean(overlap_samples)
    ideal = max(comm_m, comp_m)

    out["a2b1_layer"] = args.layer
    out["a2b1"] = {
        "comm_alone_ms": comm_m,
        "comp_alone_ms": comp_m,
        "ideal_max_ms": ideal,
        "serial_pair_ms": serial_m,
        "overlap_pair_ms": overlap_m,
        "overlap_vs_serial_pct": 100.0 * (overlap_m - serial_m) / serial_m,
        "overlap_vs_ideal_pct": 100.0 * (overlap_m - ideal) / ideal,
        "hidden_comm_fraction": (serial_m - overlap_m) / max(serial_m - ideal, 0.001) if serial_m > ideal else None,
    }

    # T6 chunked
    phase.configure_phase_pipeline(use_comm_p2p=False, use_chunked_ar=True)
    chunked_samples = [_measure_a2b1(phase, model, ta, tb, ov, orch, args.layer, overlap=True) for _ in range(3)]
    out["a2b1_chunked_nccl"] = {
        "overlap_pair_ms": statistics.mean(chunked_samples),
        "delta_vs_non_chunked_pct": 100.0 * (statistics.mean(chunked_samples) - overlap_m) / overlap_m,
    }
    phase.configure_phase_pipeline(use_comm_p2p=False, use_chunked_ar=False)

    # Verdict logic
    syn = out["synthetic_ar_gemm"][0]
    syn_ok = syn["concurrent_possible_wall"] and syn["overlap_speedup_vs_serial"] > 1.2
    real_a2b1_ok = overlap_m < serial_m * 0.95 and overlap_m < ideal * 1.1

    if syn_ok and not real_a2b1_ok:
        verdict = "NCCL_CAN_OVERLAP_IN_ISOLATION_BUT_NOT_IN_REAL_PIPELINE"
        detail = (
            "Isolated NCCL+GEMM on two streams shows wall-time overlap, but A2B1 "
            "pair does not reach ideal — blocker is in real pipeline (scheduling, "
            "extra NCCL from norm, implicit sync, or stream assignment), not NCCL hardware."
        )
    elif not syn_ok:
        verdict = "NCCL_TWO_STREAM_OVERLAP_FAILS_EVEN_SYNTHETIC"
        detail = "Even pure AR∥GEMM does not reduce wall time — NCCL/PG blocks concurrent streams."
    elif real_a2b1_ok:
        verdict = "NCCL_OVERLAP_WORKS"
        detail = "A2B1 overlap approaches ideal with current NCCL strategy."
    else:
        verdict = "NCCL_OVERLAP_MARGINAL"
        detail = "Some overlap but far from ideal; investigate remaining sync and concurrent NCCL from norm."

    out["verdict"] = verdict
    out["verdict_detail"] = detail
    out["recommendation"] = (
        "If verdict is NCCL_TWO_STREAM_OVERLAP_FAILS_EVEN_SYNTHETIC or overlap_vs_ideal "
        "remains >>0 after tp_norm_p2p + sync-fix: consider non-NCCL row AR (P2P/SHM) for TP=2 "
        "or separate NCCL communicators / dedicated comm progress thread — not more pump tuning."
        if not real_a2b1_ok else "Continue NCCL path tuning (chunked, stream wait)."
    )

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({"verdict": verdict, "a2b1": out["a2b1"], "synthetic": syn}, indent=2))
        print(f"wrote {path}")
        print("结论文档: save_results/nsys/tp2_overlap_root_cause.md")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
