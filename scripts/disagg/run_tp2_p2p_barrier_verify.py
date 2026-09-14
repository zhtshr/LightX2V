#!/usr/bin/env python3
"""Verify whether P2P norm barrier/sync blocks NCCL∥compute overlap.

Measures:
  1) Per-call cuda.sync / gloo barrier wall time inside sum_reduce
  2) A2B1 overlap pair wall: P2P default vs skip_barrier vs skip_cuda_sync vs NCCL norm
  3) Micro: async NCCL on comm_stream ∥ repeated sum_reduce on compute_stream

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_p2p_barrier_verify.py \\
    --output_json save_results/optimization_study/tp2_p2p_barrier_verify.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


@dataclass
class P2pStats:
    calls: int = 0
    calls_during_peer_pump: int = 0
    cuda_sync_ms: float = 0.0
    barrier_ms: float = 0.0
    copy_peer_ms: float = 0.0
    total_ms: float = 0.0
    cuda_sync_during_pump_ms: float = 0.0
    barrier_during_pump_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "calls_during_peer_pump": self.calls_during_peer_pump,
            "cuda_sync_ms": self.cuda_sync_ms,
            "barrier_ms": self.barrier_ms,
            "copy_peer_ms": self.copy_peer_ms,
            "total_ms": self.total_ms,
            "cuda_sync_during_pump_ms": self.cuda_sync_during_pump_ms,
            "barrier_during_pump_ms": self.barrier_during_pump_ms,
            "avg_barrier_ms": self.barrier_ms / max(self.calls, 1),
            "avg_barrier_during_pump_ms": self.barrier_during_pump_ms / max(self.calls_during_peer_pump, 1),
        }


_STATS = P2pStats()
_SKIP_CUDA_SYNC = False
_SKIP_BARRIER = False
_ORIG_SUM_REDUCE = None


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _install_p2p_patch(phase_mod: Any) -> None:
    global _ORIG_SUM_REDUCE
    from lightx2v.common.ops.norm import tp_p2p_exchange as p2p_mod

    if _ORIG_SUM_REDUCE is None:
        _ORIG_SUM_REDUCE = p2p_mod.TpP2pScalarExchange.sum_reduce

    orch_getter = phase_mod._get_orch

    def instrumented_sum_reduce(self, local_sum: torch.Tensor) -> torch.Tensor:
        n = local_sum.numel()
        flat = local_sum.contiguous().view(-1)
        t0 = time.perf_counter()
        if flat.dtype == torch.float32:
            self.local_buf[:n].copy_(flat)
        else:
            self.local_buf[:n].copy_(flat.float())
        t1 = time.perf_counter()

        if not _SKIP_CUDA_SYNC:
            torch.cuda.current_stream().synchronize()
        t2 = time.perf_counter()

        if not _SKIP_BARRIER:
            dist.barrier(group=self._gloo_group)
        t3 = time.perf_counter()

        self.staging[:n].copy_(self._peer_buf[:n])
        out = self.local_buf[:n] + self.staging[:n]
        t4 = time.perf_counter()
        result = out.view_as(local_sum).to(dtype=local_sum.dtype)

        orch = orch_getter(torch.device(f"cuda:{torch.cuda.current_device()}"))
        during_pump = bool(getattr(orch, "in_peer_comp", False))

        _STATS.calls += 1
        sync_ms = (t2 - t1) * 1000.0
        bar_ms = (t3 - t2) * 1000.0
        copy_ms = (t4 - t3) * 1000.0
        _STATS.cuda_sync_ms += sync_ms
        _STATS.barrier_ms += bar_ms
        _STATS.copy_peer_ms += copy_ms
        _STATS.total_ms += (t4 - t0) * 1000.0
        if during_pump:
            _STATS.calls_during_peer_pump += 1
            _STATS.cuda_sync_during_pump_ms += sync_ms
            _STATS.barrier_during_pump_ms += bar_ms
        return result

    p2p_mod.TpP2pScalarExchange.sum_reduce = instrumented_sum_reduce


def _reset_stats() -> None:
    global _STATS
    _STATS = P2pStats()


def _wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _prep_pair(phase_mod, model, ov, layer, comm_t, comp_t):
    phase_mod.run_tenant_phase(model, comm_t, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    phase_mod.run_tenant_phase(model, comm_t, layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    phase_mod.run_tenant_phase(model, comp_t, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)


def _profile_a2b1_overlap(
    phase_mod: Any,
    model: Any,
    ov: Any,
    layer: int,
    tenant_a: Any,
    tenant_b: Any,
) -> dict[str, Any]:
    orch = phase_mod._get_orch(torch.device(f"cuda:{dist.get_rank()}"))
    _prep_pair(phase_mod, model, ov, layer, tenant_a, tenant_b)
    _reset_stats()

    def run_overlap() -> None:
        orch.enabled = True
        orch.set_peer(phase_mod._make_peer_runner(
            model, tenant_b, layer, 1,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        phase_mod.run_tenant_phase(
            model, tenant_a, layer, 2,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
        orch.drain_peer()
        orch.enabled = False
        orch.set_peer(None)

    wall = _wall_ms(run_overlap)
    return {"pair_wall_ms": wall, "p2p_stats": _STATS.to_dict()}


def _micro_nccl_vs_p2p(
    device: torch.device,
    tp_group: Any,
    orch: Any,
    exchange: Any,
    n_elems: int = 64,
) -> dict[str, float]:
    ar_buf = torch.randn(8192, 5120, device=device, dtype=torch.float16)
    local_sum = torch.randn(n_elems, device=device, dtype=torch.float32)

    def serial_p2p(k: int) -> None:
        for _ in range(k):
            exchange.sum_reduce(local_sum)
        dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=tp_group, async_op=False)

    def overlap_p2p(k: int) -> None:
        with torch.cuda.stream(orch.comm_stream):
            work = dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=tp_group, async_op=True)
        with torch.cuda.stream(orch.compute_stream):
            for _ in range(k):
                exchange.sum_reduce(local_sum)
        if work is not None:
            work.wait()

    def cuda_overlap_ms(k: int, fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn(k)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    k = 4
    ser = cuda_overlap_ms(k, serial_p2p)
    ovl = cuda_overlap_ms(k, overlap_p2p)
    return {
        "micro_k": float(k),
        "micro_serial_ms": ser,
        "micro_overlap_ms": ovl,
        "micro_speedup": ser / max(ovl, 0.001),
        "micro_saved_ms": ser - ovl,
    }


def _dual_stream_naive(
    phase_mod: Any,
    model: Any,
    ov: Any,
    orch: Any,
    layer: int,
    tenant_a: Any,
    tenant_b: Any,
) -> float:
    _prep_pair(phase_mod, model, ov, layer, tenant_a, tenant_b)
    err: list[BaseException] = []

    def run_comm() -> None:
        try:
            with torch.cuda.stream(orch.comm_stream):
                phase_mod.run_tenant_phase(
                    model, tenant_a, layer, 2,
                    ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
                )
        except BaseException as exc:
            err.append(exc)

    def run_comp() -> None:
        try:
            with torch.cuda.stream(orch.compute_stream):
                phase_mod.run_tenant_phase(
                    model, tenant_b, layer, 1,
                    ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
                )
        except BaseException as exc:
            err.append(exc)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_comm = threading.Thread(target=run_comm)
    t_comp = threading.Thread(target=run_comp)
    t_comm.start()
    t_comp.start()
    t_comm.join()
    t_comp.join()
    torch.cuda.synchronize()
    if err:
        raise err[0]
    return (time.perf_counter() - t0) * 1000.0


def _prepare_tenants(p1: Any, ov: Any, model: Any, config: dict, payload_a: dict, payload_b: dict):
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
    return tenant_a, tenant_b


def _load_model_and_tenants(p1, ov, config, payload_a, payload_b):
    model = load_wan_transformer(config)
    tenant_a, tenant_b = _prepare_tenants(p1, ov, model, config, payload_a, payload_b)
    wan, ti = ov._bind_tenant(model, tenant_a)
    ov._preload_blocks(tenant_a, wan, ti, len(wan.transformer_weights.blocks))
    wan_b, ti_b = ov._bind_tenant(model, tenant_b)
    ov._preload_blocks(tenant_b, wan_b, ti_b, len(wan_b.transformer_weights.blocks))
    tenant_a.pipe_states = {}
    tenant_b.pipe_states = {}
    return model, tenant_a, tenant_b, wan


def _run_p2p_variants(
    phase_mod: Any,
    ov: Any,
    p1: Any,
    config: dict,
    payload_a: dict,
    payload_b: dict,
    layer: int,
) -> dict[str, Any]:
    model, tenant_a, tenant_b, wan = _load_model_and_tenants(p1, ov, config, payload_a, payload_b)
    orch = phase_mod._get_orch(torch.device(f"cuda:{dist.get_rank()}"))

    from lightx2v.common.ops.norm.tp_p2p_exchange import get_tp_p2p_exchange

    ph = wan.transformer_weights.blocks[layer].compute_phases[0]
    tp_group = ph.self_attn_norm_q.tp_group
    exchange = get_tp_p2p_exchange(tp_group)

    results: dict[str, Any] = {}
    for name, skip_sync, skip_bar in (
        ("p2p_default", False, False),
        ("p2p_skip_barrier", False, True),
        ("p2p_skip_cuda_sync", True, False),
    ):
        global _SKIP_CUDA_SYNC, _SKIP_BARRIER
        _SKIP_CUDA_SYNC = skip_sync
        _SKIP_BARRIER = skip_bar
        tenant_a.pipe_states = {}
        tenant_b.pipe_states = {}
        rec = _profile_a2b1_overlap(phase_mod, model, ov, layer, tenant_a, tenant_b)
        rec["micro"] = _micro_nccl_vs_p2p(
            torch.device(f"cuda:{dist.get_rank()}"), tp_group, orch, exchange,
        )
        results[name] = rec

    _SKIP_CUDA_SYNC = False
    _SKIP_BARRIER = False
    results["dual_stream_naive_ms"] = _dual_stream_naive(
        phase_mod, model, ov, orch, layer, tenant_a, tenant_b,
    )

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _run_nccl_variant(
    phase_mod: Any,
    ov: Any,
    p1: Any,
    config: dict,
    payload_a: dict,
    payload_b: dict,
    layer: int,
) -> dict[str, Any]:
    global _SKIP_CUDA_SYNC, _SKIP_BARRIER
    _SKIP_CUDA_SYNC = False
    _SKIP_BARRIER = False
    model, tenant_a, tenant_b, _wan = _load_model_and_tenants(p1, ov, config, payload_a, payload_b)
    orch = phase_mod._get_orch(torch.device(f"cuda:{dist.get_rank()}"))
    rec = _profile_a2b1_overlap(phase_mod, model, ov, layer, tenant_a, tenant_b)
    rec["dual_stream_naive_ms"] = _dual_stream_naive(
        phase_mod, model, ov, orch, layer, tenant_a, tenant_b,
    )
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load("overlap", here / "run_phase3_dual_overlap_bench.py")
    phase_mod = _load("phase_pipe", here / "tp_phase_pipeline.py")
    _install_p2p_patch(phase_mod)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--output_json", default="save_results/optimization_study/tp2_p2p_barrier_verify.json")
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
    config_base = p1._load_config(ns)
    seed_all(42)
    p1._init_distributed(config_base)

    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(
            config=config_base,
            cache_path=Path(args.inputs_cache),
            prompt=ns.prompt,
            image_path=ns.image_path,
            seed=42,
            force=False,
            task="i2v",
            negative_prompt=ns.negative_prompt,
        )
    )
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    out: dict[str, Any] = {"layer": args.layer, "rank": dist.get_rank()}

    config_p2p = dict(config_base)
    config_p2p["tp_norm_p2p"] = True
    p2p_results = _run_p2p_variants(
        phase_mod, ov, p1, config_p2p, payload_a, payload_b, args.layer,
    )
    out.update(p2p_results)

    config_nccl = dict(config_base)
    config_nccl["tp_norm_p2p"] = False
    out["nccl_norm"] = _run_nccl_variant(
        phase_mod, ov, p1, config_nccl, payload_a, payload_b, args.layer,
    )

    if is_main_process():
        d = out["p2p_default"]
        sb = out["p2p_skip_barrier"]
        sc = out["p2p_skip_cuda_sync"]
        naive = out["dual_stream_naive_ms"]
        out["summary"] = {
            "barrier_is_primary_cause": (
                sb["pair_wall_ms"] < naive * 0.95
                and d["pair_wall_ms"] - sb["pair_wall_ms"] > 1.0
            ),
            "pair_wall_delta_skip_barrier_ms": d["pair_wall_ms"] - sb["pair_wall_ms"],
            "pair_wall_delta_skip_cuda_sync_ms": d["pair_wall_ms"] - sc["pair_wall_ms"],
            "barrier_ms_during_pump": d["p2p_stats"]["barrier_during_pump_ms"],
            "barrier_calls_during_pump": d["p2p_stats"]["calls_during_peer_pump"],
            "avg_barrier_ms_during_pump": d["p2p_stats"]["avg_barrier_during_pump_ms"],
            "dual_stream_naive_ms": naive,
            "gap_overlap_default_vs_naive_ms": d["pair_wall_ms"] - naive,
            "gap_overlap_skip_barrier_vs_naive_ms": sb["pair_wall_ms"] - naive,
            "micro_p2p_default": d.get("micro"),
            "micro_p2p_skip_barrier": sb.get("micro"),
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out["summary"], indent=2))
        print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
