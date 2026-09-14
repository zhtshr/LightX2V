#!/usr/bin/env python3
"""A/B: Form B2 P2P mesh vs collective (all_gather+alltoall), aligned + skew.

Skew pattern: ranks 1..P-1 post/enter Q exchange immediately; rank 0 burns
``imbalance_ms`` first. With P2P, ready ranks can finish peer↔peer transfers
during the burn; with all_gather they sit idle until rank 0 enters.

CVD: use 0,1,2,4 for P=4 (skip physical GPU3).
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

from lightx2v.common.kvcache.base import BaseKVCachePool


class _Pool(BaseKVCachePool):
    pass


def _ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD

    ql, kl, h, d = 1170, 4680, 12, 128
    iters, warmup = 20, 5
    imbalance_ms = float(os.environ.get("IMBALANCE_MS", "3.0"))

    q = torch.randn(ql, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kl, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kl, h, d, device="cuda", dtype=torch.bfloat16)
    pool = _Pool(
        num_layers=1,
        cache_size=kl,
        num_heads=h,
        head_dim=d,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    # Correctness: P2P vs collective
    os.environ["LIGHTX2V_STRIPE_B2_COLLECTIVE"] = "0"
    out_p2p = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    os.environ["LIGHTX2V_STRIPE_B2_COLLECTIVE"] = "1"
    out_col = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    diff = (out_p2p - out_col).abs().max().item()

    def run_p2p() -> None:
        os.environ["LIGHTX2V_STRIPE_B2_COLLECTIVE"] = "0"
        pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)

    def run_col() -> None:
        os.environ["LIGHTX2V_STRIPE_B2_COLLECTIVE"] = "1"
        pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)

    def skew_cuda_burn(ms: float) -> None:
        """Busy-spin on GPU ~ms (rank 0 only)."""
        if rank != 0:
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Cheap GEMM loop until wall approx ms
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        start.record()
        while True:
            torch.mm(a, b)
            end.record()
            end.synchronize()
            if start.elapsed_time(end) >= ms:
                break

    def run_p2p_skew() -> None:
        skew_cuda_burn(imbalance_ms)
        run_p2p()

    def run_col_skew() -> None:
        skew_cuda_burn(imbalance_ms)
        run_col()

    # Aligned (barrier before each timed arm inside _ms)
    aligned_p2p = _ms(run_p2p, iters, warmup)
    aligned_col = _ms(run_col, iters, warmup)
    skew_p2p = _ms(run_p2p_skew, iters, warmup)
    skew_col = _ms(run_col_skew, iters, warmup)

    # Reduce max so we report the waiter-side latency
    def pack(x: float) -> float:
        t = torch.tensor([x], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    out = {
        "world_size": world,
        "payload": {"Ql": ql, "Kl": kl, "H": h, "D": d},
        "imbalance_ms": imbalance_ms,
        "max_diff_p2p_vs_collective": diff,
        "aligned_ms_maxrank": {
            "b2_p2p": round(pack(aligned_p2p), 3),
            "b2_collective": round(pack(aligned_col), 3),
        },
        "skew_ms_maxrank": {
            "b2_p2p": round(pack(skew_p2p), 3),
            "b2_collective": round(pack(skew_col), 3),
        },
    }
    a = out["aligned_ms_maxrank"]
    s = out["skew_ms_maxrank"]
    out["skew_extra_ms"] = {
        "p2p": round(s["b2_p2p"] - a["b2_p2p"], 3),
        "collective": round(s["b2_collective"] - a["b2_collective"], 3),
    }
    out["ratio_p2p_over_collective"] = {
        "aligned": round(a["b2_p2p"] / max(a["b2_collective"], 1e-9), 3),
        "skew": round(s["b2_p2p"] / max(s["b2_collective"], 1e-9), 3),
    }

    if rank == 0:
        print(json.dumps(out, indent=2))
        path = os.environ.get(
            "OUT_JSON",
            "save_results/optimization_study/sf_b2_p2p_vs_collective.json",
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
