#!/usr/bin/env python3
"""Form B2: Q-ring vs P2P mesh (aligned). CVD=0,1,2,4 for P=4."""

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

    ql, kl, h, d = 1170, int(os.environ.get("KL", "4680")), 12, 128
    iters, warmup = 30, 8

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

    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "ring"
    out_r = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "p2p"
    out_p = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    diff = (out_r - out_p).abs().max().item()

    def run(mode: str) -> None:
        os.environ["LIGHTX2V_STRIPE_B2_MODE"] = mode
        pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)

    ms_ring = _ms(lambda: run("ring"), iters, warmup)
    ms_p2p = _ms(lambda: run("p2p"), iters, warmup)

    def pack(x: float) -> float:
        t = torch.tensor([x], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    out = {
        "world_size": world,
        "payload": {"Ql": ql, "Kl": kl, "H": h, "D": d},
        "max_diff_ring_vs_p2p": diff,
        "aligned_ms_maxrank": {
            "b2_ring": round(pack(ms_ring), 3),
            "b2_p2p": round(pack(ms_p2p), 3),
        },
    }
    a = out["aligned_ms_maxrank"]
    out["ratio_ring_over_p2p"] = round(a["b2_ring"] / max(a["b2_p2p"], 1e-9), 3)

    if rank == 0:
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_b2_ring_vs_p2p.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
