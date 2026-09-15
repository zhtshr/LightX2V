#!/usr/bin/env python3
"""B2 xor_rd (recursive-doubling Q) vs p2p overlap vs collective. CVD=0,1,2,4."""

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
    torch.cuda.set_device(rank)
    group = dist.group.WORLD

    ql = 1170
    kl = int(os.environ.get("KL", "4680"))
    h, d = 12, 128
    iters, warmup = 30, 8

    q = torch.randn(ql, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kl, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kl, h, d, device="cuda", dtype=torch.bfloat16)
    pool = _Pool(
        num_layers=1, cache_size=kl, num_heads=h, head_dim=d,
        dtype=torch.bfloat16, device=torch.device("cuda"),
    )

    def run(mode: str) -> None:
        os.environ["LIGHTX2V_STRIPE_B2_MODE"] = mode
        os.environ.pop("LIGHTX2V_STRIPE_B2_P2P_NO_OVERLAP", None)
        if mode == "collective":
            os.environ["LIGHTX2V_STRIPE_B2_COLLECTIVE"] = "1"
        else:
            os.environ.pop("LIGHTX2V_STRIPE_B2_COLLECTIVE", None)
        pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)

    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "xor_rd"
    out_x = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "p2p"
    out_p = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    diff = (out_x - out_p).abs().max().item()

    ms_xor = _ms(lambda: run("xor_rd"), iters, warmup)
    ms_p2p = _ms(lambda: run("p2p"), iters, warmup)
    ms_col = _ms(lambda: run("collective"), iters, warmup)

    def pack(x: float) -> float:
        t = torch.tensor([x], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    out = {
        "payload": {"Ql": ql, "Kl": kl, "H": h, "D": d},
        "max_diff_xor_vs_p2p": diff,
        "aligned_ms_maxrank": {
            "xor_rd": round(pack(ms_xor), 3),
            "p2p_overlap": round(pack(ms_p2p), 3),
            "collective": round(pack(ms_col), 3),
        },
    }
    if rank == 0:
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_b2_xor_rd_vs_p2p.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
