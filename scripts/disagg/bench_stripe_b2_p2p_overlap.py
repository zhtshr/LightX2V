#!/usr/bin/env python3
"""B2 P2P overlap vs sequential mesh. CVD=0,1,2,4 for P=4."""

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
        num_layers=1,
        cache_size=kl,
        num_heads=h,
        head_dim=d,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "p2p"
    os.environ["LIGHTX2V_STRIPE_B2_P2P_NO_OVERLAP"] = "0"
    out_ov = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    os.environ["LIGHTX2V_STRIPE_B2_P2P_NO_OVERLAP"] = "1"
    out_seq = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)
    diff = (out_ov - out_seq).abs().max().item()

    def run(no_overlap: str) -> None:
        os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "p2p"
        os.environ["LIGHTX2V_STRIPE_B2_P2P_NO_OVERLAP"] = no_overlap
        pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=d, q_exchange=True)

    ms_ov = _ms(lambda: run("0"), iters, warmup)
    ms_seq = _ms(lambda: run("1"), iters, warmup)

    def pack(x: float) -> float:
        t = torch.tensor([x], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    out = {
        "payload": {"Ql": ql, "Kl": kl, "H": h, "D": d},
        "max_diff_ov_vs_seq": diff,
        "aligned_ms_maxrank": {
            "p2p_overlap": round(pack(ms_ov), 3),
            "p2p_sequential": round(pack(ms_seq), 3),
        },
    }
    a = out["aligned_ms_maxrank"]
    out["ratio_ov_over_seq"] = round(a["p2p_overlap"] / max(a["p2p_sequential"], 1e-9), 3)
    out["speedup_vs_seq"] = round(a["p2p_sequential"] / max(a["p2p_overlap"], 1e-9), 3)

    if rank == 0:
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_b2_p2p_overlap.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
