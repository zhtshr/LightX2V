#!/usr/bin/env python3
"""A/B microbench: Form C split alltoall(out)+alltoall(lse) vs packed single alltoall."""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from lightx2v.common.kvcache.base import BaseKVCachePool


def _ms(fn, iters: int = 80, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    starts: list[torch.cuda.Event] = []
    ends: list[torch.cuda.Event] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        starts.append(s)
        ends.append(e)
    torch.cuda.synchronize()
    return sum(float(s.elapsed_time(e)) for s, e in zip(starts, ends, strict=True)) / iters


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    if ws != 4:
        if rank == 0:
            print("requires world_size=4", file=sys.stderr)
        return 1
    torch.cuda.set_device(rank)
    group = dist.group.WORLD

    ql, h, d = 1170, 12, 128
    q = ql * ws
    block_out = torch.randn(1, q, h, d, device="cuda", dtype=torch.bfloat16)
    block_lse = torch.randn(1, h, q, device="cuda", dtype=torch.float32)
    q_lens = [ql] * ws

    def split_a2a() -> None:
        os.environ.pop("LIGHTX2V_STRIPE_PACK_OUT_LSE", None)
        BaseKVCachePool._stripe_partial_out_alltoall(block_out, block_lse, q_lens, group, None)

    def pack_a2a() -> None:
        os.environ["LIGHTX2V_STRIPE_PACK_OUT_LSE"] = "1"
        BaseKVCachePool._stripe_partial_out_alltoall(block_out, block_lse, q_lens, group, None)

    q_local = torch.randn(ql, h, d, device="cuda", dtype=torch.bfloat16)
    q_full = torch.empty(q, h, d, device="cuda", dtype=torch.bfloat16)
    send_out = [c.contiguous() for c in torch.split(block_out, q_lens, dim=1)]
    send_lse = [c.contiguous() for c in torch.split(block_lse, q_lens, dim=-1)]
    recv_out = [torch.empty_like(send_out[rank]) for _ in range(ws)]
    recv_lse = [torch.empty_like(send_lse[rank]) for _ in range(ws)]
    send_pack = [
        BaseKVCachePool._pack_out_lse_chunk(o, l) for o, l in zip(send_out, send_lse, strict=True)
    ]
    recv_pack = [torch.empty_like(send_pack[rank]) for _ in range(ws)]

    def gather_only() -> None:
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    def split_then_gather() -> None:
        dist.all_to_all(recv_out, send_out, group=group)
        dist.all_to_all(recv_lse, send_lse, group=group)
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    def pack_then_gather() -> None:
        dist.all_to_all(recv_pack, send_pack, group=group)
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    def lse_a2a() -> None:
        dist.all_to_all(recv_lse, send_lse, group=group)

    def lse_then_gather() -> None:
        dist.all_to_all(recv_lse, send_lse, group=group)
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    split_ms = _ms(split_a2a)
    pack_ms = _ms(pack_a2a)
    gather_ms = _ms(gather_only)
    split_g = _ms(split_then_gather)
    pack_g = _ms(pack_then_gather)
    lse_ms = _ms(lse_a2a)
    lse_g = _ms(lse_then_gather)

    results = {
        "world_size": ws,
        "q_local": [ql, h, d],
        "bytes_out_per_peer": ql * h * d * 2,
        "bytes_lse_per_peer": h * ql * 4,
        "bytes_pack_per_peer": int(send_pack[0].numel()),
        "split_a2a_ms": split_ms,
        "pack_a2a_ms": pack_ms,
        "pack_vs_split_delta_ms": pack_ms - split_ms,
        "pack_vs_split_ratio": pack_ms / max(split_ms, 1e-9),
        "gather_only_warm_ms": gather_ms,
        "split_then_gather_ms": split_g,
        "pack_then_gather_ms": pack_g,
        "implied_gather_after_split_ms": split_g - split_ms,
        "implied_gather_after_pack_ms": pack_g - pack_ms,
        "lse_a2a_ms": lse_ms,
        "lse_then_gather_ms": lse_g,
        "implied_gather_after_lse_ms": lse_g - lse_ms,
    }

    if rank == 0:
        print(json.dumps(results, indent=2))
        out_path = os.environ.get(
            "OUT_JSON",
            "save_results/optimization_study/formc_pack_out_lse_micro.json",
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out_path}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
