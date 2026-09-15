#!/usr/bin/env python3
"""Isolate why stripe all_gather_q is slower than Ulysses all2all_q.

Compares (P=4, Q_local=[1170,12,128] bf16):
  - barrier only
  - all_gather (stripe Q path)
  - all2all_seq2head (Ulysses Q path)
  - all_to_all partial-out sized exchange
  - all_gather full partial out (baseline stripe)
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


def _cuda_event_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
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
    total = sum(float(s.elapsed_time(e)) for s, e in zip(starts, ends, strict=True))
    return total / iters


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

    Ql, H, D = 1170, 12, 128
    Q = Ql * ws
    iters = 150
    warmup = 20

    q_local = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    gathered_reuse = [torch.empty_like(q_local) for _ in range(ws)]

    def barrier_only() -> None:
        dist.barrier(group=group)

    def gather_q_alloc_each() -> None:
        bufs = [torch.empty_like(q_local) for _ in range(ws)]
        dist.all_gather(bufs, q_local, group=group)

    def gather_q_into() -> None:
        q_full = torch.empty((Ql * ws, H, D), device="cuda", dtype=torch.bfloat16)
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    def gather_q() -> None:
        dist.all_gather(gathered_reuse, q_local, group=group)

    def gather_q_with_cat() -> None:
        dist.all_gather(gathered_reuse, q_local, group=group)
        _ = torch.cat(gathered_reuse, dim=0)

    def ulysses_a2a_q() -> None:
        _ = all2all_seq2head(q_local, group=group)

    block_out = torch.randn(1, Q, H, D, device="cuda", dtype=torch.bfloat16)
    out_chunks = torch.split(block_out, [Ql] * ws, dim=1)
    send_out = [c.contiguous() for c in out_chunks]
    recv_out = [torch.empty_like(send_out[rank]) for _ in range(ws)]

    def formc_a2a_out() -> None:
        dist.all_to_all(recv_out, send_out, group=group)

    full_out = torch.randn(1, Q, H, D, device="cuda", dtype=torch.bfloat16)
    out_list = [torch.empty_like(full_out) for _ in range(ws)]

    def baseline_gather_out() -> None:
        dist.all_gather(out_list, full_out.contiguous(), group=group)

    results = {
        "world_size": ws,
        "q_local_shape": [Ql, H, D],
        "iters": iters,
        "bytes_q_local_send_bf16": Ql * H * D * 2,
        "bytes_q_full_recv_bf16": Q * H * D * 2,
        "barrier_ms_per_call": _cuda_event_ms(barrier_only, iters, warmup),
        "all_gather_q_ms_per_call": _cuda_event_ms(gather_q, iters, warmup),
        "all_gather_into_q_ms_per_call": _cuda_event_ms(gather_q_into, iters, warmup),
        "all_gather_q_alloc_each_ms_per_call": _cuda_event_ms(gather_q_alloc_each, iters, warmup),
        "all_gather_q_with_cat_ms_per_call": _cuda_event_ms(gather_q_with_cat, iters, warmup),
        "ulysses_all2all_q_ms_per_call": _cuda_event_ms(ulysses_a2a_q, iters, warmup),
        "formc_alltoall_out_ms_per_call": _cuda_event_ms(formc_a2a_out, iters, warmup),
        "baseline_gather_out_ms_per_call": _cuda_event_ms(baseline_gather_out, iters, warmup),
    }

    if rank == 0:
        print(json.dumps(results, indent=2))
        print("\n=== summary ===")
        for k in [
            "barrier_ms_per_call",
            "all_gather_q_ms_per_call",
            "ulysses_all2all_q_ms_per_call",
            "formc_alltoall_out_ms_per_call",
            "baseline_gather_out_ms_per_call",
        ]:
            print(f"{k}: {results[k]:.3f} ms")
        ratio = results["all_gather_q_ms_per_call"] / max(results["ulysses_all2all_q_ms_per_call"], 1e-6)
        print(f"all_gather_q / ulysses_a2a_q ratio: {ratio:.2f}x")
        overhead = results["all_gather_q_ms_per_call"] - results["barrier_ms_per_call"]
        print(f"all_gather_q minus barrier: {overhead:.3f} ms (pure comm estimate)")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
