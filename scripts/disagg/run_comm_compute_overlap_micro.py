#!/usr/bin/env python3
"""Micro: P2P AR vs NCCL AR overlapped with GEMM on separate streams."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from lightx2v.common.ops.tp_p2p_allreduce import get_tp_p2p_allreduce


def _cuda_time(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    # Wan O partial size
    n = 25159680
    ar = torch.randn(n // 2, device=dev, dtype=torch.float16)
    gemm = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    comm = torch.cuda.Stream()
    comp = torch.cuda.Stream()
    ex = get_tp_p2p_allreduce(group)

    def serial_nccl() -> None:
        dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=False)
        with torch.cuda.stream(comp):
            for _ in range(6):
                torch.matmul(gemm, gemm)

    def overlap_nccl() -> None:
        with torch.cuda.stream(comm):
            work = dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=True)
        with torch.cuda.stream(comp):
            for _ in range(6):
                torch.matmul(gemm, gemm)
        if work is not None:
            work.wait()

    def serial_p2p() -> None:
        ex.all_reduce(ar, comm, async_op=False)
        with torch.cuda.stream(comp):
            for _ in range(6):
                torch.matmul(gemm, gemm)

    def overlap_p2p() -> None:
        with torch.cuda.stream(comm):
            work = ex.all_reduce(ar, comm, async_op=True)
        with torch.cuda.stream(comp):
            for _ in range(6):
                torch.matmul(gemm, gemm)
        if work is not None:
            work.wait()

    results = {
        "serial_nccl_ms": _cuda_time(serial_nccl),
        "overlap_nccl_ms": _cuda_time(overlap_nccl),
        "serial_p2p_ms": _cuda_time(serial_p2p),
        "overlap_p2p_ms": _cuda_time(overlap_p2p),
    }
    if rank == 0:
        for k, v in results.items():
            print(f"{k}: {v:.3f}")
        print(f"nccl_speedup: {results['serial_nccl_ms']/max(results['overlap_nccl_ms'],0.001):.2f}x")
        print(f"p2p_speedup: {results['serial_p2p_ms']/max(results['overlap_p2p_ms'],0.001):.2f}x")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
