#!/usr/bin/env python3
"""Synthetic: async NCCL AR on comm_stream vs GEMM on compute_stream (TP=2)."""
import json
import statistics

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    ar = torch.randn(8192, 5120, device=dev, dtype=torch.float16)
    gemm = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    comm_stream = torch.cuda.Stream(device=dev)
    compute_stream = torch.cuda.Stream(device=dev)

    def serial() -> None:
        dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=False)
        for _ in range(8):
            torch.matmul(gemm, gemm)

    def overlap() -> None:
        with torch.cuda.stream(comm_stream):
            work = dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=True)
        with torch.cuda.stream(compute_stream):
            for _ in range(8):
                torch.matmul(gemm, gemm)
        if work is not None:
            work.wait()

    def ms(fn) -> float:
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    serial_samples = [ms(serial) for _ in range(5)]
    overlap_samples = [ms(overlap) for _ in range(5)]
    out = {
        "serial_ms_mean": statistics.mean(serial_samples),
        "overlap_ms_mean": statistics.mean(overlap_samples),
        "speedup": statistics.mean(serial_samples) / statistics.mean(overlap_samples),
        "device": torch.cuda.get_device_name(rank),
    }
    if rank == 0:
        from pathlib import Path
        Path("save_results/nsys/synthetic_ar_gemm_control.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
