#!/usr/bin/env python3
"""Check if NCCL AR on comm_stream stalls compute_stream micro-steps (real AR size)."""
import statistics

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    ar = torch.randn(4096, 2560, device=dev, dtype=torch.bfloat16)
    comm = torch.cuda.Stream()
    comp = torch.cuda.Stream()
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.float16)

    def gemm_steps_only(n: int = 8) -> list[float]:
        out: list[float] = []
        with torch.cuda.stream(comp):
            for _ in range(n):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(comp)
                torch.matmul(a, b)
                e.record(comp)
                comp.synchronize()
                out.append(s.elapsed_time(e))
        return out

    baseline = gemm_steps_only()

    during: list[float] = []
    with torch.cuda.stream(comm):
        work = dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=group, async_op=True)
    with torch.cuda.stream(comp):
        for _ in range(8):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(comp)
            torch.matmul(a, b)
            e.record(comp)
            comp.synchronize()
            during.append(s.elapsed_time(e))
    if work is not None:
        work.wait()

    if rank == 0:
        bmean = statistics.mean(baseline)
        dmean = statistics.mean(during)
        print(
            {
                "gemm_step_ms_baseline": bmean,
                "gemm_step_ms_during_nccl": dmean,
                "slowdown_ratio": dmean / max(bmean, 0.001),
                "nccl_blocked_compute": dmean > bmean * 1.5,
            }
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
