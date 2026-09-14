#!/usr/bin/env python3
"""Precise gather-Q diagnosis: CUDA events wrap ONLY the gather (not Flash).

Answers:
  - How much does torch.cat really cost after Flash?
  - How much of all_gather_q time is rank-skew wait vs NCCL?
  - Why E2E ~5.5ms/call while isolate NCCL ~1.5ms?
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

try:
    import flash_attn
except ImportError:
    flash_attn = None


def _phase_ms(prep_fn, timed_fn, iters: int, warmup: int) -> float:
    """Run prep_fn then time only timed_fn (CUDA events)."""
    for _ in range(warmup):
        prep_fn()
        timed_fn()
    torch.cuda.synchronize()
    dist.barrier()
    total = 0.0
    for _ in range(iters):
        prep_fn()
        torch.cuda.synchronize()  # finish prep so timed region is clean
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        timed_fn()
        e.record()
        torch.cuda.synchronize()
        total += float(s.elapsed_time(e))
    return total / iters


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    if ws < 2:
        return 1
    if flash_attn is None:
        return 1

    Ql, H, D = 1170, 12, 128
    Kl = 4680  # ~chunk-1 KV stripe size
    Q = Ql * ws
    iters, warmup = 60, 8
    scale = D ** -0.5

    q_local = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_full = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    gathered = [torch.empty_like(q_local) for _ in range(ws)]
    # warm gather once
    dist.all_gather_into_tensor(q_full, q_local, group=group)

    def flash() -> None:
        qb = q_full.unsqueeze(0).contiguous()
        flash_attn.flash_attn_interface._flash_attn_forward(
            qb, k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous(),
            dropout_p=0.0, softmax_scale=scale, causal=False,
            window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, return_softmax=False,
        )

    def noop() -> None:
        return None

    def gather_into() -> None:
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    def gather_list() -> None:
        dist.all_gather(gathered, q_local, group=group)

    def gather_list_cat() -> None:
        dist.all_gather(gathered, q_local, group=group)
        _ = torch.cat(gathered, dim=0)

    def only_cat() -> None:
        _ = torch.cat(gathered, dim=0)

    # FFN-like imbalance: all ranks flash once; slow ranks do extra GEMM
    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Ql, 1536, device="cuda", dtype=torch.bfloat16)

    def flash_balanced() -> None:
        flash()

    def flash_imbalanced() -> None:
        flash()
        # ranks 1..ws-1 do extra matmul → they arrive late at next gather
        if rank != 0:
            for _ in range(2):
                x2 = x @ w
                x.copy_(x2)

    def barrier() -> None:
        dist.barrier(group=group)

    results = {
        "world_size": ws,
        "shape": {"q_local": [Ql, H, D], "k_local": [Kl, H, D]},
        "iters": iters,
        "gather_only_ms": {
            # Pure NCCL, ranks already synced by synchronize+barrier in harness
            "into_after_idle": _phase_ms(noop, gather_into, iters, warmup),
            "list_after_idle": _phase_ms(noop, gather_list, iters, warmup),
            "list_plus_cat_after_idle": _phase_ms(noop, gather_list_cat, iters, warmup),
            "cat_only_after_list": _phase_ms(gather_list, only_cat, iters, warmup),
            # After Flash (memory/cache/SM warm), ranks still balanced if all flash same
            "into_after_flash_balanced": _phase_ms(flash_balanced, gather_into, iters, warmup),
            "list_plus_cat_after_flash_balanced": _phase_ms(flash_balanced, gather_list_cat, iters, warmup),
            # After imbalanced FFN-like work (no barrier): gather absorbs wait
            "into_after_imbalance_no_barrier": _phase_ms(flash_imbalanced, gather_into, iters, warmup),
            # Same imbalance but barrier first → pure NCCL after align
            "into_after_imbalance_then_barrier": _phase_ms(
                lambda: (flash_imbalanced(), barrier()), gather_into, iters, warmup
            ),
        },
    }
    g = results["gather_only_ms"]
    results["derived"] = {
        "cat_cost_ms_idle": g["list_plus_cat_after_idle"] - g["list_after_idle"],
        "cat_cost_ms_after_flash": g["list_plus_cat_after_flash_balanced"] - g["into_after_flash_balanced"],
        "into_vs_list_idle": g["list_after_idle"] - g["into_after_idle"],
        "skew_wait_in_gather_ms": (
            g["into_after_imbalance_no_barrier"] - g["into_after_imbalance_then_barrier"]
        ),
        "e2e_like_gather_ms": g["into_after_imbalance_no_barrier"],
        "pure_nccl_after_align_ms": g["into_after_imbalance_then_barrier"],
    }

    if rank == 0:
        print(json.dumps(results, indent=2))
        out = "save_results/optimization_study/sf_gather_q_precise_diag.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", out)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
