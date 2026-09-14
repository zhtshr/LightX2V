#!/usr/bin/env python3
"""Diagnose why all_gather_q is slow in E2E vs isolation.

Measures (P=4, Q_local=[1170,12,128] bf16):
  1. back-to-back all_gather_into_tensor
  2. barrier then all_gather_into (pure NCCL after align)
  3. Flash(Q_full,K) then all_gather (GPU busy → contend)
  4. asymmetric GPU work then all_gather (skew wait absorbed in gather)
  5. same-as-E2E: flash then list-all_gather + cat
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


def _ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    starts, ends = [], []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        starts.append(s)
        ends.append(e)
    torch.cuda.synchronize()
    return sum(float(a.elapsed_time(b)) for a, b in zip(starts, ends, strict=True)) / iters


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    if ws != 4:
        if rank == 0:
            print("need P=4", file=sys.stderr)
        return 1
    if flash_attn is None:
        if rank == 0:
            print("flash_attn required", file=sys.stderr)
        return 1

    Ql, H, D, Kl = 1170, 12, 128, 4680
    Q = Ql * ws
    iters, warmup = 80, 10
    q_local = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_full_buf = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    scale = D ** -0.5

    def gather_into() -> None:
        dist.all_gather_into_tensor(q_full_buf, q_local, group=group)

    def barrier_then_gather() -> None:
        dist.barrier(group=group)
        dist.all_gather_into_tensor(q_full_buf, q_local, group=group)

    def flash_once() -> None:
        # Use already-gathered-ish full Q content for compute pressure.
        qb = q_full_buf.unsqueeze(0).contiguous()
        kb = k.unsqueeze(0).contiguous()
        vb = v.unsqueeze(0).contiguous()
        flash_attn.flash_attn_interface._flash_attn_forward(
            qb, kb, vb,
            dropout_p=0.0, softmax_scale=scale, causal=False,
            window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, return_softmax=False,
        )

    # Seed q_full_buf once so Flash has real data.
    dist.all_gather_into_tensor(q_full_buf, q_local, group=group)

    def flash_then_gather() -> None:
        flash_once()
        dist.all_gather_into_tensor(q_full_buf, q_local, group=group)

    # Skew: rank 0 does extra flash before join.
    def skew_then_gather() -> None:
        if rank == 0:
            flash_once()
            flash_once()
        dist.all_gather_into_tensor(q_full_buf, q_local, group=group)

    gathered = [torch.empty_like(q_local) for _ in range(ws)]

    def flash_then_gather_cat() -> None:
        flash_once()
        dist.all_gather(gathered, q_local, group=group)
        _ = torch.cat(gathered, dim=0)

    # Also measure: after flash+barrier, only gather (skew removed).
    def flash_barrier_gather() -> None:
        flash_once()
        dist.barrier(group=group)
        dist.all_gather_into_tensor(q_full_buf, q_local, group=group)

    results = {
        "q_local": [Ql, H, D],
        "iters": iters,
        "ms_per_call": {
            "1_back_to_back_gather_into": _ms(gather_into, iters, warmup),
            "2_barrier_then_gather_into": _ms(barrier_then_gather, iters, warmup),
            "3_flash_then_gather_into": _ms(flash_then_gather, iters, warmup),
            "4_skew_rank0_extra_flash_then_gather": _ms(skew_then_gather, iters, warmup),
            "5_flash_barrier_then_gather_into": _ms(flash_barrier_gather, iters, warmup),
            "6_flash_then_gather_list_plus_cat": _ms(flash_then_gather_cat, iters, warmup),
            "flash_only": _ms(flash_once, iters, warmup),
        },
    }
    # Derived: how much of flash+gather is flash vs gather
    m = results["ms_per_call"]
    results["derived"] = {
        "gather_after_flash_minus_flash": m["3_flash_then_gather_into"] - m["flash_only"],
        "cat_path_extra_vs_into_after_flash": (
            m["6_flash_then_gather_list_plus_cat"] - m["3_flash_then_gather_into"]
        ),
        "skew_wait_estimate": m["4_skew_rank0_extra_flash_then_gather"] - m["1_back_to_back_gather_into"],
        "post_flash_pure_nccl_via_barrier": m["5_flash_barrier_then_gather_into"] - m["flash_only"],
    }

    if rank == 0:
        print(json.dumps(results, indent=2))
        out = "save_results/optimization_study/sf_gather_q_diag_isolate.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
