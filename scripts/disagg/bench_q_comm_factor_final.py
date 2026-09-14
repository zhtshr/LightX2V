#!/usr/bin/env python3
"""Payload vs skew for Q-comm — NO dist.barrier in measurement arms.

Align ranks with an untimed all2all (same group), then time the op.
CUDA_VISIBLE_DEVICES=0,1,2,4 only.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


def main() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "3" in {x.strip() for x in vis.split(",") if x.strip()}:
        raise RuntimeError("GPU3 forbidden")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    g = dist.group.WORLD
    if ws != 4:
        return 1

    Ql, H, D = 1170, 12, 128
    Hp, Q = H // ws, Ql * ws
    iters, warmup = 120, 20

    q = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_full = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    q_thin = torch.randn(Ql, Hp, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_thin_full = torch.empty(Q, Hp, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Ql, 1536, device="cuda", dtype=torch.bfloat16)

    def gather12():
        dist.all_gather_into_tensor(q_full, q, group=g)

    def gather3():
        dist.all_gather_into_tensor(q_thin_full, q_thin, group=g)

    def a2a():
        all2all_seq2head(q, group=g)

    def align():
        # Untimed: syncs ranks without dist.barrier
        all2all_seq2head(q, group=g)

    def time_after_align(op) -> float:
        for _ in range(warmup):
            align()
            op()
        torch.cuda.synchronize()
        # one CPU-side barrier only at arm start (ranks already NCCL-synced by align)
        dist.barrier()
        acc = 0.0
        for _ in range(iters):
            align()  # untimed
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            op()
            e.record()
            torch.cuda.synchronize()
            acc += float(s.elapsed_time(e))
        return acc / iters

    def time_after_skew(op) -> float:
        def imb():
            if rank != 0:
                for _ in range(4):
                    x.copy_(x @ w)

        for _ in range(warmup):
            imb()
            op()
        torch.cuda.synchronize()
        dist.barrier()
        acc = 0.0
        for _ in range(iters):
            imb()
            torch.cuda.synchronize()  # local only
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            op()
            e.record()
            torch.cuda.synchronize()
            acc += float(s.elapsed_time(e))
        return acc / iters

    # Reduce max across ranks (rank0 is waiter under skew)
    def pack(local: float) -> float:
        t = torch.tensor([local], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    aligned = {
        "stripe_gather_H12": pack(time_after_align(gather12)),
        "gather_H3_same_recv_bytes": pack(time_after_align(gather3)),
        "ulysses_all2all": pack(time_after_align(a2a)),
    }
    skewed = {
        "stripe_gather_H12": pack(time_after_skew(gather12)),
        "ulysses_all2all": pack(time_after_skew(a2a)),
    }

    gap = aligned["stripe_gather_H12"] - aligned["ulysses_all2all"]
    bytes_part = aligned["stripe_gather_H12"] - aligned["gather_H3_same_recv_bytes"]
    kind_part = aligned["gather_H3_same_recv_bytes"] - aligned["ulysses_all2all"]
    wait_s = skewed["stripe_gather_H12"] - aligned["stripe_gather_H12"]
    wait_u = skewed["ulysses_all2all"] - aligned["ulysses_all2all"]

    # Prior E2E (Form C chunk6 / Ulysses chunk6)
    e2e_stripe = 832.462 / 150
    e2e_uly = 61.844 / 150

    out = {
        "cvd": vis,
        "aligned_ms": aligned,
        "skewed_ms": skewed,
        "decomp_aligned_ms": {
            "total_stripe_minus_ulysses": gap,
            "bytes_H12_vs_H3_gather": bytes_part,
            "allgather_vs_all2all_at_H3_bytes": kind_part,
            "pct_bytes_of_aligned_gap": 100.0 * bytes_part / max(gap, 1e-9),
            "pct_kind_of_aligned_gap": 100.0 * kind_part / max(gap, 1e-9),
            "ratio": aligned["stripe_gather_H12"] / max(aligned["ulysses_all2all"], 1e-9),
        },
        "decomp_skew_extra_ms": {
            "stripe_gather": wait_s,
            "ulysses_all2all": wait_u,
        },
        "e2e_reference_ms_per_call": {
            "stripe_all_gather_q": e2e_stripe,
            "ulysses_all2all_q": e2e_uly,
            "ratio": e2e_stripe / e2e_uly,
        },
        "verdict": None,
    }

    # Primary: largest share of aligned gap; compare to E2E
    if abs(bytes_part) >= abs(kind_part):
        primary = "recv_bytes (full H=12 gather vs H/P-sized tensor)"
    else:
        primary = "collective kind (all_gather vs all2all at equal bytes)"

    # Can aligned gap alone explain E2E gap?
    e2e_gap = e2e_stripe - e2e_uly
    aligned_explains_pct = 100.0 * gap / max(e2e_gap, 1e-9)
    out["verdict"] = {
        "primary_for_aligned_gap": primary,
        "aligned_gap_ms": round(gap, 3),
        "e2e_gap_ms": round(e2e_gap, 3),
        "aligned_explains_pct_of_e2e_gap": round(aligned_explains_pct, 1),
        "skew_extra_stripe_ms": round(wait_s, 3),
        "skew_extra_ulysses_ms": round(wait_u, 3),
        "summary": (
            f"Aligned: stripe gather {aligned['stripe_gather_H12']:.2f}ms vs "
            f"ulysses {aligned['ulysses_all2all']:.2f}ms "
            f"({aligned['stripe_gather_H12']/max(aligned['ulysses_all2all'],1e-9):.1f}x). "
            f"Of the {gap:.2f}ms gap: {bytes_part:.2f}ms from 4x recv bytes, "
            f"{kind_part:.2f}ms from allgather vs all2all. "
            f"E2E gap was {e2e_gap:.2f}ms; aligned structural gap is "
            f"{aligned_explains_pct:.0f}% of that."
        ),
    }

    if rank == 0:
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_q_comm_factor_final.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", path)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
