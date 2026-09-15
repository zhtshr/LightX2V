#!/usr/bin/env python3
"""Factorize Stripe vs Ulysses Q-comm: payload/algorithm vs rank-skew.

Physical GPUs: CUDA_VISIBLE_DEVICES must skip GPU3 (use 0,1,2,4).
No per-layer barrier spam — optional one barrier only to measure "aligned" arm.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


def _assert_no_gpu3() -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "3" in {x.strip() for x in vis.split(",") if x.strip()}:
        raise RuntimeError(f"refuse GPU3 in CUDA_VISIBLE_DEVICES={vis!r}")


def _time_only(prep, timed, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        prep()
        timed()
    torch.cuda.synchronize()
    dist.barrier()  # once before measured loop (align ranks for fair start)
    total = 0.0
    for _ in range(iters):
        prep()
        torch.cuda.synchronize()
        # Do NOT barrier here for the "skew" arm — caller encodes skew in prep.
        # For "aligned" arm, prep ends with barrier (see below).
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        timed()
        e.record()
        torch.cuda.synchronize()
        total += float(s.elapsed_time(e))
    dist.barrier()
    return total / iters


def main() -> int:
    _assert_no_gpu3()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    if ws != 4:
        if rank == 0:
            print("need nproc=4 with CUDA_VISIBLE_DEVICES=0,1,2,4", file=sys.stderr)
        return 1

    Ql, H, D = 1170, 12, 128
    Hp = H // ws
    Q = Ql * ws
    iters, warmup = 80, 10
    bytes_local = Ql * H * D * 2
    bytes_full_h = Q * H * D * 2
    bytes_ulysses_out = Q * Hp * D * 2

    q_local = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_full = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    # same payload volume as Ulysses all2all result: gather of [Ql, Hp, D]
    q_thin = torch.randn(Ql, Hp, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_thin_full = torch.empty(Q, Hp, D, device="cuda", dtype=torch.bfloat16)

    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Ql, 1536, device="cuda", dtype=torch.bfloat16)

    def gather_full_h() -> None:
        dist.all_gather_into_tensor(q_full, q_local, group=group)

    def gather_thin_h() -> None:
        # recv bytes == Ulysses all2all out volume (full seq × H/P)
        dist.all_gather_into_tensor(q_thin_full, q_thin, group=group)

    def ulysses_a2a() -> None:
        _ = all2all_seq2head(q_local, group=group)

    def prep_noop() -> None:
        return None

    def prep_align() -> None:
        # Explicit align so timed op excludes skew wait.
        dist.barrier(group=group)

    def prep_imbalance() -> None:
        # Same imbalance for BOTH ops: rank0 idle-ish, others do 3 GEMMs.
        if rank != 0:
            for _ in range(3):
                y = x @ w
                x.copy_(y)

    # Arms
    aligned_gather_h = _time_only(prep_align, gather_full_h, iters, warmup)
    aligned_gather_thin = _time_only(prep_align, gather_thin_h, iters, warmup)
    aligned_a2a = _time_only(prep_align, ulysses_a2a, iters, warmup)

    # Skew arms: prep_imbalance has NO barrier before timed op.
    # Outer harness still barrier() once before the loop starts — each iter
    # then imbalances then times — so wait is only the imbalance within iter.
    skew_gather_h = _time_only(prep_imbalance, gather_full_h, iters, warmup)
    skew_a2a = _time_only(prep_imbalance, ulysses_a2a, iters, warmup)

    # Also: idle (only cuda sync before timed, no barrier in prep) back-to-back
    idle_gather_h = _time_only(prep_noop, gather_full_h, iters, warmup)
    idle_a2a = _time_only(prep_noop, ulysses_a2a, iters, warmup)

    out = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "world_size": ws,
        "shapes": {
            "q_local": [Ql, H, D],
            "bytes_q_local": bytes_local,
            "bytes_after_stripe_gather": bytes_full_h,
            "bytes_after_ulysses_a2a": bytes_ulysses_out,
            "bytes_after_thin_gather": Q * Hp * D * 2,
        },
        "ms_per_call": {
            "aligned_stripe_gather_full_H": aligned_gather_h,
            "aligned_gather_same_bytes_as_ulysses_out": aligned_gather_thin,
            "aligned_ulysses_all2all_q": aligned_a2a,
            "idle_stripe_gather_full_H": idle_gather_h,
            "idle_ulysses_all2all_q": idle_a2a,
            "skew_stripe_gather_full_H": skew_gather_h,
            "skew_ulysses_all2all_q": skew_a2a,
        },
        "derived": {},
    }
    m = out["ms_per_call"]
    d = out["derived"]
    d["ratio_aligned_gatherH_over_a2a"] = m["aligned_stripe_gather_full_H"] / max(
        m["aligned_ulysses_all2all_q"], 1e-9
    )
    d["ratio_aligned_thinGather_over_a2a"] = m[
        "aligned_gather_same_bytes_as_ulysses_out"
    ] / max(m["aligned_ulysses_all2all_q"], 1e-9)
    d["skew_wait_ms_on_gatherH"] = (
        m["skew_stripe_gather_full_H"] - m["aligned_stripe_gather_full_H"]
    )
    d["skew_wait_ms_on_a2a"] = m["skew_ulysses_all2all_q"] - m["aligned_ulysses_all2all_q"]
    d["payload_gap_ms_aligned"] = (
        m["aligned_stripe_gather_full_H"] - m["aligned_ulysses_all2all_q"]
    )
    d["payload_gap_vs_same_byte_gather"] = (
        m["aligned_stripe_gather_full_H"]
        - m["aligned_gather_same_bytes_as_ulysses_out"]
    )
    # Primary-cause verdict for THIS microbench (aligned ≈ algorithm; delta skew ≈ wait)
    d["verdict_aligned"] = (
        "payload_or_algorithm"
        if d["ratio_aligned_gatherH_over_a2a"] >= 1.5
        else "similar_when_aligned"
    )
    # If same imbalance adds similar wait to both, skew is not why *ratio* is large
    d["skew_wait_similar"] = abs(
        d["skew_wait_ms_on_gatherH"] - d["skew_wait_ms_on_a2a"]
    ) < 0.5
    if d["skew_wait_similar"] and d["ratio_aligned_gatherH_over_a2a"] >= 1.5:
        d["primary_cause"] = (
            "payload_algorithm (skew wait similar on both; gap remains when aligned)"
        )
    elif d["skew_wait_ms_on_gatherH"] > d["payload_gap_ms_aligned"]:
        d["primary_cause"] = "rank_skew_wait (extra wait on gather larger than aligned gap)"
    else:
        d["primary_cause"] = (
            "mixed: aligned payload gap "
            f"{d['payload_gap_ms_aligned']:.2f}ms + skew_extra_gather "
            f"{d['skew_wait_ms_on_gatherH']:.2f}ms vs skew_extra_a2a "
            f"{d['skew_wait_ms_on_a2a']:.2f}ms"
        )

    if rank == 0:
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_q_comm_factor_ab.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", path)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
