#!/usr/bin/env python3
"""Clean A/B: Stripe gather_Q vs Ulysses all2all_Q — payload vs skew.

Uses CUDA_VISIBLE_DEVICES=0,1,2,4 only (never GPU3).
No barrier inside the timed region. One warmup barrier before each arm only.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


def _refuse_gpu3() -> None:
    ids = {x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()}
    if "3" in ids:
        raise RuntimeError("GPU3 forbidden")


def _bench(fn, iters: int, warmup: int) -> tuple[float, float]:
    """Return (mean_ms_this_rank, mean_ms_max_across_ranks)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()  # once, before arm — not timed
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()  # align ALL ranks before EACH sample (untimed)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(float(s.elapsed_time(e)))
    local = sum(times) / len(times)
    t = torch.tensor([local], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return local, float(t.item())


def _bench_skew(fn, imbalance, iters: int, warmup: int) -> tuple[float, float]:
    """Imbalance then immediate collective (no barrier between). Report max over ranks."""
    for _ in range(warmup):
        imbalance()
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        imbalance()
        # local sync only — other ranks may still be in GEMM
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(float(s.elapsed_time(e)))
    local = sum(times) / len(times)
    t = torch.tensor([local], device="cuda")
    # For skew: rank0 (fast) sees wait; take MAX across ranks for "how slow Q-comm looks"
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return local, float(t.item())


def main() -> int:
    _refuse_gpu3()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    if ws != 4:
        if rank == 0:
            print("need P=4", file=sys.stderr)
        return 1

    Ql, H, D = 1170, 12, 128
    Hp, Q = H // ws, Ql * ws
    iters, warmup = 100, 15

    q = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_full = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    q_thin = torch.randn(Ql, Hp, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_thin_full = torch.empty(Q, Hp, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Ql, 1536, device="cuda", dtype=torch.bfloat16)

    def gather_H12():
        dist.all_gather_into_tensor(q_full, q, group=dist.group.WORLD)

    def gather_H3():
        dist.all_gather_into_tensor(q_thin_full, q_thin, group=dist.group.WORLD)

    def a2a_ulysses():
        all2all_seq2head(q, group=dist.group.WORLD)

    def imb():
        if rank != 0:
            for _ in range(4):
                x.copy_(x @ w)

    # --- Arm A: aligned (barrier before each sample, outside timed region) ---
    a_g12_loc, a_g12 = _bench(gather_H12, iters, warmup)
    a_g3_loc, a_g3 = _bench(gather_H3, iters, warmup)
    a_a2a_loc, a_a2a = _bench(a2a_ulysses, iters, warmup)

    # --- Arm B: same skew before either collective ---
    s_g12_loc, s_g12 = _bench_skew(gather_H12, imb, iters, warmup)
    s_a2a_loc, s_a2a = _bench_skew(a2a_ulysses, imb, iters, warmup)

    out = {
        "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "bytes": {
            "stripe_gather_recv": Q * H * D * 2,
            "thin_gather_recv": Q * Hp * D * 2,
            "ulysses_a2a_xfer": Ql * H * D * 2,
        },
        "aligned_ms_maxrank": {
            "stripe_allgather_H12": a_g12,
            "allgather_H3_same_bytes_as_ulysses": a_g3,
            "ulysses_all2all": a_a2a,
        },
        "skew_ms_maxrank": {
            "stripe_allgather_H12": s_g12,
            "ulysses_all2all": s_a2a,
        },
        "rank0_local_ms": {
            "aligned_gather_H12": a_g12_loc,
            "aligned_a2a": a_a2a_loc,
            "skew_gather_H12": s_g12_loc,
            "skew_a2a": s_a2a_loc,
        },
    }
    A = out["aligned_ms_maxrank"]
    S = out["skew_ms_maxrank"]
    # Contributions relative to ulysses aligned baseline
    gap_payload = A["stripe_allgather_H12"] - A["ulysses_all2all"]
    gap_bytes_only = A["stripe_allgather_H12"] - A["allgather_H3_same_bytes_as_ulysses"]
    gap_algo_same_bytes = A["allgather_H3_same_bytes_as_ulysses"] - A["ulysses_all2all"]
    wait_g = S["stripe_allgather_H12"] - A["stripe_allgather_H12"]
    wait_a = S["ulysses_all2all"] - A["ulysses_all2all"]
    out["decomposition_ms"] = {
        "aligned_stripe_minus_ulysses": gap_payload,
        "of_which_4x_bytes_vs_thin_gather": gap_bytes_only,
        "of_which_allgather_vs_all2all_at_same_bytes": gap_algo_same_bytes,
        "extra_wait_when_skewed_on_stripe_gather": wait_g,
        "extra_wait_when_skewed_on_ulysses_a2a": wait_a,
        "ratio_aligned_stripe_over_ulysses": A["stripe_allgather_H12"] / max(A["ulysses_all2all"], 1e-9),
        "ratio_skew_stripe_over_ulysses": S["stripe_allgather_H12"] / max(S["ulysses_all2all"], 1e-9),
    }
    # Primary cause by magnitude for ALIGNED gap (explains structural difference)
    parts = {
        "bytes_4x_full_H_vs_HdivP": abs(gap_bytes_only),
        "collective_kind_allgather_vs_all2all": abs(gap_algo_same_bytes),
    }
    primary_aligned = max(parts, key=parts.get)
    out["verdict"] = {
        "aligned_primary_factor": primary_aligned,
        "aligned_parts_ms": parts,
        "skew_note": (
            "same intentional imbalance added "
            f"{wait_g:.2f}ms to stripe gather vs {wait_a:.2f}ms to ulysses a2a (max-rank)"
        ),
        "e2e_implication": (
            "Structural gap when ranks aligned is "
            f"{gap_payload:.2f}ms ({out['decomposition_ms']['ratio_aligned_stripe_over_ulysses']:.1f}x). "
            "If E2E ratio >> this, residual is path skew / measurement; "
            "if E2E ratio ≈ this, payload/algorithm is sufficient explanation."
        ),
    }

    if rank == 0:
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_q_comm_factor_clean.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", path)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
