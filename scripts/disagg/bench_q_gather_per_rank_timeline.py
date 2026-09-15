#!/usr/bin/env python3
"""Per-rank timeline: when does each rank obtain Q_full from all_gather?

Hypothesis under test: Q gather feels slow because ranks serialize / wait on
each other exchanging large Q payloads.

Reports (P=4, CVD=0,1,2,4):
  1. Aligned: barrier then all_gather — per-rank CUDA ms until Q_full ready
  2. Entry skew: rank r burns r*skew_ms before gather — per-rank wall from
     global t0, and CUDA gather duration (includes wait for late peers)
  3. Gather pairwise Q chunks via sequential P2P (r sends to all, ordered) —
     per-hop times, to see if mesh itself serializes

No per-layer dist.barrier spam.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "3" in {x.strip() for x in vis.split(",") if x.strip()}:
        raise RuntimeError("GPU3 forbidden — use CVD=0,1,2,4 for P=4")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    g = dist.group.WORLD

    Ql, H, D = 1170, 12, 128
    Q = Ql * ws
    iters, warmup = 40, 8
    skew_step_ms = float(os.environ.get("SKEW_STEP_MS", "1.5"))

    q_local = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    q_full = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    # per-peer recv bufs for sequential P2P assemble
    peer_bufs = [torch.empty_like(q_local) for _ in range(ws)]

    bytes_local = q_local.numel() * q_local.element_size()
    bytes_full = q_full.numel() * q_full.element_size()

    def burn_ms(ms: float) -> None:
        if ms <= 0:
            return
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        while True:
            torch.mm(a, b)
            e.record()
            e.synchronize()
            if s.elapsed_time(e) >= ms:
                break

    def gather_once() -> None:
        dist.all_gather_into_tensor(q_full, q_local, group=g)

    def time_cuda_gather() -> float:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        gather_once()
        e.record()
        torch.cuda.synchronize()
        return float(s.elapsed_time(e))

    # --- Arm A: aligned gather (barrier only before arm) ---
    for _ in range(warmup):
        gather_once()
    torch.cuda.synchronize()
    dist.barrier()
    aligned_samples = [time_cuda_gather() for _ in range(iters)]
    aligned_local = sum(aligned_samples) / len(aligned_samples)

    # --- Arm B: staggered entry — rank r burns r * skew_step_ms ---
    for _ in range(warmup):
        burn_ms(rank * skew_step_ms)
        gather_once()
    torch.cuda.synchronize()
    dist.barrier()
    wall_enter = []
    wall_exit = []
    cuda_gather = []
    for _ in range(iters):
        dist.barrier()  # common t0 each sample (arm only; not per-layer spam)
        t0 = time.perf_counter()
        burn_ms(rank * skew_step_ms)
        torch.cuda.synchronize()
        t_enter = (time.perf_counter() - t0) * 1000.0
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        gather_once()
        e.record()
        torch.cuda.synchronize()
        t_exit = (time.perf_counter() - t0) * 1000.0
        wall_enter.append(t_enter)
        wall_exit.append(t_exit)
        cuda_gather.append(float(s.elapsed_time(e)))
    skew_enter_mean = sum(wall_enter) / len(wall_enter)
    skew_exit_mean = sum(wall_exit) / len(wall_exit)
    skew_cuda_mean = sum(cuda_gather) / len(cuda_gather)

    # --- Arm C: build Q_full by sequential P2P (same order on all ranks) ---
    # Step k: exchange with partner (r+k)%P via isend/irecv of Q_local / partner Q
    def assemble_q_sequential() -> list[float]:
        """Return per-hop CUDA ms; hop0=copy local, hop k=exchange with (r+k)%P."""
        hop_ms: list[float] = []
        # local slice
        peer_bufs[rank].copy_(q_local)
        hop_ms.append(0.0)
        for k in range(1, ws):
            partner = (rank + k) % ws
            send = q_local
            recv = peer_bufs[partner]
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            ops = [
                dist.P2POp(dist.isend, send, partner, group=g),
                dist.P2POp(dist.irecv, recv, partner, group=g),
            ]
            # Deadlock-free only if partner at same k receives from us:
            # at step k, we talk to (r+k); they have k' such that (partner+k')%P=r
            # => k' = (r - partner) % P = (r - (r+k)) % P = (-k)%P = P-k
            # Different k on different ranks → DEADLOCK with naive loop.
            # Use XOR rounds instead when power-of-two.
            for req in dist.batch_isend_irecv(ops):
                req.wait()
            e.record()
            torch.cuda.synchronize()
            hop_ms.append(float(s.elapsed_time(e)))
        return hop_ms

    # Use XOR partner rounds for deadlock-free per-hop timing (power-of-2)
    def assemble_q_xor_hops() -> tuple[float, list[float]]:
        hop_ms: list[float] = []
        peer_bufs[rank].copy_(q_local)
        # total from start to Q_full
        s_all = torch.cuda.Event(enable_timing=True)
        e_all = torch.cuda.Event(enable_timing=True)
        s_all.record()
        for k in range(1, ws):
            partner = rank ^ k
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            ops = [
                dist.P2POp(dist.isend, q_local, partner, group=g),
                dist.P2POp(dist.irecv, peer_bufs[partner], partner, group=g),
            ]
            for req in dist.batch_isend_irecv(ops):
                req.wait()
            e.record()
            torch.cuda.synchronize()
            hop_ms.append(float(s.elapsed_time(e)))
        # cat into q_full layout
        for r in range(ws):
            q_full[r * Ql : (r + 1) * Ql].copy_(peer_bufs[r])
        e_all.record()
        torch.cuda.synchronize()
        return float(s_all.elapsed_time(e_all)), hop_ms

    xor_ok = ws > 0 and (ws & (ws - 1)) == 0
    xor_total_local = 0.0
    xor_hops_acc = [0.0] * (ws - 1)
    if xor_ok:
        for _ in range(warmup):
            assemble_q_xor_hops()
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(iters):
            tot, hops = assemble_q_xor_hops()
            xor_total_local += tot
            for i, h in enumerate(hops):
                xor_hops_acc[i] += h
        xor_total_local /= iters
        xor_hops_acc = [h / iters for h in xor_hops_acc]

    # Collect all ranks → rank0
    def gather_float(x: float) -> list[float]:
        t = torch.tensor([x], device="cuda", dtype=torch.float64)
        out = [torch.empty_like(t) for _ in range(ws)]
        dist.all_gather(out, t, group=g)
        return [float(v.item()) for v in out]

    aligned_all = gather_float(aligned_local)
    skew_enter_all = gather_float(skew_enter_mean)
    skew_exit_all = gather_float(skew_exit_mean)
    skew_cuda_all = gather_float(skew_cuda_mean)
    xor_total_all = gather_float(xor_total_local) if xor_ok else []

    # hops: gather vector
    xor_hops_by_rank: list[list[float]] = []
    if xor_ok:
        hops_t = torch.tensor(xor_hops_acc, device="cuda", dtype=torch.float64)
        hops_list = [torch.empty_like(hops_t) for _ in range(ws)]
        dist.all_gather(hops_list, hops_t, group=g)
        xor_hops_by_rank = [v.cpu().tolist() for v in hops_list]

    out = {
        "cvd": vis,
        "world_size": ws,
        "payload": {
            "Ql": Ql,
            "H": H,
            "D": D,
            "Q_full": Q,
            "bytes_q_local": bytes_local,
            "bytes_q_full": bytes_full,
        },
        "aligned_allgather_ms_by_rank": [round(x, 3) for x in aligned_all],
        "aligned_allgather_ms_max": round(max(aligned_all), 3),
        "aligned_allgather_ms_min": round(min(aligned_all), 3),
        "aligned_spread_ms": round(max(aligned_all) - min(aligned_all), 3),
        "skew_step_ms": skew_step_ms,
        "skew_enter_wall_ms_by_rank": [round(x, 3) for x in skew_enter_all],
        "skew_exit_wall_ms_by_rank_when_qfull": [round(x, 3) for x in skew_exit_all],
        "skew_cuda_gather_ms_by_rank": [round(x, 3) for x in skew_cuda_all],
        "skew_note": (
            "exit≈when that rank has Q_full. Early ranks' cuda_gather includes "
            "wait for late peers; exit times should cluster near the slowest entry + NCCL."
        ),
        "xor_p2p_assemble_ms_by_rank": [round(x, 3) for x in xor_total_all] if xor_ok else None,
        "xor_p2p_hop_ms_by_rank": (
            [[round(h, 3) for h in row] for row in xor_hops_by_rank] if xor_ok else None
        ),
        "interpretation": None,
    }

    # Interpretation on rank0
    if rank == 0:
        slowest_exit = max(skew_exit_all)
        earliest_enter = min(skew_enter_all)
        wait_on_fast = skew_cuda_all[skew_enter_all.index(earliest_enter)]
        out["interpretation"] = {
            "aligned_ranks_finish_together": out["aligned_spread_ms"] < 0.2,
            "aligned_says": (
                "If spread≈0 and time~1.5ms: collective itself is the cost "
                "(large payload), not cross-rank start skew."
            ),
            "skew_fast_rank_cuda_ms": round(wait_on_fast, 3),
            "skew_slowest_exit_ms": round(slowest_exit, 3),
            "skew_says": (
                "Fast rank's all_gather CUDA time rises by ~late_peer_delay; "
                "all ranks get Q_full only after the last peer enters + transfer."
            ),
            "xor_vs_allgather": (
                None
                if not xor_ok
                else {
                    "allgather_max_ms": out["aligned_allgather_ms_max"],
                    "xor_max_ms": round(max(xor_total_all), 3),
                    "note": (
                        "XOR hops are serial P-1 exchanges; if sum(hops)>>allgather, "
                        "NCCL all_gather is already parallelizing better than pure serial mesh."
                    ),
                }
            ),
        }
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_q_gather_per_rank_timeline.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
