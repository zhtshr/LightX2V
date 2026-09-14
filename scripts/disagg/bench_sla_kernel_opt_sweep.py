#!/usr/bin/env python3
"""Sweep SLA triton forward optimizations vs baseline.

Optimizations under test:
  1) Autotune BLOCK_M/N, num_warps, num_stages
  2) tl.range(..., num_stages=) for LUT/K/V prefetch
  3) Always-mask path (drop partial-block if-branch)
  4) BHLD layout (skip per-call transpose+contiguous)
  5) Optional K pad to BLOCK_N for aligned loads

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/disagg/bench_sla_kernel_opt_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl

from lightx2v.common.ops.attn.kernels.sla_kernel import _attention as _attention_baseline
from lightx2v.common.ops.attn.utils.sla_util import get_block_map


@triton.jit
def _attn_fwd_opt(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    LSE,
    OS,
    L_Q: tl.constexpr,
    L_K: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    ALWAYS_MASK: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    q_offset = idx_bh * L_Q * D
    kv_offset = idx_bh * L_K * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L_Q
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + kv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L_Q)
    for block_idx in tl.range(topk, num_stages=LOOP_STAGES):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L_K - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        if ALWAYS_MASK:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
        else:
            if L_K - idx_n * BLOCK_N < BLOCK_N:
                qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L_Q)
    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L_Q)


def sla_fwd_opt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    topk: int,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    loop_stages: int,
    always_mask: bool,
) -> torch.Tensor:
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    B, H, L_Q, D = q.shape
    L_K = k.shape[2]
    qk_scale = D**-0.5
    m_blocks = triton.cdiv(L_Q, block_m)
    o_s = torch.empty((B, H, L_Q, D), device=q.device, dtype=v.dtype)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    grid = (m_blocks, B * H)
    _attn_fwd_opt[grid](
        q,
        k,
        v,
        qk_scale,
        topk,
        lut,
        lse,
        o_s,
        L_Q,
        L_K,
        m_blocks,
        D,
        block_m,
        block_n,
        loop_stages,
        always_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o_s


def _sync() -> None:
    torch.cuda.synchronize()


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / iters


def _attn_flops(q_len: int, k_sel_len: float, heads: int, head_dim: int) -> float:
    return 4.0 * q_len * k_sel_len * heads * head_dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=32130)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output_json",
        default="save_results/optimization_study/sla_kernel_opt_sweep.json",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller config grid")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    L, H, D = args.seq_len, args.heads, args.head_dim
    topk_ratio = 1.0 - args.sparsity
    torch.manual_seed(0)

    # NHD layout as model produces, then BHLD once (opt path benefit).
    q_nhd = torch.randn(L, H, D, device=device, dtype=dtype)
    k_nhd = torch.randn(L, H, D, device=device, dtype=dtype)
    v_nhd = torch.randn(L, H, D, device=device, dtype=dtype)
    q = q_nhd.unsqueeze(0).transpose(1, 2).contiguous()
    k = k_nhd.unsqueeze(0).transpose(1, 2).contiguous()
    v = v_nhd.unsqueeze(0).transpose(1, 2).contiguous()

    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=64, BLKK=64)
    k_blocks = sparse_map.shape[-1]
    k_sel_ratio = real_topk / max(k_blocks, 1)
    sparse_flops = _attn_flops(L, k_sel_ratio * L, H, D)

    # Baseline: production kernel with BLOCK 64/64
    def run_baseline():
        return _attention_baseline.apply(q, k, v, sparse_map, lut, real_topk, 64, 64)

    # Warm compile + correctness ref
    ref = run_baseline()
    base_ms = _bench(run_baseline, args.warmup, args.iters)
    base_tflops = sparse_flops / (base_ms * 1e-3) / 1e12

    # Also measure NHD->BHLD + baseline (end-to-end like sla.apply)
    def run_baseline_from_nhd():
        qq = q_nhd.unsqueeze(0).transpose(1, 2).contiguous()
        kk = k_nhd.unsqueeze(0).transpose(1, 2).contiguous()
        vv = v_nhd.unsqueeze(0).transpose(1, 2).contiguous()
        sm, lt, tk = get_block_map(qq, kk, topk_ratio=topk_ratio, BLKQ=64, BLKK=64)
        return _attention_baseline.apply(qq, kk, vv, sm, lt, tk, 64, 64)

    e2e_base_ms = _bench(run_baseline_from_nhd, args.warmup, max(3, args.iters // 2))

    if args.quick:
        grid = [
            # block_m, block_n, warps, stages, loop_stages, always_mask
            (64, 64, 8, 3, 1, False),  # baseline-equivalent launch
            (64, 64, 8, 3, 2, True),
            (64, 64, 8, 4, 2, True),
            (64, 64, 4, 4, 2, True),
            (128, 64, 8, 3, 2, True),
            (64, 128, 8, 3, 2, True),
            (128, 128, 8, 3, 2, True),
            (128, 64, 8, 4, 3, True),
            (64, 64, 8, 5, 3, True),
        ]
    else:
        grid = []
        for bm in (64, 128):
            for bn in (64, 128):
                for warps in (4, 8):
                    for stages in (2, 3, 4, 5):
                        for loop_s in (1, 2, 3, 4):
                            for am in (True,):
                                grid.append((bm, bn, warps, stages, loop_s, am))
        # include exact baseline-like for comparison
        grid.insert(0, (64, 64, 8, 3, 1, False))

    rows = []
    best = None
    print(
        f"baseline kernel: {base_ms:.2f} ms, {base_tflops:.1f} TFLOPS, "
        f"topk={real_topk}/{k_blocks}, e2e_with_map+layout={e2e_base_ms:.2f} ms"
    )

    for bm, bn, warps, stages, loop_s, am in grid:
        # skip configs that need more SMEM than A10 likely allows for D=128
        # rough filter: large blocks + many warps + high stages
        try:
            # Need lut rebuilt if block size changes? Block map is for 64/64.
            # Only fair to use BLK 64/64 with this lut; for 128 need rematerialize map.
            if bm != 64 or bn != 64:
                sm2, lut2, tk2 = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=bm, BLKK=bn)
            else:
                sm2, lut2, tk2 = sparse_map, lut, real_topk
                # keep same flops estimate roughly
            k_sel_r = tk2 / max(sm2.shape[-1], 1)
            flops = _attn_flops(L, k_sel_r * L, H, D)

            def run_opt(bm=bm, bn=bn, warps=warps, stages=stages, loop_s=loop_s, am=am, lut2=lut2, tk2=tk2):
                return sla_fwd_opt(
                    q,
                    k,
                    v,
                    lut2,
                    tk2,
                    block_m=bm,
                    block_n=bn,
                    num_warps=warps,
                    num_stages=stages,
                    loop_stages=loop_s,
                    always_mask=am,
                )

            out = run_opt()
            # Correctness vs baseline only when same block size
            ok = True
            max_err = 0.0
            if bm == 64 and bn == 64:
                # allow bf16 noise
                diff = (out.float() - ref.float()).abs()
                max_err = float(diff.max().item())
                mean_err = float(diff.mean().item())
                ok = mean_err < 5e-2 and max_err < 1.0
            else:
                mean_err = float("nan")

            ms = _bench(run_opt, args.warmup, args.iters)
            tflops = flops / (ms * 1e-3) / 1e12
            speedup = base_ms / ms
            row = {
                "block_m": bm,
                "block_n": bn,
                "num_warps": warps,
                "num_stages": stages,
                "loop_stages": loop_s,
                "always_mask": am,
                "ms": round(ms, 4),
                "tflops": round(tflops, 2),
                "speedup_vs_baseline_kernel": round(speedup, 3),
                "ok": ok,
                "max_err": None if math.isnan(max_err) else round(max_err, 5),
                "mean_err": None if math.isnan(mean_err) else round(mean_err, 5),
                "k_selected_ratio": round(k_sel_r, 4),
            }
            rows.append(row)
            tag = "OK" if ok else "DRIFT"
            print(
                f"[{tag}] BM={bm} BN={bn} warps={warps} stages={stages} loop={loop_s} "
                f"mask={am}: {ms:.2f} ms, {tflops:.1f} TFLOPS, {speedup:.3f}x"
            )
            if ok and (best is None or ms < best["ms"]):
                best = row
        except Exception as exc:
            print(f"[FAIL] BM={bm} BN={bn} warps={warps} stages={stages} loop={loop_s}: {exc}")
            rows.append(
                {
                    "block_m": bm,
                    "block_n": bn,
                    "num_warps": warps,
                    "num_stages": stages,
                    "loop_stages": loop_s,
                    "always_mask": am,
                    "error": str(exc),
                }
            )

    # Best config e2e with BHLD reuse + skip remap when possible
    e2e_best_ms = None
    if best is not None:
        bm, bn = best["block_m"], best["block_n"]
        if bm == 64 and bn == 64:
            sm2, lut2, tk2 = sparse_map, lut, real_topk
        else:
            sm2, lut2, tk2 = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=bm, BLKK=bn)

        def run_e2e_best():
            # layout convert once (still counted) + map + opt kernel
            qq = q_nhd.unsqueeze(0).transpose(1, 2).contiguous()
            kk = k_nhd.unsqueeze(0).transpose(1, 2).contiguous()
            vv = v_nhd.unsqueeze(0).transpose(1, 2).contiguous()
            sm, lt, tk = get_block_map(qq, kk, topk_ratio=topk_ratio, BLKQ=bm, BLKK=bn)
            return sla_fwd_opt(
                qq,
                kk,
                vv,
                lt,
                tk,
                block_m=bm,
                block_n=bn,
                num_warps=best["num_warps"],
                num_stages=best["num_stages"],
                loop_stages=best["loop_stages"],
                always_mask=best["always_mask"],
            )

        e2e_best_ms = _bench(run_e2e_best, args.warmup, max(3, args.iters // 2))

        # BHLD-cached path: map+kernel only (no layout), what production can do if weights stay BHLD
        def run_best_cached_layout():
            return sla_fwd_opt(
                q,
                k,
                v,
                lut2,
                tk2,
                block_m=bm,
                block_n=bn,
                num_warps=best["num_warps"],
                num_stages=best["num_stages"],
                loop_stages=best["loop_stages"],
                always_mask=best["always_mask"],
            )

        cached_ms = _bench(run_best_cached_layout, args.warmup, args.iters)
    else:
        cached_ms = None

    # Ablation: only always_mask / only loop_stages on 64/64/8/3
    ablations = []
    for name, conf in [
        ("baseline_equiv", dict(loop_stages=1, always_mask=False, num_stages=3, num_warps=8)),
        ("always_mask_only", dict(loop_stages=1, always_mask=True, num_stages=3, num_warps=8)),
        ("loop2_only", dict(loop_stages=2, always_mask=False, num_stages=3, num_warps=8)),
        ("loop2_mask", dict(loop_stages=2, always_mask=True, num_stages=3, num_warps=8)),
        ("loop3_mask_s4", dict(loop_stages=3, always_mask=True, num_stages=4, num_warps=8)),
    ]:
        try:
            def fn(conf=conf):
                return sla_fwd_opt(
                    q, k, v, lut, real_topk,
                    block_m=64, block_n=64,
                    num_warps=conf["num_warps"],
                    num_stages=conf["num_stages"],
                    loop_stages=conf["loop_stages"],
                    always_mask=conf["always_mask"],
                )
            ms = _bench(fn, args.warmup, args.iters)
            ablations.append({"name": name, **conf, "ms": round(ms, 4), "speedup": round(base_ms / ms, 3)})
            print(f"[ablation] {name}: {ms:.2f} ms ({base_ms/ms:.3f}x)")
        except Exception as exc:
            ablations.append({"name": name, "error": str(exc)})

    rows_ok = [r for r in rows if r.get("ok")]
    rows_ok.sort(key=lambda r: r["ms"])

    out = {
        "device": torch.cuda.get_device_name(0),
        "seq_len": L,
        "heads": H,
        "head_dim": D,
        "sparsity": args.sparsity,
        "real_topk": int(real_topk),
        "k_blocks": int(k_blocks),
        "baseline": {
            "ms": round(base_ms, 4),
            "tflops": round(base_tflops, 2),
            "e2e_ms_map_layout_kernel": round(e2e_base_ms, 4),
            "config": {"block_m": 64, "block_n": 64, "num_warps": 8, "num_stages": 3},
        },
        "best": best,
        "best_e2e_ms_map_layout_kernel": None if e2e_best_ms is None else round(e2e_best_ms, 4),
        "best_kernel_only_ms": None if cached_ms is None else round(cached_ms, 4),
        "max_speedup_kernel": None if best is None else best["speedup_vs_baseline_kernel"],
        "max_tflops": None if best is None else best["tflops"],
        "ablations": ablations,
        "top10": rows_ok[:10],
        "all_rows": rows,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(f"baseline: {base_ms:.2f} ms, {base_tflops:.1f} TFLOPS")
    if best:
        print(
            f"best: {best['ms']:.2f} ms, {best['tflops']:.1f} TFLOPS, "
            f"{best['speedup_vs_baseline_kernel']:.3f}x "
            f"(BM={best['block_m']} BN={best['block_n']} warps={best['num_warps']} "
            f"stages={best['num_stages']} loop={best['loop_stages']} mask={best['always_mask']})"
        )
        if e2e_best_ms is not None:
            print(f"e2e baseline→best: {e2e_base_ms:.2f} → {e2e_best_ms:.2f} ms ({e2e_base_ms/e2e_best_ms:.3f}x)")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
