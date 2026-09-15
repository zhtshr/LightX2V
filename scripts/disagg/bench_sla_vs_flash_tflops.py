#!/usr/bin/env python3
"""Microbench: Flash vs SLA achieved TFLOPS at different sparsity.

Reports theoretical attn FLOPs (4*Q*Ksel*H*D) and wall TFLOPS = FLOPs / time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lightx2v.common.ops.attn.flash_attn import FlashAttn2Weight
from lightx2v.common.ops.attn.sla_attn import SlaAttnWeight
from lightx2v.common.ops.attn.utils.sla_util import get_block_map


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
    return start.elapsed_time(end) / iters  # ms


def _attn_flops(q_len: int, k_len: int, heads: int, head_dim: int) -> float:
    # Forward QK^T + AV, no softmax counted (same convention as Flash TFLOPS)
    return 4.0 * q_len * k_len * heads * head_dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=32130, help="480p MoE img tokens (~SP2 global)")
    parser.add_argument("--heads", type=int, default=20, help="heads after Ulysses SP2 shard (40/2)")
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--sparsities", default="0.0,0.5,0.8")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument(
        "--output_json",
        default="save_results/optimization_study/sla_vs_flash_tflops.json",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    L, H, D = args.seq_len, args.heads, args.head_dim
    torch.manual_seed(0)

    q = torch.randn(L, H, D, device=device, dtype=dtype)
    k = torch.randn(L, H, D, device=device, dtype=dtype)
    v = torch.randn(L, H, D, device=device, dtype=dtype)

    flash = FlashAttn2Weight()
    dense_flops = _attn_flops(L, L, H, D)
    flash_ms = _bench(
        lambda: flash.apply(q, k, v, max_seqlen_q=L, max_seqlen_kv=L),
        args.warmup,
        args.iters,
    )
    flash_tflops = dense_flops / (flash_ms * 1e-3) / 1e12

    results: dict = {
        "device": torch.cuda.get_device_name(0),
        "seq_len": L,
        "heads": H,
        "head_dim": D,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "flash_attn2": {
            "ms": round(flash_ms, 4),
            "theoretical_flops": dense_flops,
            "theoretical_tflop": round(dense_flops / 1e12, 4),
            "achieved_tflops": round(flash_tflops, 2),
            "k_selected_ratio": 1.0,
        },
        "sla_triton": {},
    }

    sparsities = [float(x) for x in args.sparsities.split(",") if x.strip()]
    for sp in sparsities:
        sla = SlaAttnWeight()
        sla.sparsity_ratio = sp
        sla.topk = 1.0 - sp
        sla.operator = "triton"
        sla.BLKQ, sla.BLKK = 64, 64
        sla.apply_func = sla.apply_triton

        # Measure actual selected token-equivalent: topk * K_blocks * BLKK / K_len
        qt = q.unsqueeze(0).transpose(1, 2).contiguous()
        kt = k.unsqueeze(0).transpose(1, 2).contiguous()
        sparse_map, lut, real_topk = get_block_map(qt, kt, topk_ratio=sla.topk, BLKQ=64, BLKK=64)
        k_blocks = sparse_map.shape[-1]
        # Per Q-block selected K blocks = real_topk; FLOPs ~ 4 * Q * (real_topk*BLKK_eff) * H * D
        # Use exact: each Q token attends real_topk blocks of up to BLKK tokens (edge block shorter)
        # Approximate selected K length = real_topk / k_blocks * L
        k_sel_ratio = real_topk / max(k_blocks, 1)
        k_sel = k_sel_ratio * L
        sla_flops = _attn_flops(L, k_sel, H, D)

        ms = _bench(
            lambda: sla.apply(q, k, v, max_seqlen_q=L, max_seqlen_kv=L),
            args.warmup,
            args.iters,
        )
        # Also count block-map time separately once
        map_ms = _bench(
            lambda: get_block_map(qt, kt, topk_ratio=sla.topk, BLKQ=64, BLKK=64),
            max(2, args.warmup // 2),
            max(5, args.iters // 2),
        )
        tflops = sla_flops / (ms * 1e-3) / 1e12
        tflops_vs_dense_denom = dense_flops / (ms * 1e-3) / 1e12  # if someone naively uses dense FLOPs

        results["sla_triton"][str(sp)] = {
            "sparsity_ratio": sp,
            "topk_ratio": sla.topk,
            "real_topk_blocks": int(real_topk),
            "k_blocks": int(k_blocks),
            "k_selected_ratio": round(k_sel_ratio, 4),
            "ms_total": round(ms, 4),
            "ms_block_map_only": round(map_ms, 4),
            "theoretical_flops_sparse": sla_flops,
            "theoretical_tflop_sparse": round(sla_flops / 1e12, 4),
            "achieved_tflops_vs_sparse_flops": round(tflops, 2),
            "naive_tflops_vs_dense_flops": round(tflops_vs_dense_denom, 2),
            "speedup_vs_flash": round(flash_ms / ms, 3),
            "flops_ratio_vs_dense": round(k_sel_ratio, 4),
            "tflops_ratio_vs_flash": round(tflops / flash_tflops, 3) if flash_tflops > 0 else None,
        }
        print(
            f"sla sp={sp:.2f}: {ms:.2f} ms, k_sel={k_sel_ratio:.3f}, "
            f"TFLOPS(sparse-denom)={tflops:.1f}, TFLOPS(dense-denom)={tflops_vs_dense_denom:.1f}, "
            f"vs flash {flash_ms/ms:.2f}x"
        )

    print(
        f"flash: {flash_ms:.2f} ms, TFLOPS={flash_tflops:.1f}, "
        f"dense_tflop={dense_flops/1e12:.2f}"
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
