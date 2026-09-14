#!/usr/bin/env python3
"""Flash-only microbench: Stripe shape vs Ulysses shape (same FLOPs).

Stripe:  Q=[B,Qlen,H,D], K=[B,S/P,H,D]
Ulysses: Q=[B,Qlen,H/P,D], K=[B,S,H/P,D]

Per-rank FLOPs (Flash attn QK+PV): 4*Q*(S/P)*H*D = 4*Q*S*(H/P)*D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    import flash_attn
except ImportError as e:
    raise SystemExit(f"flash_attn required: {e}") from e


def _flash(q, k, v, scale: float):
    return flash_attn.flash_attn_interface._flash_attn_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )


def _bench(q, k, v, scale: float, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _flash(q, k, v, scale)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _flash(q, k, v, scale)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--q_len", type=int, default=4680)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--sp", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--kv_lens",
        default="4680,9360,14040,18720,23400,28080,32760",
        help="Global KV lengths (Ulysses K length; stripe uses kv/sp)",
    )
    parser.add_argument(
        "--output_json",
        default="save_results/optimization_study/sf_flash_stripe_vs_ulysses_p2.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    H = args.num_heads
    D = args.head_dim
    Q = args.q_len
    P = args.sp
    scale = D**-0.5
    kv_lens = [int(x) for x in args.kv_lens.split(",") if x.strip()]

    rows = []
    for S in kv_lens:
        assert S % P == 0, f"kv_len {S} must be divisible by sp={P}"
        Hs = H // P
        Ss = S // P

        # Stripe: full heads, sharded K
        q_s = torch.randn(1, Q, H, D, device=device, dtype=dtype)
        k_s = torch.randn(1, Ss, H, D, device=device, dtype=dtype)
        v_s = torch.randn_like(k_s)
        ms_stripe = _bench(q_s, k_s, v_s, scale, args.warmup, args.iters)

        # Ulysses: sharded heads, full K
        q_u = torch.randn(1, Q, Hs, D, device=device, dtype=dtype)
        k_u = torch.randn(1, S, Hs, D, device=device, dtype=dtype)
        v_u = torch.randn_like(k_u)
        ms_ulysses = _bench(q_u, k_u, v_u, scale, args.warmup, args.iters)

        # Actual per-rank work (equal for both shapes).
        flops = 4.0 * Q * Ss * H * D
        peak_tflops = 125.0  # A10 BF16 Tensor Core dense peak
        tflops_s = flops / (ms_stripe * 1e-3) / 1e12
        tflops_u = flops / (ms_ulysses * 1e-3) / 1e12
        rows.append(
            {
                "global_kv_len": S,
                "stripe_local_kv_len": Ss,
                "ulysses_heads": Hs,
                "stripe": {
                    "shape": f"Q=[{Q},{H},{D}] K=[{Ss},{H},{D}]",
                    "flash_ms": round(ms_stripe, 4),
                    "tflops": round(tflops_s, 2),
                    "util_pct_vs_125t": round(100.0 * tflops_s / peak_tflops, 1),
                },
                "ulysses": {
                    "shape": f"Q=[{Q},{Hs},{D}] K=[{S},{Hs},{D}]",
                    "flash_ms": round(ms_ulysses, 4),
                    "tflops": round(tflops_u, 2),
                    "util_pct_vs_125t": round(100.0 * tflops_u / peak_tflops, 1),
                },
                "stripe_over_ulysses": round(ms_stripe / ms_ulysses, 3),
                "stripe_faster": ms_stripe < ms_ulysses,
            }
        )
        print(
            f"S={S:5d} localK={Ss:5d}  "
            f"stripe {ms_stripe:7.3f}ms ({tflops_s:5.1f}T {100*tflops_s/peak_tflops:4.1f}%)  "
            f"ulysses {ms_ulysses:7.3f}ms ({tflops_u:5.1f}T {100*tflops_u/peak_tflops:4.1f}%)  "
            f"ratio={ms_stripe/ms_ulysses:.3f}"
        )

    mean_ratio = sum(r["stripe_over_ulysses"] for r in rows) / len(rows)
    result = {
        "metric": "flash_only_stripe_vs_ulysses",
        "sp": P,
        "q_len": Q,
        "num_heads": H,
        "head_dim": D,
        "dtype": args.dtype,
        "device": str(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_stripe_over_ulysses": round(mean_ratio, 3),
        "conclusion": (
            "stripe Flash faster" if mean_ratio < 0.98 else
            "ulysses Flash faster" if mean_ratio > 1.02 else
            "Flash compute roughly equal"
        ),
        "rows": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mean_stripe_over_ulysses", "conclusion")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
