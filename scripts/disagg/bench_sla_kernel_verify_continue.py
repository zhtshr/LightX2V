#!/usr/bin/env python3
"""Continue SLA kernel optimization verification with proper per-config warmup."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import triton

# Avoid importing lightx2v.common.ops (slow / triggers qtorch JIT).
_SPEC = importlib.util.spec_from_file_location(
    "sla_kernel", Path(__file__).resolve().parents[2] / "lightx2v/common/ops/attn/kernels/sla_kernel.py"
)
_SLA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SLA)
_attn_fwd = _SLA._attn_fwd
_attention = _SLA._attention

_SPEC2 = importlib.util.spec_from_file_location(
    "sla_util", Path(__file__).resolve().parents[2] / "lightx2v/common/ops/attn/utils/sla_util.py"
)
_SLA_UTIL = importlib.util.module_from_spec(_SPEC2)
_SPEC2.loader.exec_module(_SLA_UTIL)
get_block_map = _SLA_UTIL.get_block_map

# Reuse opt kernel from sweep script.
from bench_sla_kernel_opt_sweep import sla_fwd_opt  # noqa: E402


def bench(fn, warmup: int = 20, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def prod_fwd(q, k, v, lut, topk, bm, bn, warps, stages):
    b, h, lq, d = q.shape
    lk = k.shape[2]
    mb = triton.cdiv(lq, bm)
    o = torch.empty((b, h, lq, d), device=q.device, dtype=v.dtype)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    _attn_fwd[(mb, b * h)](
        q, k, v, d**-0.5, topk, lut, lse, o, lq, lk, mb, d, bm, bn,
        num_warps=warps, num_stages=stages,
    )
    return o


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", default="save_results/optimization_study/sla_kernel_verify_continue.json")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    l, h, d = 32130, 20, 128
    torch.manual_seed(0)

    q = torch.randn(1, h, l, d, device=device, dtype=dtype).contiguous()
    k = torch.randn(1, h, l, d, device=device, dtype=dtype).contiguous()
    v = torch.randn(1, h, l, d, device=device, dtype=dtype).contiguous()
    q_nhd = torch.randn(l, h, d, device=device, dtype=dtype)
    k_nhd = torch.randn(l, h, d, device=device, dtype=dtype)
    v_nhd = torch.randn(l, h, d, device=device, dtype=dtype)

    sm, lut, topk = get_block_map(q, k, topk_ratio=0.2, BLKQ=64, BLKK=64)
    _, lut128, topk128 = get_block_map(q, k, topk_ratio=0.2, BLKQ=128, BLKK=64)

    prod_ms = bench(lambda: _attention.apply(q, k, v, sm, lut, topk, 64, 64), args.warmup, args.iters)
    ref = _attention.apply(q, k, v, sm, lut, topk, 64, 64)

    warps_rows = []
    for warps, stages in ((8, 3), (4, 2), (4, 3)):
        prod_fwd(q, k, v, lut, topk, 64, 64, warps, stages)
        out = prod_fwd(q, k, v, lut, topk, 64, 64, warps, stages)
        ms = bench(lambda w=warps, s=stages: prod_fwd(q, k, v, lut, topk, 64, 64, w, s), args.warmup, args.iters)
        err = float((out.float() - ref.float()).abs().mean())
        warps_rows.append(
            {"warps": warps, "stages": stages, "ms": ms, "speedup": prod_ms / ms, "mean_err": err}
        )
        print(f"prod warps={warps} stages={stages}: {ms:.2f} ms ({prod_ms / ms:.3f}x) err={err:.5f}")

    opt_configs = [
        ("prod_equiv", 64, 64, 8, 3, 1, False, lut, topk),
        ("warps4", 64, 64, 4, 3, 1, False, lut, topk),
        ("loop2_mask", 64, 64, 8, 3, 2, True, lut, topk),
        ("loop3_mask_s4", 64, 64, 8, 4, 3, True, lut, topk),
        ("loop3_mask_w4", 64, 64, 4, 3, 3, True, lut, topk),
        ("bm128_s3", 128, 64, 8, 3, 3, True, lut128, topk128),
    ]
    opt_rows = []
    for name, bm, bn, warps, stages, loop_s, am, lt, tk in opt_configs:
        try:
            sla_fwd_opt(q, k, v, lt, tk, block_m=bm, block_n=bn, num_warps=warps,
                        num_stages=stages, loop_stages=loop_s, always_mask=am)
            out = sla_fwd_opt(q, k, v, lt, tk, block_m=bm, block_n=bn, num_warps=warps,
                              num_stages=stages, loop_stages=loop_s, always_mask=am)
            ms = bench(
                lambda bm=bm, bn=bn, warps=warps, stages=stages, loop_s=loop_s, am=am, lt=lt, tk=tk:
                sla_fwd_opt(q, k, v, lt, tk, block_m=bm, block_n=bn, num_warps=warps,
                            num_stages=stages, loop_stages=loop_s, always_mask=am),
                args.warmup, args.iters,
            )
            err = float((out.float() - ref.float()).abs().mean())
            cos = None
            if bm == 128:
                cos = float(torch.nn.functional.cosine_similarity(
                    ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
                ).item())
            opt_rows.append({
                "name": name, "bm": bm, "bn": bn, "warps": warps, "stages": stages,
                "loop": loop_s, "mask": am, "ms": ms, "speedup": prod_ms / ms,
                "mean_err": err, "cos_vs_prod": cos,
            })
            extra = f" cos={cos:.4f}" if cos is not None else ""
            print(f"opt {name}: {ms:.2f} ms ({prod_ms / ms:.3f}x) err={err:.5f}{extra}")
        except Exception as exc:
            print(f"opt {name}: FAIL {exc}")
            opt_rows.append({"name": name, "error": str(exc)})

    pad = (64 - (l % 64)) % 64
    kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
    vp = torch.nn.functional.pad(v, (0, 0, 0, pad))
    kp_ms = bench(lambda: _attention.apply(q, kp, vp, sm, lut, topk, 64, 64), args.warmup, args.iters)
    print(f"K pad {l}->{l + pad}: {kp_ms:.2f} ms ({prod_ms / kp_ms:.3f}x)")

    layout_ms = bench(
        lambda: (
            q_nhd.unsqueeze(0).transpose(1, 2).contiguous(),
            k_nhd.unsqueeze(0).transpose(1, 2).contiguous(),
            v_nhd.unsqueeze(0).transpose(1, 2).contiguous(),
        ),
        warmup=5, iters=40,
    )
    qq = q_nhd.unsqueeze(0).transpose(1, 2).contiguous()
    kk = k_nhd.unsqueeze(0).transpose(1, 2).contiguous()
    map_ms = bench(lambda: get_block_map(qq, kk, topk_ratio=0.2, BLKQ=64, BLKK=64), warmup=5, iters=20)
    print(f"e2e layout={layout_ms:.3f} ms map={map_ms:.2f} ms kernel={prod_ms:.2f} ms")

    flash_ms = None
    try:
        from flash_attn import flash_attn_func
        qfl = q.transpose(1, 2).contiguous()
        kfl = k.transpose(1, 2).contiguous()
        vfl = v.transpose(1, 2).contiguous()
        flash_ms = bench(lambda: flash_attn_func(qfl, kfl, vfl, causal=False), warmup=8, iters=15)
        print(f"flash dense: {flash_ms:.2f} ms")
    except Exception as exc:
        print(f"flash skip: {exc}")

    best_dropin = min(warps_rows, key=lambda r: r["ms"])
    best_opt = min((r for r in opt_rows if "ms" in r), key=lambda r: r["ms"], default=None)

    out = {
        "device": torch.cuda.get_device_name(0),
        "shape": {"L": l, "H": h, "D": d, "sparsity": 0.8},
        "production_ms": prod_ms,
        "topk64": int(topk),
        "topk128": int(topk128),
        "warps_sweep": warps_rows,
        "opt_ablations": opt_rows,
        "k_pad": {"pad": int(pad), "ms": kp_ms, "speedup": prod_ms / kp_ms},
        "e2e": {"layout_ms": layout_ms, "map_ms": map_ms, "kernel_ms": prod_ms},
        "flash_dense_ms": flash_ms,
        "best_dropin": best_dropin,
        "best_opt": best_opt,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
