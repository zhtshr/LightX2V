#!/usr/bin/env python3
"""Minimal correctness check: stripe baseline vs form-C vs form-B2."""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from lightx2v.common.kvcache.base import BaseKVCachePool


class _StripePool(BaseKVCachePool):
    pass


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD

    Ql, Kl, H, D = 1170, 4680, 12, 128
    q = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    pool = _StripePool(
        num_layers=1,
        cache_size=Kl,
        num_heads=H,
        head_dim=D,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    out_base = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D)
    out_pe = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, partial_out_exchange=True)

    os.environ["LIGHTX2V_STRIPE_PACK_OUT_LSE"] = "1"
    out_pe_pack = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, partial_out_exchange=True)
    os.environ.pop("LIGHTX2V_STRIPE_PACK_OUT_LSE", None)

    out_hier = None
    if dist.get_world_size() in (4, 6):
        out_hier = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, hier_exchange=True)

    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "p2p"
    for _env in (
        "LIGHTX2V_STRIPE_B2_P2P_SERIAL",
        "LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH",
        "LIGHTX2V_STRIPE_B2_P2P_OVERLAP",
    ):
        os.environ.pop(_env, None)
    out_b2_default = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, q_exchange=True)

    os.environ["LIGHTX2V_STRIPE_B2_P2P_SERIAL"] = "1"
    out_b2_serial = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, q_exchange=True)
    os.environ.pop("LIGHTX2V_STRIPE_B2_P2P_SERIAL", None)

    os.environ["LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH"] = "1"
    out_b2_multi = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, q_exchange=True)
    os.environ.pop("LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH", None)

    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "xor_rd"
    out_b2_xor = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, q_exchange=True)
    os.environ["LIGHTX2V_STRIPE_B2_MODE"] = "ring"
    out_b2_ring = pool.sp_kvcache_attn_stripe(q, k, v, group, head_dim=D, q_exchange=True)

    if rank == 0:
        print("shapes", tuple(out_base.shape), tuple(out_pe.shape), tuple(out_b2_default.shape))
        diffs = {
            "formC": (out_base - out_pe).abs().max().item(),
            "formC_pack": (out_base - out_pe_pack).abs().max().item(),
            "B2default": (out_base - out_b2_default).abs().max().item(),
            "B2serial": (out_base - out_b2_serial).abs().max().item(),
            "B2multi": (out_base - out_b2_multi).abs().max().item(),
            "B2xor_rd": (out_base - out_b2_xor).abs().max().item(),
            "B2ring": (out_base - out_b2_ring).abs().max().item(),
        }
        if out_hier is not None:
            diffs["hier"] = (out_base - out_hier).abs().max().item()
        print("max_diff", " ".join(f"{k}={v}" for k, v in diffs.items()))
        if out_base.shape != (Ql, H * D):
            print("FAIL: bad shape", file=sys.stderr)
            return 1
        if max(diffs.values()) > 0.05:
            print("FAIL: numerical mismatch", file=sys.stderr)
            return 1
        print("OK")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
