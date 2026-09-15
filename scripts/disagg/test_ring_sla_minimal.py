#!/usr/bin/env python3
"""Minimal ring_sla smoke test (4 ranks)."""

import os
import sys

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.ring_sla_attn import RingSlaAttnWeight
from lightx2v.common.ops.attn.sla_attn import SlaAttnWeight


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    world = dist.get_world_size()
    shard = 512
    heads, dim = 8, 64
    L = shard * world

    q = torch.randn(shard, heads, dim, device=device, dtype=torch.float16)
    k = torch.randn(shard, heads, dim, device=device, dtype=torch.float16)
    v = torch.randn(shard, heads, dim, device=device, dtype=torch.float16)

    sla = SlaAttnWeight()
    for sparse in (False, True):
        mod = RingSlaAttnWeight()
        mod.sparse_comm = sparse
        mod.sparsity_ratio = 0.8
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = mod.apply(
            q, k, v,
            slice_qkv_len=shard,
            cu_seqlens_qkv=torch.tensor([0, shard], dtype=torch.int32, device=device),
            attention_module=sla,
            seq_p_group=dist.group.WORLD,
        )
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        if rank == 0:
            print(f"sparse={sparse} out={tuple(out.shape)} ms={ms:.2f}", flush=True)
        dist.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
