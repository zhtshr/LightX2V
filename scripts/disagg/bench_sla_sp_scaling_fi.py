#!/usr/bin/env python3
"""Measure real Ulysses SP scaling for SLA attention (Triton vs FlashInfer).

Simulates one self-attention layer's SP path:
  local seq shard (L/SP, H, D) -> all2all(head<-seq) -> (L, H/SP, D)
  -> block-sparse attention (full L, H/SP heads) -> all2all(seq<-head) -> (L/SP, H, D)

Compares single-GPU full attention vs per-rank SP time to get scale efficiency.
Run with torchrun --nproc_per_node=<SP> on safe GPUs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import triton

_ROOT = Path(__file__).resolve().parents[2]
_S = importlib.util.spec_from_file_location("sla_kernel", _ROOT / "lightx2v/common/ops/attn/kernels/sla_kernel.py")
_SLA = importlib.util.module_from_spec(_S); _S.loader.exec_module(_SLA)
_attention = _SLA._attention
_S2 = importlib.util.spec_from_file_location("sla_util", _ROOT / "lightx2v/common/ops/attn/utils/sla_util.py")
_U = importlib.util.module_from_spec(_S2); _S2.loader.exec_module(_U)
get_block_map = _U.get_block_map

try:
    import flashinfer
except Exception:
    flashinfer = None


def bench(fn, warmup=10, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def make_fi(device, mb=512):
    return flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=device), backend="fa2"
    )


def plan_fi(w, mask_h, seqlen, blk, nh, hd, dtype):
    H, Mq, Mk = mask_h.shape
    row = torch.full((Mq,), blk, dtype=torch.int32, device=mask_h.device)
    row[-1] = seqlen - blk * (Mq - 1)
    col = torch.full((Mk,), blk, dtype=torch.int32, device=mask_h.device)
    col[-1] = seqlen - blk * (Mk - 1)
    w.plan(block_mask_map=mask_h.bool(),
           block_row_sz=row.unsqueeze(0).expand(H, -1).contiguous(),
           block_col_sz=col.unsqueeze(0).expand(H, -1).contiguous(),
           num_qo_heads=nh, num_kv_heads=nh, head_dim=hd, q_data_type=dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32760)
    p.add_argument("--heads", type=int, default=40)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--blk", type=int, default=64)
    p.add_argument("--backend", choices=["triton", "flashinfer"], default="flashinfer")
    p.add_argument("--output_json", default="save_results/optimization_study/sla_sp_scaling_fi.json")
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    L, H, D, blk = args.L, args.heads, args.head_dim, args.blk
    assert H % world == 0 and L % world == 0
    Hloc = H // world
    Lshard = L // world
    topk_ratio = 1.0 - args.sparsity
    torch.manual_seed(0)

    # local seq shard: (Lshard, H, D) in head-major layout for all2all
    q_shard = torch.randn(Lshard, H, D, device=device, dtype=dtype)
    k_shard = torch.randn(Lshard, H, D, device=device, dtype=dtype)
    v_shard = torch.randn(Lshard, H, D, device=device, dtype=dtype)

    def all2all_seq_to_head(x):  # (Lshard, H, D) -> (L, Hloc, D)
        # reshape to (Lshard, world, Hloc, D) -> all2all over world -> (world, Lshard, Hloc, D) -> (L, Hloc, D)
        xr = x.reshape(Lshard, world, Hloc, D).permute(1, 0, 2, 3).contiguous()
        out = torch.empty_like(xr)
        dist.all_to_all_single(out, xr)
        return out.reshape(L, Hloc, D)

    def all2all_head_to_seq(x):  # (L, Hloc, D) -> (Lshard, H, D)
        xr = x.reshape(world, Lshard, Hloc, D).contiguous()
        out = torch.empty_like(xr)
        dist.all_to_all_single(out, xr)
        return out.permute(1, 0, 2, 3).reshape(Lshard, H, D)

    # Build block map once on full-L, Hloc (post-a2a shapes)
    q_full = all2all_seq_to_head(q_shard)  # (L, Hloc, D)
    k_full = all2all_seq_to_head(k_shard)
    v_full = all2all_seq_to_head(v_shard)
    q_bhld = q_full.unsqueeze(0).transpose(1, 2).contiguous()  # (1,Hloc,L,D)
    k_bhld = k_full.unsqueeze(0).transpose(1, 2).contiguous()
    v_bhld = v_full.unsqueeze(0).transpose(1, 2).contiguous()
    sm, lut, topk = get_block_map(q_bhld, k_bhld, topk_ratio=topk_ratio, BLKQ=blk, BLKK=blk)
    k_blocks = sm.shape[-1]

    fi_w = None
    if args.backend == "flashinfer":
        fi_w = make_fi(device)
        mh = sm.squeeze(0).bool().contiguous()
        plan_fi(fi_w, mh, L, blk, Hloc, D, dtype)
        qf = q_bhld.squeeze(0).contiguous()
        kf = k_bhld.squeeze(0).contiguous()
        vf = v_bhld.squeeze(0).contiguous()

    def attn_only():
        if args.backend == "flashinfer":
            return fi_w.run(qf, kf, vf)
        return _attention.apply(q_bhld, k_bhld, v_bhld, sm, lut, topk, blk, blk)

    def full_sp_step():
        qh = all2all_seq_to_head(q_shard)
        kh = all2all_seq_to_head(k_shard)
        vh = all2all_seq_to_head(v_shard)
        qb = qh.unsqueeze(0).transpose(1, 2).contiguous()
        kb = kh.unsqueeze(0).transpose(1, 2).contiguous()
        vb = vh.unsqueeze(0).transpose(1, 2).contiguous()
        if args.backend == "flashinfer":
            o = fi_w.run(qb.squeeze(0).contiguous(), kb.squeeze(0).contiguous(), vb.squeeze(0).contiguous())
            o = o.unsqueeze(0)
        else:
            o = _attention.apply(qb, kb, vb, sm, lut, topk, blk, blk)
        o_hld = o.squeeze(0).transpose(0, 1).contiguous()  # (L, Hloc, D)
        return all2all_head_to_seq(o_hld)

    dist.barrier()
    attn_ms = bench(attn_only)
    dist.barrier()
    step_ms = bench(full_sp_step)
    comm_ms = step_ms - attn_ms

    # gather to rank0
    stats = torch.tensor([attn_ms, step_ms, comm_ms], device=device)
    gathered = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats)

    if rank == 0:
        rows = [g.tolist() for g in gathered]
        avg_attn = sum(r[0] for r in rows) / world
        avg_step = sum(r[1] for r in rows) / world
        avg_comm = sum(r[2] for r in rows) / world
        out = {
            "backend": args.backend,
            "L": L, "heads": H, "head_dim": D, "sparsity": args.sparsity, "blk": blk,
            "world_size": world, "Hloc": Hloc, "Lshard": Lshard,
            "topk": int(topk), "k_blocks": int(k_blocks),
            "avg_attn_ms": avg_attn, "avg_step_ms": avg_step, "avg_comm_ms": avg_comm,
            "per_rank": rows,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        # append-merge by (backend, world)
        agg_path = Path(args.output_json)
        existing = {}
        if agg_path.exists():
            try:
                existing = json.loads(agg_path.read_text())
            except Exception:
                existing = {}
        existing[f"{args.backend}_sp{world}"] = out
        agg_path.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"[{args.backend} SP{world}] attn={avg_attn:.2f}ms comm={avg_comm:.2f}ms step={avg_step:.2f}ms "
              f"(topk={topk}/{k_blocks})")

    dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
