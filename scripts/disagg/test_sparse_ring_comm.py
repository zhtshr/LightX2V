#!/usr/bin/env python3
import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.sparse_ring_comm import (
    SparseRingComm,
    block_map_from_pooled,
    k_block_means,
    needed_k_block_ids,
)
from lightx2v.common.ops.attn.utils.sla_util import mean_pool


def main() -> None:
    dist.init_process_group("nccl")
    r = dist.get_rank()
    torch.cuda.set_device(r)
    dev = torch.device("cuda", r)
    shard, heads, dim = 512, 8, 64
    k = torch.randn(1, shard, heads, dim, device=dev, dtype=torch.float16)
    v = k.clone()
    q = torch.randn(1, shard, heads, dim, device=dev, dtype=torch.float16)
    kb = (shard + 63) // 64
    comm = SparseRingComm(dist.group.WORLD, max_k_blocks=kb, blkk=64, heads=heads, hidden=dim, dtype=torch.float16)

    if r == 0:
        print("kmeans", flush=True)
    km = k_block_means(k, 64)
    recv = comm.exchange_kmeans(km)
    dist.barrier()
    if r == 0:
        print("kmeans ok", flush=True)

    qm = mean_pool(q.transpose(1, 2).contiguous(), 64)
    sm = block_map_from_pooled(qm, recv, 0.2)
    need = needed_k_block_ids(sm).clamp(0, kb - 1)
    world = dist.get_world_size()
    if r == 0:
        print("need exchange", flush=True)
    send_need = torch.full((kb + 2,), -1, device=dev, dtype=torch.int32)
    send_need[0] = need.numel()
    if need.numel():
        send_need[1 : 1 + need.numel()] = need
    gathered = [torch.empty_like(send_need) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, send_need)
    recv_need = gathered[(r + 1) % dist.get_world_size()]
    dist.barrier()
    if r == 0:
        print("need ok", flush=True)

    nright = int(recv_need[0].item())
    sids = recv_need[1 : 1 + nright].clone() if nright > 0 else need.new_zeros(0)
    if r == 0:
        print("sparse kv", flush=True)
    kr, vr, _ = comm.exchange_sparse_kv(k, v, sids, need)
    dist.barrier()
    if r == 0:
        print("done", tuple(kr.shape), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
