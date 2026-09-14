#!/usr/bin/env python3
"""Reproduce Wan TP weight send/recv pattern with VCCL."""
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
tp_group = dist.group.WORLD
tp_rank = rank
tp_size = 2
global_src = dist.get_global_rank(tp_group, 0)
n = 13107200
device = torch.device(f"cuda:{rank}")

dist.barrier()
buf = torch.empty(n, dtype=torch.float16, device=device)
if rank == 0:
    shard = torch.ones(n, dtype=torch.float16, device=device)
    buf.copy_(shard)
    dist.send(shard, dst=dist.get_global_rank(tp_group, 1), group=tp_group)
    print("rank0 sent")
else:
    dist.recv(buf, src=global_src, group=tp_group)
    print("rank1 recv ok", float(buf[0].item()))

dist.barrier()
if rank == 0:
    print("tp send/recv pattern ok")
