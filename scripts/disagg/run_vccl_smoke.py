#!/usr/bin/env python3
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
n = 13107200
t = torch.zeros(n, device="cuda", dtype=torch.float16)
if rank == 0:
    t.fill_(1.0)
dist.broadcast(t, src=0)
dist.barrier()
if rank == 0:
    print("broadcast ok", float(t[0].item()), n)

