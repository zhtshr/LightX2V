import torch
import torch.distributed as dist

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())
t = torch.ones(1, device="cuda")
dist.all_reduce(t)
if dist.get_rank() == 0:
    print("NCCL_OK", float(t), "visible", torch.cuda.device_count())
dist.destroy_process_group()
