import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
assert world == 2
torch.cuda.set_device(rank)
peer_dev = 1 - rank

t = torch.zeros(4, device=f"cuda:{rank}")
h = t.untyped_storage()._share_cuda_()
handles = [None, None]
dist.all_gather_object(handles, h)
peer_h = handles[1 - rank]
(
    _storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
) = peer_h
peer_storage = torch.UntypedStorage._new_shared_cuda(
    peer_dev,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
)
peer = torch.tensor([], dtype=torch.float32, device=f"cuda:{peer_dev}").set_(peer_storage, 0, (4,))
t.fill_(float(rank + 1))
torch.cuda.synchronize()
gloo = dist.new_group(ranks=[0, 1], backend="gloo")
dist.barrier(group=gloo)
staging = torch.empty(4, dtype=torch.float32, device=f"cuda:{rank}")
staging.copy_(peer, non_blocking=False)
out = t + staging
if rank == 0:
    print("sum", out.cpu().tolist())
dist.destroy_process_group()
