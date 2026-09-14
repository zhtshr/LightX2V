"""Dedicated NCCL process group for TP row all-reduce (overlap comm path)."""

from __future__ import annotations

import torch
import torch.distributed as dist

_ROW_AR_GROUPS: dict[int, dist.ProcessGroup] = {}


def get_row_ar_nccl_group(tp_group: dist.ProcessGroup) -> dist.ProcessGroup:
    """Return a separate NCCL communicator over the same TP ranks as *tp_group*."""
    key = id(tp_group)
    if key in _ROW_AR_GROUPS:
        return _ROW_AR_GROUPS[key]

    world_size = dist.get_world_size(tp_group)
    ranks = [dist.get_global_rank(tp_group, r) for r in range(world_size)]

    pg_options = None
    try:
        from torch.distributed import ProcessGroupNCCL

        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
    except ImportError:
        pass

    group = dist.new_group(ranks=ranks, backend="nccl", pg_options=pg_options)
    warmup = torch.zeros(8, device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    dist.all_reduce(warmup, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()
    _ROW_AR_GROUPS[key] = group
    return group


def resolve_row_ar_group(tp_group: dist.ProcessGroup, *, use_dedicated: bool) -> dist.ProcessGroup:
    if not use_dedicated or tp_group is None:
        return tp_group
    return get_row_ar_nccl_group(tp_group)
