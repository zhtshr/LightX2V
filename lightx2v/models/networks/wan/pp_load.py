"""Distribute Wan checkpoint shards for layer-wise pipeline parallel."""

from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.wan.pp_utils import weight_key_for_pp_rank


def distribute_weights_pp_from_rank0(
    weight_dict: dict | None,
    is_weight_loader: bool,
    pp_group,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int | None = None,
) -> dict:
    global_src = dist.get_global_rank(pp_group, 0)

    if is_weight_loader:
        assert weight_dict is not None
        meta: dict[str, dict] = {}
        for key, tensor in weight_dict.items():
            owners = [
                r
                for r in range(pp_size)
                if weight_key_for_pp_rank(key, r, pp_size, num_layers, layers_per_stage)
            ]
            if not owners:
                continue
            meta[key] = {
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "owners": owners,
            }
        obj = [meta]
        dist.broadcast_object_list(obj, src=global_src)
        synced_meta = obj[0]
    else:
        obj = [None]
        dist.broadcast_object_list(obj, src=global_src)
        synced_meta = obj[0]

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    distributed: dict = {}

    dist.barrier(group=pp_group)
    for key in sorted(synced_meta.keys()):
        meta = synced_meta[key]
        owners = meta["owners"]
        buf = torch.empty(meta["shape"], dtype=meta["dtype"], device=device)
        if is_weight_loader and pp_rank in owners:
            src_tensor = weight_dict[key]
            if src_tensor.device.type != device.type:
                src_tensor = src_tensor.to(device, non_blocking=True)
            buf.copy_(src_tensor)
            del weight_dict[key]
        dist.broadcast(buf, src=global_src, group=pp_group)
        if pp_rank in owners:
            distributed[key] = buf

    torch.cuda.synchronize()
    logger.info(
        f"Wan PP weights distributed (pp_size={pp_size}, pp_rank={pp_rank}, "
        f"keys={len(distributed)})"
    )
    return distributed
