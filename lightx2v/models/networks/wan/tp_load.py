"""Distribute Wan checkpoint shards for tensor parallel (rank-0 loader)."""

from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.wan.tp_utils import get_split_type, is_tp_weight, split_weight_for_tp


def distribute_weights_from_rank0(weight_dict, is_weight_loader: bool, tp_group, tp_rank: int, tp_size: int) -> dict:
    global_src_rank = dist.get_global_rank(tp_group, 0)

    if is_weight_loader:
        processed: dict = {}
        meta_dict: dict = {}
        handled_bias: set[str] = set()

        for key, tensor in weight_dict.items():
            if key.endswith(".weight") and is_tp_weight(key):
                split_weights = split_weight_for_tp(key, tensor, tp_size)
                for rank_idx in range(tp_size):
                    processed[f"{key}__tp_rank_{rank_idx}"] = split_weights[rank_idx]
                meta_dict[key] = {
                    "shape": split_weights[0].shape,
                    "dtype": split_weights[0].dtype,
                    "is_tp": True,
                }

                bias_key = key.replace(".weight", ".bias")
                if bias_key in weight_dict and get_split_type(key) == "col":
                    bias_tensor = weight_dict[bias_key]
                    assert bias_tensor.shape[0] % tp_size == 0, (
                        f"bias ({bias_tensor.shape[0]}) must be divisible by tp_size ({tp_size}) for {bias_key}"
                    )
                    chunk = bias_tensor.shape[0] // tp_size
                    for rank_idx in range(tp_size):
                        processed[f"{bias_key}__tp_rank_{rank_idx}"] = bias_tensor[
                            rank_idx * chunk : (rank_idx + 1) * chunk
                        ]
                    meta_dict[bias_key] = {
                        "shape": bias_tensor[0:chunk].shape,
                        "dtype": bias_tensor.dtype,
                        "is_tp": True,
                    }
                    handled_bias.add(bias_key)
            elif key.endswith(".bias") and key in handled_bias:
                continue
            else:
                processed[key] = tensor
                meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype, "is_tp": False}

        obj_list = [meta_dict]
        dist.broadcast_object_list(obj_list, src=global_src_rank)
        synced_meta = obj_list[0]
        weight_dict = processed
    else:
        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=global_src_rank)
        synced_meta = obj_list[0]

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    distributed: dict = {}
    for key, meta in synced_meta.items():
        distributed[key] = torch.empty(meta["shape"], dtype=meta["dtype"], device=device)

    dist.barrier()

    for key in sorted(synced_meta.keys()):
        meta = synced_meta[key]
        if meta.get("is_tp", False):
            dist.barrier(group=tp_group)
            for rank_idx in range(tp_size):
                if is_weight_loader:
                    rank_key = f"{key}__tp_rank_{rank_idx}"
                    shard = weight_dict[rank_key].contiguous()
                    if shard.device.type != device.type:
                        shard = shard.to(device, non_blocking=True)
                else:
                    shard = distributed[key] if rank_idx == tp_rank else torch.empty_like(distributed[key])
                dist.broadcast(shard, src=global_src_rank, group=tp_group)
                if rank_idx == tp_rank:
                    distributed[key].copy_(shard, non_blocking=True)
        else:
            if is_weight_loader:
                tensor = weight_dict[key]
                if tensor.device.type != device.type:
                    tensor = tensor.to(device, non_blocking=True)
                distributed[key].copy_(tensor, non_blocking=True)
            dist.broadcast(distributed[key], src=global_src_rank, group=tp_group)

    torch.cuda.synchronize()
    logger.info(f"Wan TP weights distributed (tp_size={tp_size}, tp_rank={tp_rank})")
    return distributed
