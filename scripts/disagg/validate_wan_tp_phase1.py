#!/usr/bin/env python3
"""TP-1: weight split + FFN/head numerical alignment vs single-GPU reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.disagg.validate_wan_tp_common import init_distributed, init_tp_config, is_main, local_device, max_abs_diff, sync_cuda

from lightx2v.models.networks.wan.tp_utils import split_weight_for_tp


def _check_split_shapes(dim: int, ffn_dim: int, tp_size: int) -> None:
    w0_key = "blocks.0.ffn.0.weight"
    w2_key = "blocks.0.ffn.2.weight"
    head_key = "head.head.weight"
    w0 = torch.randn(ffn_dim, dim)
    w2 = torch.randn(dim, ffn_dim)
    head = torch.randn(dim, dim)
    w0_shards = split_weight_for_tp(w0_key, w0, tp_size)
    w2_shards = split_weight_for_tp(w2_key, w2, tp_size)
    head_shards = split_weight_for_tp(head_key, head, tp_size)
    assert len(w0_shards) == tp_size
    assert len(w2_shards) == tp_size
    assert w0_shards[0].shape == (ffn_dim // tp_size, dim)
    assert w2_shards[0].shape == (dim, ffn_dim // tp_size)
    out_dim = head.shape[0]
    assert head_shards[0].shape == (out_dim, dim // tp_size)
    if is_main():
        print(f"TP-1 split shapes OK (dim={dim}, ffn_dim={ffn_dim}, tp={tp_size})")


def _ffn_reference(x: torch.Tensor, w0, b0, w2, b2) -> torch.Tensor:
    h = F.linear(x, w0, b0)
    h = F.gelu(h, approximate="tanh")
    return F.linear(h, w2, b2)


def _distributed_ffn_tp(
    x: torch.Tensor,
    w0,
    b0,
    w2,
    b2,
    tp_group,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    device = x.device
    dtype = x.dtype
    w0_shards = split_weight_for_tp("blocks.0.ffn.0.weight", w0.cpu(), tp_size)
    w2_shards = split_weight_for_tp("blocks.0.ffn.2.weight", w2.cpu(), tp_size)
    b0_chunk = b0.shape[0] // tp_size
    weight_dict = {
        "blocks.0.ffn.0.weight": w0_shards[tp_rank].to(device=device, dtype=dtype),
        "blocks.0.ffn.0.bias": b0[tp_rank * b0_chunk : (tp_rank + 1) * b0_chunk].to(device=device, dtype=dtype),
        "blocks.0.ffn.2.weight": w2_shards[tp_rank].to(device=device, dtype=dtype),
        "blocks.0.ffn.2.bias": b2.to(device=device, dtype=dtype),
    }

    w0_local = weight_dict["blocks.0.ffn.0.weight"]
    b0_local = weight_dict["blocks.0.ffn.0.bias"]
    w2_local = weight_dict["blocks.0.ffn.2.weight"]
    b2 = weight_dict["blocks.0.ffn.2.bias"]
    h = F.linear(x, w0_local, b0_local)
    h = F.gelu(h, approximate="tanh")
    out = F.linear(h, w2_local, None)
    if tp_size > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out + b2


def _run_ffn_numerical(config: dict, atol: float) -> None:
    tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    dim = int(config["dim"])
    ffn_dim = int(config["ffn_dim"])
    device = local_device()
    dtype = torch.float32
    torch.manual_seed(1234)

    x = torch.randn(32, dim, device=device, dtype=dtype)
    w0 = torch.randn(ffn_dim, dim, device=device, dtype=dtype)
    b0 = torch.randn(ffn_dim, device=device, dtype=dtype)
    w2 = torch.randn(dim, ffn_dim, device=device, dtype=dtype)
    b2 = torch.randn(dim, device=device, dtype=dtype)

    out_ref = _ffn_reference(x, w0, b0, w2, b2)
    out_tp = _distributed_ffn_tp(x, w0, b0, w2, b2, tp_group, tp_rank, tp_size)
    diff = max_abs_diff(out_ref, out_tp)
    if is_main():
        print(f"TP-1 FFN max_abs_diff={diff:.6e} (atol={atol})")
    assert diff <= atol, f"FFN TP mismatch: {diff} > {atol}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/dist_infer/wan_t2v_tensorp.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.1-T2V-1.3B")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--model_cls", default="wan2.1")
    parser.add_argument("--tensor_p_size", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    config = init_tp_config(
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        tensor_p_size=args.tensor_p_size,
    )
    init_distributed(config)
    tp_size = dist.get_world_size()
    _check_split_shapes(int(config["dim"]), int(config["ffn_dim"]), tp_size)
    _run_ffn_numerical(config, args.atol)
    sync_cuda()
    if is_main():
        print("TP-1 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
