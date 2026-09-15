#!/usr/bin/env python3
"""TP-0: parallel mesh init + Wan TP config validation smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.distributed as dist

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.disagg.validate_wan_tp_common import init_distributed, init_tp_config, is_main, sync_cuda


def _assert_tp_mesh(config: dict) -> None:
    assert config.get("tensor_parallel"), "tensor_parallel should be True"
    assert config.get("load_from_rank0"), "load_from_rank0 should be True"
    assert not config.get("seq_parallel"), "seq_parallel must be False for pure TP"
    mesh = config["device_mesh"]
    tp_group = mesh.get_group(mesh_dim="tensor_p")
    tp_size = dist.get_world_size(tp_group)
    tp_rank = dist.get_rank(tp_group)
    expected = int(config["parallel"]["tensor_p_size"])
    assert tp_size == expected, f"tp_size={tp_size} != expected {expected}"
    assert 0 <= tp_rank < tp_size
    if is_main():
        print(f"TP-0 OK: mesh tensor_p size={tp_size}, rank={tp_rank}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/dist_infer/wan_t2v_tensorp.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.1-T2V-1.3B")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--model_cls", default="wan2.1")
    parser.add_argument("--tensor_p_size", type=int, default=0)
    args = parser.parse_args()

    config = init_tp_config(
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        tensor_p_size=args.tensor_p_size,
    )
    init_distributed(config)
    _assert_tp_mesh(config)
    sync_cuda()
    if is_main():
        print("TP-0 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
