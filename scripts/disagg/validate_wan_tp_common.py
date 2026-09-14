"""Shared helpers for Wan tensor-parallel phase validation."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor.device_mesh import init_device_mesh

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if "lightx2v" not in sys.modules:
    _pkg = types.ModuleType("lightx2v")
    _pkg.__path__ = [str(_ROOT / "lightx2v")]
    sys.modules["lightx2v"] = _pkg

import lightx2v_platform.set_ai_device  # noqa: F401

from lightx2v.models.networks.wan.tp_utils import validate_wan_tp_config
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

_WAN_T2V_DEFAULTS = {
    "dim": 1536,
    "num_heads": 12,
    "ffn_dim": 8960,
    "num_layers": 30,
    "model_cls": "wan2.1",
    "task": "t2v",
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
}


def _load_config_dict(
    *,
    config_json: str,
    model_path: str,
    task: str,
    model_cls: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_path": model_path,
        "task": task,
        "model_cls": model_cls,
        "cpu_offload": False,
    }
    if model_cls.startswith("wan2.1"):
        config.update(_WAN_T2V_DEFAULTS)
    cfg_path = Path(config_json)
    if cfg_path.is_file():
        config.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    model_cfg = Path(model_path) / "config.json"
    if model_cfg.is_file():
        config.update(json.loads(model_cfg.read_text(encoding="utf-8")))
    return config


def set_parallel_config(config: dict[str, Any]) -> None:
    if not config.get("parallel"):
        return
    tensor_p_size = int(config["parallel"].get("tensor_p_size", 1))
    if tensor_p_size > 1:
        assert tensor_p_size == dist.get_world_size(), (
            f"tensor_p_size ({tensor_p_size}) must equal world_size ({dist.get_world_size()})"
        )
        validate_wan_tp_config(config)
        config["device_mesh"] = init_device_mesh(AI_DEVICE, (tensor_p_size,), mesh_dim_names=("tensor_p",))
        config["tensor_parallel"] = True
        config["seq_parallel"] = False
        config["cfg_parallel"] = False
        config["load_from_rank0"] = True
    else:
        cfg_p_size = int(config["parallel"].get("cfg_p_size", 1))
        seq_p_size = int(config["parallel"].get("seq_p_size", 1))
        assert cfg_p_size * seq_p_size == dist.get_world_size()
        config["device_mesh"] = init_device_mesh(AI_DEVICE, (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))
        config["tensor_parallel"] = False
        config["seq_parallel"] = seq_p_size > 1
        config["cfg_parallel"] = bool(config.get("enable_cfg")) and cfg_p_size > 1
    warmup = torch.zeros([1], device=f"{AI_DEVICE}:{dist.get_rank()}")
    dist.all_reduce(warmup)


def init_tp_config(
    *,
    config_json: str,
    model_path: str,
    task: str = "t2v",
    model_cls: str = "wan2.1",
    tensor_p_size: int = 2,
) -> dict[str, Any]:
    config = _load_config_dict(
        config_json=config_json,
        model_path=model_path,
        task=task,
        model_cls=model_cls,
    )
    cfg_parallel = config.get("parallel") or {}
    tp_size = tensor_p_size or int(cfg_parallel.get("tensor_p_size", 1))
    if tp_size > 1:
        config["parallel"] = {"tensor_p_size": tp_size}
        config["enable_cfg"] = False
        config["cpu_offload"] = False
    else:
        config["parallel"] = False
    return config


def init_distributed(config: dict[str, Any]) -> None:
    if not config.get("parallel"):
        return
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def local_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()
