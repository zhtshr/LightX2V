#!/usr/bin/env python3
"""Debug PP=2 model load only."""
import gc
import os

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import set_config
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.utils.set_config import set_parallel_config
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def mem():
    if torch.cuda.is_available():
        r = dist.get_rank() if dist.is_initialized() else 0
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated(r) / 1e9
        rsv = torch.cuda.memory_reserved(r) / 1e9
        print(f"[rank{r}] alloc={a:.2f}GB reserved={rsv:.2f}GB", flush=True)


def main():
    config = set_config(
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v",
        model_cls="wan2.2_moe",
        config_path="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json",
    )
    config["parallel"] = {"pipe_p_size": 2}
    PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda")).init_parallel_env()
    set_parallel_config(config)
    rank = dist.get_rank()
    high_path = config["high_noise_quantized_ckpt"]
    low_path = config["low_noise_quantized_ckpt"]
    device = torch.device(f"cuda:{rank}")

    print(f"[rank{rank}] loading high", flush=True)
    mem()
    high = WanModel(high_path, config, device, model_type="wan2.2_moe_high_noise")
    print(f"[rank{rank}] high done", flush=True)
    mem()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[rank{rank}] loading low", flush=True)
    mem()
    low = WanModel(low_path, config, device, model_type="wan2.2_moe_low_noise")
    print(f"[rank{rank}] low done", flush=True)
    mem()
    dist.barrier()
    if rank == 0:
        print("LOAD OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
