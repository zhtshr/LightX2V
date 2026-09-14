import os
from datetime import timedelta

import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

try:
    from torch.distributed import ProcessGroupNCCL
except ImportError:
    ProcessGroupNCCL = None


@PLATFORM_DEVICE_REGISTER("cuda")
class CudaDevice:
    name = "cuda"

    @staticmethod
    def init_device_env():
        pass

    @staticmethod
    def is_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        return "cuda"

    @staticmethod
    def init_parallel_env():
        if ProcessGroupNCCL is None:
            raise RuntimeError("ProcessGroupNCCL is not available. Please check your runtime environment.")
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
        timeout_s = int(os.getenv("LIGHTX2V_NCCL_TIMEOUT_S", "600"))
        dist.init_process_group(
            backend="nccl",
            pg_options=pg_options,
            timeout=timedelta(seconds=timeout_s),
        )
        torch.cuda.set_device(dist.get_rank())
