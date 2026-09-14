"""Layer-wise pipeline parallel helpers for Wan DiT."""

from __future__ import annotations


def pp_layers_per_stage(config: dict) -> int | None:
    lps = config.get("pp_layers_per_stage")
    if lps is None:
        return None
    return int(lps)


def pp_num_stages(num_layers: int, pp_size: int, layers_per_stage: int | None) -> int:
    if layers_per_stage is None:
        return pp_size
    if num_layers % layers_per_stage != 0:
        raise ValueError(f"num_layers {num_layers} not divisible by pp_layers_per_stage {layers_per_stage}")
    return num_layers // layers_per_stage


def pp_stage_owner(stage: int, pp_size: int) -> int:
    return stage % pp_size


def pp_stage_layer_range(stage: int, layers_per_stage: int) -> tuple[int, int]:
    start = stage * layers_per_stage
    return start, start + layers_per_stage


def pp_owned_block_indices(
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int | None = None,
) -> list[int]:
    if layers_per_stage is None:
        start, end = pp_layer_range(pp_rank, pp_size, num_layers)
        return list(range(start, end))
    num_stages = pp_num_stages(num_layers, pp_size, layers_per_stage)
    indices: list[int] = []
    for stage in range(num_stages):
        if pp_stage_owner(stage, pp_size) != pp_rank:
            continue
        lo, hi = pp_stage_layer_range(stage, layers_per_stage)
        indices.extend(range(lo, hi))
    return indices


def pp_last_stage_owner(num_layers: int, pp_size: int, layers_per_stage: int | None) -> int:
    last_stage = pp_num_stages(num_layers, pp_size, layers_per_stage) - 1
    return pp_stage_owner(last_stage, pp_size)


def pp_layer_range(pp_rank: int, pp_size: int, num_layers: int) -> tuple[int, int]:
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"pp_rank {pp_rank} out of range for pp_size {pp_size}")
    base = num_layers // pp_size
    rem = num_layers % pp_size
    start = pp_rank * base + min(pp_rank, rem)
    end = start + base + (1 if pp_rank < rem else 0)
    return start, end


def block_index_from_key(key: str) -> int | None:
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def weight_key_for_pp_rank(
    key: str,
    pp_rank: int,
    pp_size: int,
    num_layers: int,
    layers_per_stage: int | None = None,
) -> bool:
    """Return True if checkpoint key should be loaded on this pp_rank."""
    if pp_size <= 1:
        return True
    last = pp_size - 1
    blk = block_index_from_key(key)
    if blk is not None:
        if layers_per_stage is not None:
            stage = blk // layers_per_stage
            return pp_stage_owner(stage, pp_size) == pp_rank
        start, end = pp_layer_range(pp_rank, pp_size, num_layers)
        return start <= blk < end
    if pp_rank == 0:
        if key.startswith("head.") or key.startswith("head_"):
            return False
        if ".head." in key and key.split(".")[-2] == "head":
            return False
        if key.endswith("head.head.weight") or key.endswith("head.head.bias"):
            return False
        if "head.modulation" in key:
            return False
        if key == "norm.weight" or key == "norm.bias" or key.endswith(".norm.weight"):
            return False
        return True
    head_rank = pp_last_stage_owner(num_layers, pp_size, layers_per_stage)
    if pp_rank == head_rank:
        if blk is None and (
            key.startswith("head.")
            or "head.modulation" in key
            or key.endswith("head.head.weight")
            or key.endswith("head.head.bias")
            or key.startswith("norm.")
            or key.endswith(".norm.weight")
            or key.endswith(".norm.bias")
        ):
            return True
        return False
    if layers_per_stage is not None:
        return False
    start, end = pp_layer_range(pp_rank, pp_size, num_layers)
    return blk is not None and start <= blk < end


def validate_wan_pp_config(config: dict) -> None:
    if not config.get("pipeline_parallel"):
        return
    pp_size = int(config.get("pp_size", 1))
    if pp_size < 2:
        raise ValueError("pipeline_parallel requires pipe_p_size >= 2")
    if config.get("tensor_parallel"):
        raise NotImplementedError("Wan PP + TP is not supported yet")
    if config.get("cpu_offload"):
        raise NotImplementedError("Wan pipeline parallel requires cpu_offload=False")
    lps = pp_layers_per_stage(config)
    if lps is not None:
        num_layers = int(config.get("num_layers", 40))
        if num_layers % lps != 0:
            raise ValueError(f"num_layers {num_layers} must divide pp_layers_per_stage {lps}")
        if lps < 1:
            raise ValueError(f"pp_layers_per_stage must be >= 1, got {lps}")
    if config.get("seq_parallel"):
        seq_p_size = int(config.get("parallel", {}).get("seq_p_size", 1))
        if seq_p_size < 2:
            raise ValueError("Wan PP + SP requires seq_p_size >= 2 in parallel config")
