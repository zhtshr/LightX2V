"""Tensor-parallel helpers for Wan DiT (Megatron-style col/row split)."""

from __future__ import annotations

from typing import Literal

SplitType = Literal["col", "row", "norm"]


def validate_wan_tp_config(config: dict) -> None:
    parallel = config.get("parallel") or {}
    tp_size = int(parallel.get("tensor_p_size", 1))
    if tp_size <= 1:
        return
    if config.get("seq_parallel"):
        raise ValueError("Wan pure TP does not support seq_parallel; set seq_p_size=1")
    if config.get("cpu_offload"):
        raise NotImplementedError("Wan tensor parallel requires cpu_offload=False")

    num_heads = int(config["num_heads"])
    dim = int(config["dim"])
    ffn_dim = int(config["ffn_dim"])
    for name, value in (("num_heads", num_heads), ("dim", dim), ("ffn_dim", ffn_dim)):
        if value % tp_size != 0:
            raise ValueError(f"{name}={value} must be divisible by tensor_p_size={tp_size}")


def is_tp_weight(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    split = get_split_type(key)
    return split is not None


def get_split_type(key: str) -> SplitType | None:
    if any(s in key for s in (".self_attn.norm_q.", ".self_attn.norm_k.", ".cross_attn.norm_q.")):
        return "norm"
    if any(
        s in key
        for s in (
            ".self_attn.q.",
            ".self_attn.k.",
            ".self_attn.v.",
            ".cross_attn.q.",
            ".ffn.0.",
        )
    ):
        return "col"
    if any(s in key for s in (".self_attn.o.", ".cross_attn.o.", ".ffn.2.")):
        return "row"
    return None


def split_weight_for_tp(key: str, weight, tp_size: int):
    split_type = get_split_type(key)
    if split_type is None:
        return [weight] * tp_size

    if split_type == "norm":
        assert weight.dim() == 1, f"Norm weight should be 1D, got {weight.dim()}D for {key}"
        assert weight.shape[0] % tp_size == 0, f"hidden_dim ({weight.shape[0]}) must be divisible by tp_size ({tp_size}) for {key}"
        chunk = weight.shape[0] // tp_size
        return [weight[i * chunk : (i + 1) * chunk] for i in range(tp_size)]

    assert weight.dim() == 2, f"Linear weight should be 2D, got {weight.dim()}D for {key}"
    weight_t = weight.t()

    if split_type == "col":
        assert weight_t.shape[1] % tp_size == 0, f"out_dim ({weight_t.shape[1]}) must be divisible by tp_size ({tp_size}) for {key}"
        chunk = weight_t.shape[1] // tp_size
        return [weight_t[:, i * chunk : (i + 1) * chunk].t() for i in range(tp_size)]

    assert weight_t.shape[0] % tp_size == 0, f"in_dim ({weight_t.shape[0]}) must be divisible by tp_size ({tp_size}) for {key}"
    chunk = weight_t.shape[0] // tp_size
    return [weight_t[i * chunk : (i + 1) * chunk, :].t() for i in range(tp_size)]
