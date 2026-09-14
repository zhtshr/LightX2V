#!/usr/bin/env python3
"""Convert Self-Forcing Wan DiT safetensors (model.* keys) to per-channel INT8.

Avoids tools/convert/converter.py because it imports qtorch (broken gcc/nvcc on this box).
Scale key format matches LightX2V: `<weight_key>_scale` -> `xxx.weight_scale`.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

TARGET_PARTS = {"self_attn", "cross_attn", "ffn"}


def should_quantize(key: str, key_idx: int = 3) -> bool:
    if not key.endswith(".weight"):
        return False
    parts = key.split(".")
    return len(parts) > key_idx and parts[key_idx] in TARGET_PARTS


@torch.no_grad()
def quantize_int8_perchannel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # per-output-channel symmetric int8
    max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = max_val / 127.0
    w_q = torch.clamp(torch.round(w / scales), -128, 127).to(torch.int8)
    return w_q, scales.to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-idx", type=int, default=3, help="self_forcing=3, wan_dit=2")
    parser.add_argument("--non-linear-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    nl_dtype = getattr(torch, args.non_linear_dtype)

    print(f"Loading {src} ...")
    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in tqdm(keys, desc="load"):
            weights[k] = f.get_tensor(k)

    n_q = 0
    for key in tqdm(list(weights.keys()), desc="quantize"):
        t = weights[key]
        if not isinstance(t, torch.Tensor):
            continue
        if should_quantize(key, args.key_idx) and t.dim() == 2:
            w_q, scales = quantize_int8_perchannel(t.float())
            weights[key] = w_q
            weights[key + "_scale"] = scales
            n_q += 1
            del w_q, scales
        else:
            if t.dtype != nl_dtype and t.is_floating_point():
                weights[key] = t.to(nl_dtype)
        if n_q % 50 == 0:
            gc.collect()

    total = sum(v.numel() * v.element_size() for v in weights.values())
    print(f"Quantized {n_q} linear weights; output size={total / 1e9:.2f} GB")
    print(f"Saving {out} ...")
    save_file(weights, str(out))
    print("done")


if __name__ == "__main__":
    main()
