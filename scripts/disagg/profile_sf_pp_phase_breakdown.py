#!/usr/bin/env python3
"""Profile SF 14B PP=2 phase breakdown (pre / P2P / stage blocks / attn / FFN).

Compares against the single-GPU 1.3B ``sf_forward_breakdown`` style buckets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_transformer, set_config
from lightx2v.models.networks.wan.infer.pipeline_parallel import (
    recv_activation,
    recv_noise_pred,
    recv_pre_metadata,
    send_activation,
    send_noise_pred,
    send_pre_metadata,
)
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

SAFE_CUDA = {2: "0,2"}
FORBIDDEN = {"1", "3"}


class CudaTimer:
    def __init__(self) -> None:
        self._s: torch.cuda.Event | None = None

    def start(self) -> None:
        self._s = torch.cuda.Event(enable_timing=True)
        self._s.record()

    def stop_ms(self) -> float:
        assert self._s is not None
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        return float(self._s.elapsed_time(end))


def _assert_safe(cuda_devices: str) -> None:
    bad = {x.strip() for x in cuda_devices.split(",") if x.strip()} & FORBIDDEN
    if bad:
        raise RuntimeError(f"forbidden GPUs {sorted(bad)} in CUDA_VISIBLE_DEVICES={cuda_devices}")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _move(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move(v, device) for v in obj)
    return obj


def _patch_pp_profiled_infer(model: Any, acc: dict[str, float]) -> None:
    """Replace _infer_cond_uncond_pp with timed segments; accumulate ms into ``acc``."""

    def _infer_pp(self, inputs, infer_condition=True):
        from lightx2v.models.networks.wan.infer.pipeline_parallel import (  # noqa: F401
            recv_activation as _ra,
            recv_noise_pred as _rn,
            recv_pre_metadata as _rm,
            send_activation as _sa,
            send_noise_pred as _sn,
            send_pre_metadata as _sm,
        )

        self.scheduler.infer_condition = infer_condition
        last = self.pp_size - 1
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = CudaTimer()

        if self.pp_rank == 0:
            t.start()
            pre = self.pre_infer.infer(self.pre_weight, inputs)
            acc["pre_infer"] += t.stop_ms()

            t.start()
            send_pre_metadata(pre, last, self.pp_group)
            acc["p2p_send_meta"] += t.stop_ms()

            self.transformer_infer.cos_sin = pre.cos_sin
            self.transformer_infer.reset_infer_states()
            t.start()
            x = self.transformer_infer.infer_main_blocks(self.transformer_weights.blocks, pre)
            acc["blocks_stage0"] += t.stop_ms()

            t.start()
            send_activation(x, last, self.pp_group)
            acc["p2p_send_activation"] += t.stop_ms()

            t.start()
            noise_pred = recv_noise_pred(last, self.pp_group, device)
            acc["p2p_recv_noise_pred"] += t.stop_ms()
            return noise_pred

        if self.pp_rank == last:
            t.start()
            pre = recv_pre_metadata(0, self.pp_group, device)
            acc["p2p_recv_meta"] += t.stop_ms()

            t.start()
            x = recv_activation(0, self.pp_group, device)
            acc["p2p_recv_activation"] += t.stop_ms()

            pre.x = x
            if getattr(pre, "freqs", None) is None and hasattr(self.pre_infer, "freqs"):
                pre.freqs = self.pre_infer.freqs
            if getattr(pre, "seq_lens", None) is None:
                pre.seq_lens = torch.tensor([x.size(0)], dtype=torch.int32, device=device).unsqueeze(0)

            self.transformer_infer.cos_sin = pre.cos_sin
            self.transformer_infer.reset_infer_states()
            t.start()
            x = self.transformer_infer.infer_main_blocks(self.transformer_weights.blocks, pre)
            acc["blocks_stage1"] += t.stop_ms()

            t.start()
            x = self.transformer_infer.infer_non_blocks(self.transformer_weights, x, pre.embed)
            noise_pred = self.post_infer.infer(x, pre)[0]
            acc["post_infer"] += t.stop_ms()

            t.start()
            send_noise_pred(noise_pred, 0, self.pp_group)
            acc["p2p_send_noise_pred"] += t.stop_ms()
            return noise_pred

        raise RuntimeError(f"unsupported pp_rank={self.pp_rank}")

    import types

    model._infer_cond_uncond_pp = types.MethodType(_infer_pp, model)


def _enable_kv_profiler(config: dict[str, Any]) -> None:
    ar = dict(config.get("ar_config") or {})
    ar["profile_kv_store"] = True
    config["ar_config"] = ar


def _run_full(model, scheduler, config, payload, include_rerun: bool) -> dict[str, Any]:
    inputs = payload["inputs"]
    latent_shape = list(payload["latent_shape"])
    seed = int(payload["seed"])

    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_out
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=seed, latent_shape=list(ls_adj), image_encoder_output=None)

    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    seg_acc: dict[str, float] = defaultdict(float)
    _patch_pp_profiled_infer(model, seg_acc)

    infer_steps = int(scheduler.infer_steps)
    _sync()
    t0 = time.perf_counter()
    try:
        for seg_idx in range(num_chunks):
            for step_index in range(infer_steps):
                model.kv_cache_manager.current_step = step_index
                scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
                model.infer(inputs)
                scheduler.step_post()
            if include_rerun:
                scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
                model.infer(inputs)
    finally:
        DisaggSFKVCacheManager.teardown(model)
    _sync()
    wall_s = time.perf_counter() - t0

    kv = ti.finalize_kv_store_profile() if hasattr(ti, "finalize_kv_store_profile") else {}
    return {
        "wall_s": wall_s,
        "num_chunks": num_chunks,
        "infer_steps": infer_steps,
        "include_rerun": include_rerun,
        "num_forwards": num_chunks * infer_steps + (num_chunks if include_rerun else 0),
        "segments_ms_local": {k: round(v, 3) for k, v in seg_acc.items()},
        "kv_profiler": kv,
    }


def _allgather_dict(local: dict[str, float], device: torch.device) -> list[dict[str, float]]:
    """Gather float dicts from all ranks; key set = union across ranks."""
    local_keys = sorted(local.keys())
    if not dist.is_initialized():
        return [dict(local)]

    # Collect key lists then take union so stage0/stage1 keys both appear.
    key_lists: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(key_lists, local_keys)
    keys = sorted(set().union(*(set(k or []) for k in key_lists)))

    vec = torch.tensor([float(local.get(k, 0.0)) for k in keys], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(vec) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, vec)
    return [dict(zip(keys, g.tolist())) for g in gathered]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument(
        "--config_json",
        default="configs/self_forcing/wan_t2v_sf_14b_bf16_pp2_5s_kivi.json",
    )
    parser.add_argument("--pipe_p_size", type=int, default=2)
    parser.add_argument("--cuda_devices", default="0,2")
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/sf_14b_pp2_encoder_inputs_5s.pt",
    )
    parser.add_argument(
        "--output_json",
        default="save_results/optimization_study/sf_14b_bf16_pp2_5s_phase_profile.json",
    )
    parser.add_argument("--output_md", default="save_results/optimization_study/sf_14b_bf16_pp2_5s_phase_profile.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_rerun", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    _assert_safe(args.cuda_devices)

    config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    config["cpu_offload"] = False
    config["parallel"] = {"pipe_p_size": int(args.pipe_p_size)}
    _enable_kv_profiler(config)
    seed_all(args.seed)

    PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda")).init_parallel_env()
    set_parallel_config(config)

    device = torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    payload = torch.load(args.inputs_cache, map_location="cpu", weights_only=False)
    payload = _move(payload, device)

    model = load_wan_transformer(config)
    scheduler = load_wan_scheduler(config)
    model.set_scheduler(scheduler)
    _sync()

    meta = _run_full(model, scheduler, config, payload, include_rerun=not args.no_rerun)
    segs_all = _allgather_dict(meta["segments_ms_local"], device)
    kv_all = _allgather_dict(
        {k: float(v) for k, v in (meta.get("kv_profiler") or {}).items() if isinstance(v, (int, float))},
        device,
    )

    # Critical-path estimate for single-request PP=2 (no microbatch overlap):
    # pre + send_meta + stage0 + send_act + stage1 + post + send_noise
    # (recv_* on the idle rank is waiting, not additive)
    r0 = segs_all[0]
    r1 = segs_all[1] if len(segs_all) > 1 else {}
    crit = {
        "pre_infer": r0.get("pre_infer", 0.0),
        "p2p_meta": r0.get("p2p_send_meta", 0.0),
        "blocks_stage0": r0.get("blocks_stage0", 0.0),
        "p2p_activation": r0.get("p2p_send_activation", 0.0),
        "blocks_stage1": r1.get("blocks_stage1", 0.0),
        "post_infer": r1.get("post_infer", 0.0),
        "p2p_noise": r1.get("p2p_send_noise_pred", 0.0),
    }
    crit_total = sum(crit.values())
    blocks_total = crit["blocks_stage0"] + crit["blocks_stage1"]

    # Attn/FFN from both ranks (each owns half the layers)
    kv0, kv1 = kv_all[0], (kv_all[1] if len(kv_all) > 1 else {})
    sa = kv0.get("self_attn_ms", 0.0) + kv1.get("self_attn_ms", 0.0)
    ca = kv0.get("cross_attn_ms", 0.0) + kv1.get("cross_attn_ms", 0.0)
    store = kv0.get("store_kv_ms", 0.0) + kv1.get("store_kv_ms", 0.0)
    read = kv0.get("kv_read_ms", 0.0) + kv1.get("kv_read_ms", 0.0)
    ffn_other = max(0.0, blocks_total - sa - ca)

    result = {
        "description": "SF 14B BF16 PP=2 phase profile (CUDA events + kv_store profiler)",
        "config_json": args.config_json,
        "pipe_p_size": args.pipe_p_size,
        "dit_quantized": bool(config.get("dit_quantized", False)),
        "kv_offload": bool((config.get("ar_config") or {}).get("kv_offload")),
        "kv_quant": bool((config.get("ar_config") or {}).get("kv_quant")),
        "resolution": f"{config.get('target_width')}x{config.get('target_height')}",
        "target_video_length": config.get("target_video_length"),
        "num_layers": int(config.get("num_layers", 0)),
        "dim": int(config.get("dim", 0)),
        "wall_s": round(meta["wall_s"], 4),
        "num_chunks": meta["num_chunks"],
        "num_forwards": meta["num_forwards"],
        "segments_ms_per_rank": [{k: round(v, 3) for k, v in d.items()} for d in segs_all],
        "critical_path_ms": {k: round(v, 3) for k, v in crit.items()},
        "critical_path_pct": {k: round(100 * v / crit_total, 2) if crit_total else 0 for k, v in crit.items()},
        "critical_path_total_ms": round(crit_total, 3),
        "transformer_blocks_ms": round(blocks_total, 3),
        "transformer_breakdown_ms": {
            "self_attn": round(sa, 3),
            "cross_attn": round(ca, 3),
            "ffn_and_block_overhead": round(ffn_other, 3),
            "self_attn_store_kv": round(store, 3),
            "self_attn_kv_read": round(read, 3),
        },
        "transformer_breakdown_pct_of_blocks": {
            "self_attn": round(100 * sa / blocks_total, 2) if blocks_total else 0,
            "cross_attn": round(100 * ca / blocks_total, 2) if blocks_total else 0,
            "ffn_and_block_overhead": round(100 * ffn_other / blocks_total, 2) if blocks_total else 0,
        },
        "p2p_total_ms": round(
            crit["p2p_meta"] + crit["p2p_activation"] + crit["p2p_noise"], 3
        ),
        "kv_profiler_per_rank": kv_all,
        "baseline_1p3b_sf_forward_breakdown": {
            "note": "wan2.1_sf 1.3B single-GPU 832x480, from save_results/sf_forward_breakdown.md",
            "self_attn_pct": 63.05,
            "cross_attn_pct": 9.0,
            "ffn_and_block_overhead_pct": 27.95,
            "per_forward_ms": 488.17,
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in (
            "wall_s", "critical_path_total_ms", "critical_path_pct",
            "transformer_breakdown_pct_of_blocks", "p2p_total_ms",
        )}, indent=2))
        print(f"wrote {out}")

        md = Path(args.output_md)
        lines = [
            "# SF 14B BF16 PP=2 Phase Profile",
            "",
            f"- Config: `{args.config_json}`",
            f"- Resolution: {result['resolution']}, length={result['target_video_length']}, "
            f"layers={result['num_layers']}, dim={result['dim']}",
            f"- Weight quant: **no** (BF16), KV: KIVI, kv_offload={result['kv_offload']}",
            f"- Wall: **{result['wall_s']} s** ({result['num_forwards']} forwards / {result['num_chunks']} chunks)",
            "",
            "## Critical-path stages (single-request PP=2)",
            "",
            "| Stage | ms | % |",
            "|-------|-----|---|",
        ]
        for k, v in result["critical_path_ms"].items():
            lines.append(f"| {k} | {v} | {result['critical_path_pct'][k]}% |")
        lines += [
            f"| **total** | **{result['critical_path_total_ms']}** | 100% |",
            "",
            f"P2P sum (meta+act+noise): **{result['p2p_total_ms']} ms** "
            f"({round(100 * result['p2p_total_ms'] / result['critical_path_total_ms'], 2)}% of critical path)",
            "",
            "## Transformer blocks (stage0+stage1)",
            "",
            "| Component | ms | % of blocks |",
            "|-----------|-----|-------------|",
        ]
        for k, v in result["transformer_breakdown_ms"].items():
            if k.startswith("self_attn_store") or k.startswith("self_attn_kv"):
                lines.append(f"| {k} | {v} | - |")
            else:
                pct = result["transformer_breakdown_pct_of_blocks"].get(k, "-")
                lines.append(f"| {k} | {v} | {pct}% |")
        lines += [
            "",
            "## vs prior `sf_forward_breakdown` (1.3B single-GPU)",
            "",
            "| Bucket | 1.3B P=1 | 14B BF16 PP=2 |",
            "|--------|----------|---------------|",
            f"| self_attn | 63.1% | {result['transformer_breakdown_pct_of_blocks']['self_attn']}% |",
            f"| cross_attn | 9.0% | {result['transformer_breakdown_pct_of_blocks']['cross_attn']}% |",
            f"| ffn_and_block | 28.0% | {result['transformer_breakdown_pct_of_blocks']['ffn_and_block_overhead']}% |",
            "",
            "Notes:",
            "- 1.3B profile had **no PP** (no P2P stages); 14B adds stage split + activation P2P.",
            "- 14B is BF16 weights (no INT8 GEMM); 1.3B was also non-quant in that profile.",
            "- 14B uses KIVI KV + kv_offload; 1.3B breakdown had negligible kv_io (~0.8%).",
            "- PP does **not** reduce single-request compute: blocks_stage0+stage1 ≈ full 40-layer work.",
            "",
        ]
        md.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {md}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
