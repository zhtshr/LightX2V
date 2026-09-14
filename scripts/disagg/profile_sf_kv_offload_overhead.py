#!/usr/bin/env python3
"""Profile where kv_offload overhead comes from (5s A/B).

Compares off vs on with profile_kv_store + offload-specific probes:
  begin_wait / prefetch_h2d / store_d2h (CPU writeback)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)


class OffloadExtraProbe:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.begin_wait_ms = 0.0
        self.prefetch_h2d_ms = 0.0
        self.store_d2h_ms = 0.0
        self.begin_calls = 0
        self.prefetch_calls = 0
        self.d2h_calls = 0
        self._pending: list[tuple[torch.cuda.Event, torch.cuda.Event, str]] = []

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for s, e, kind in self._pending:
            ms = float(s.elapsed_time(e))
            if kind == "begin_wait":
                self.begin_wait_ms += ms
            elif kind == "prefetch_h2d":
                self.prefetch_h2d_ms += ms
            elif kind == "store_d2h":
                self.store_d2h_ms += ms
        self._pending.clear()

    def snapshot(self) -> dict[str, float | int]:
        self.finalize()
        return {
            "begin_wait_ms": round(self.begin_wait_ms, 3),
            "prefetch_h2d_ms": round(self.prefetch_h2d_ms, 3),
            "store_d2h_ms": round(self.store_d2h_ms, 3),
            "begin_calls": self.begin_calls,
            "prefetch_calls": self.prefetch_calls,
            "d2h_calls": self.d2h_calls,
        }


def _patch_offload_extra(kv_cache: Any, probe: OffloadExtraProbe) -> None:
    """Instrument dual-buffer begin wait / H2D prefetch / async D2H writeback."""
    orig_begin = kv_cache.begin_layer
    orig_prefetch_into = kv_cache._prefetch_into_slot
    orig_schedule_wb = kv_cache._schedule_writeback
    orig_copy_gpu = kv_cache._copy_layer_to_gpu
    orig_copy_cpu = kv_cache._copy_layer_to_cpu

    def begin_layer(layer_id: int) -> None:
        if not kv_cache._kv_offload:
            return orig_begin(layer_id)
        layer_id = int(layer_id)
        probe.begin_calls += 1
        slot = None
        for s in (0, 1):
            if kv_cache._slot_layer[s] == layer_id:
                slot = s
                break
        if slot is None:
            if kv_cache._slot_layer[0] < 0:
                slot = 0
            elif kv_cache._slot_layer[1] < 0:
                slot = 1
            else:
                slot = 1 - kv_cache._active_slot
            kv_cache._prefetch_into_slot(layer_id, slot)
        kv_cache._active_slot = slot
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda.current_stream().wait_event(kv_cache._load_done[slot])
        _ = torch.empty(1, device=kv_cache._device)
        e.record()
        probe._pending.append((s, e, "begin_wait"))

    def _prefetch_into_slot(layer_id: int, slot: int) -> None:
        if layer_id >= kv_cache._num_layers:
            return
        if kv_cache._slot_layer[slot] == int(layer_id):
            return
        probe.prefetch_calls += 1
        with torch.cuda.stream(kv_cache._prefetch_stream):
            kv_cache._prefetch_stream.wait_event(kv_cache._d2h_done[slot])
            kv_cache._prefetch_stream.wait_event(kv_cache._cpu_update_event(layer_id))
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(kv_cache._prefetch_stream)
            orig_copy_gpu(layer_id, slot)
            e.record(kv_cache._prefetch_stream)
            probe._pending.append((s, e, "prefetch_h2d"))
            kv_cache._load_done[slot].record(kv_cache._prefetch_stream)
        kv_cache._slot_layer[slot] = int(layer_id)

    def _schedule_writeback(layer_id: int, slot: int) -> None:
        if not kv_cache._slot_has_dirty(slot):
            return
        compute_done = kv_cache._slot_compute_done[slot]
        compute_done.record(torch.cuda.current_stream())
        probe.d2h_calls += 1
        with torch.cuda.stream(kv_cache._writeback_stream):
            kv_cache._writeback_stream.wait_event(compute_done)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(kv_cache._writeback_stream)
            orig_copy_cpu(layer_id, slot)
            e.record(kv_cache._writeback_stream)
            probe._pending.append((s, e, "store_d2h"))
            kv_cache._cpu_update_event(layer_id).record(kv_cache._writeback_stream)
            kv_cache._d2h_done[slot].record(kv_cache._writeback_stream)
        kv_cache._clear_slot_dirty(slot)

    kv_cache.begin_layer = begin_layer  # type: ignore[method-assign]
    kv_cache._prefetch_into_slot = _prefetch_into_slot  # type: ignore[method-assign]
    kv_cache._schedule_writeback = _schedule_writeback  # type: ignore[method-assign]
    _ = orig_begin, orig_prefetch_into, orig_schedule_wb


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


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


def _latent_shape(cfg: dict[str, Any]) -> list[int]:
    return [
        cfg.get("num_channels_latents", 16),
        (cfg["target_video_length"] - 1) // cfg["vae_stride"][0] + 1,
        cfg["target_height"] // cfg["vae_stride"][1],
        cfg["target_width"] // cfg["vae_stride"][2],
    ]


def _run_mode(
    *,
    model_path: str,
    config_json: str,
    kv_offload: bool,
    payload: dict[str, Any],
    seq_p_size: int,
    seed: int,
) -> dict[str, Any]:
    cfg = set_config(model_path=model_path, task="t2v", model_cls="wan2.1_sf", config_path=config_json)
    cfg["cpu_offload"] = False
    ar = dict(cfg.get("ar_config", {}))
    ar["kv_offload"] = kv_offload
    ar["profile_kv_store"] = True
    cfg["ar_config"] = ar
    if seq_p_size > 1:
        cfg["parallel"] = {"seq_p_size": seq_p_size, "seq_p_attn_type": "ulysses"}
    else:
        cfg["parallel"] = False

    seed_all(seed)
    # parallel env already initialized by caller when seq_p>1
    if cfg.get("parallel"):
        set_parallel_config(cfg)

    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)
    _sync()

    inputs = payload["inputs"]
    latent = list(payload["latent_shape"])
    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent)
    scheduler.num_output_frames = num_out
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=seed, latent_shape=list(ls_adj), image_encoder_output=None)

    extra = OffloadExtraProbe()
    if kv_offload:
        _patch_offload_extra(model.kv_cache_manager.self_attn_kv_cache, extra)

    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

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
            scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
            model.infer(inputs)
    finally:
        kv_prof = ti.finalize_kv_store_profile() if hasattr(ti, "finalize_kv_store_profile") else {}
        extra_snap = extra.snapshot() if kv_offload else {}
        DisaggSFKVCacheManager.teardown(model)
    _sync()
    wall_s = time.perf_counter() - t0
    peak = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else None

    wall_ms = wall_s * 1000.0
    sa = float(kv_prof.get("self_attn_ms", 0))
    store = float(kv_prof.get("store_kv_ms", 0))
    read = float(kv_prof.get("kv_read_ms", 0))
    cross = float(kv_prof.get("cross_attn_ms", 0))
    a2a = float(kv_prof.get("sp_a2a_ms", 0))
    other = max(0.0, wall_ms - sa - cross)  # rough: wall includes sync; self_attn embeds store/read/a2a

    out = {
        "kv_offload": kv_offload,
        "wall_s": round(wall_s, 4),
        "peak_mb": peak,
        "num_chunks": num_chunks,
        "kv_profile": kv_prof,
        "offload_extra": extra_snap,
        "breakdown_ms": {
            "wall": round(wall_ms, 2),
            "self_attn": round(sa, 2),
            "cross_attn": round(cross, 2),
            "store_kv": round(store, 2),
            "kv_read": round(read, 2),
            "sp_a2a_in_self_attn": round(a2a, 2),
            "begin_wait": round(float(extra_snap.get("begin_wait_ms", 0)), 2),
            "prefetch_h2d": round(float(extra_snap.get("prefetch_h2d_ms", 0)), 2),
            "store_d2h": round(float(extra_snap.get("store_d2h_ms", 0)), 2),
            "non_attn_proxy": round(other, 2),
        },
    }
    del model, scheduler
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    _sync()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_14b_int8_sp_5s_kivi.json")
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_14b_int8_encoder_inputs_f81.pt")
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_14b_kv_offload_overhead_profile_5s.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--cuda_devices", default="0,2,4,5")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "no_offload", "kv_offload"],
        help="Run one mode or both (both may OOM if model not fully freed)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Build config once for payload shape / parallel init
    base = set_config(model_path=args.model_path, task="t2v", model_cls="wan2.1_sf", config_path=args.config_json)
    base["cpu_offload"] = False
    if args.seq_p_size > 1:
        base["parallel"] = {"seq_p_size": args.seq_p_size, "seq_p_attn_type": "ulysses"}
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(base)

    latent = _latent_shape(base)
    cache = Path(args.inputs_cache)
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        payload["latent_shape"] = latent
        payload["inputs"]["latent_shape"] = latent
        if dist.is_initialized():
            dist.barrier()
    else:
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            payload = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            text_encoder = load_wan_text_encoder(base)[0]
            text_len = int(base.get("text_len", 512))
            context = text_encoder.infer([args.prompt])
            context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
            del text_encoder
            payload = {
                "seed": args.seed,
                "latent_shape": latent,
                "inputs": {
                    "text_encoder_output": {"context": context, "context_null": None},
                    "image_encoder_output": None,
                    "latent_shape": latent,
                },
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache)
            if dist.is_initialized():
                dist.barrier()
    payload = _move(payload, _device())

    modes = {}
    run_list = (
        [("no_offload", False), ("kv_offload", True)]
        if args.mode == "both"
        else [(args.mode, args.mode == "kv_offload")]
    )
    for name, off in run_list:
        if is_main_process():
            print(f"\n=== {name} ===")
        modes[name] = _run_mode(
            model_path=args.model_path,
            config_json=args.config_json,
            kv_offload=off,
            payload=payload,
            seq_p_size=args.seq_p_size,
            seed=args.seed,
        )
        if is_main_process():
            b = modes[name]["breakdown_ms"]
            print(
                f"wall={modes[name]['wall_s']:.3f}s peak={modes[name]['peak_mb']} "
                f"self_attn={b['self_attn']:.0f} store={b['store_kv']:.0f} "
                f"kv_read={b['kv_read']:.0f} a2a={b['sp_a2a_in_self_attn']:.0f} "
                f"wait={b['begin_wait']:.0f} h2d={b['prefetch_h2d']:.0f} d2h={b['store_d2h']:.0f}"
            )

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        if "no_offload" in modes and "kv_offload" in modes:
            a, b = modes["no_offload"], modes["kv_offload"]
            ba, bb = a["breakdown_ms"], b["breakdown_ms"]
            delta = {
                k: round(bb[k] - ba[k], 2)
                for k in ("wall", "self_attn", "cross_attn", "store_kv", "kv_read", "sp_a2a_in_self_attn", "non_attn_proxy")
            }
            delta["begin_wait"] = bb["begin_wait"]
            delta["prefetch_h2d"] = bb["prefetch_h2d"]
            delta["store_d2h"] = bb["store_d2h"]
            wall_delta = bb["wall"] - ba["wall"]
            contrib = {
                "store_kv_delta_ms": delta["store_kv"],
                "kv_read_delta_ms": delta["kv_read"],
                "self_attn_delta_ms": delta["self_attn"],
                "begin_wait_ms": delta["begin_wait"],
                "store_d2h_ms": delta["store_d2h"],
                "prefetch_h2d_ms": delta["prefetch_h2d"],
                "wall_delta_ms": round(wall_delta, 2),
            }
            result = {
                "metric": "sf_kv_offload_overhead_profile",
                "setting": "INT8+KIVI, Ulysses P=4, 5s A/B",
                "modes": modes,
                "delta_ms": delta,
                "contrib": contrib,
                "note": (
                    "store_d2h ⊆ store_kv; begin_wait ⊆ self_attn; prefetch_h2d overlaps previous layer "
                    "compute (only begin_wait is true stall)."
                ),
            }
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("\n=== delta (offload - baseline) ms ===")
            print(json.dumps(delta, indent=2))
            print("=== contrib ===")
            print(json.dumps(contrib, indent=2))
        else:
            out.write_text(json.dumps({"modes": modes}, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
