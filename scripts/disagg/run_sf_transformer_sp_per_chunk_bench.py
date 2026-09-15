#!/usr/bin/env python3
"""SF transformer per-chunk latency bench with seq parallel (stripe / ulysses).

Measures wall time per chunk including all infer steps + rerun, with CUDA sync
and distributed barrier so communication is included.
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

from lightx2v.common.kvcache.base import (
    enable_stripe_attn_profiler,
    enable_stripe_gather_skew,
    enable_ulysses_attn_profiler,
    get_stripe_attn_profiler,
    get_stripe_gather_skew,
    get_ulysses_attn_profiler,
)
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

# GPU1 (0000:6A:00.0) and GPU3 (0000:72:00.0) excluded: Xid-79 bus falloff.
# Physical indices only — remapped cuda:0..N never include physical GPU1/3.
SAFE_CUDA_DEVICES = {1: "0", 2: "0,2", 4: "0,2,4,5", 6: "0,2,4,5,6,7"}
FORBIDDEN_PHYSICAL_GPUS = {"1", "3"}


def _assert_no_gpu3(cuda_devices: str) -> None:
    ids = {x.strip() for x in cuda_devices.split(",") if x.strip()}
    bad = ids & FORBIDDEN_PHYSICAL_GPUS
    if bad:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cuda_devices!r} includes forbidden physical GPU(s) "
            f"{sorted(bad)}; use 0,2,4,5 for P=4 (skip 1 and 3)."
        )


def _run_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _move_tensor_tree(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_tensor_tree(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tensor_tree(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(value, device) for value in obj)
    return obj


def _latent_shape_from_config(config: dict[str, Any]) -> list[int]:
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    config["target_video_length"] = int(args.target_video_length)
    config["cpu_offload"] = False
    ar = dict(config.get("ar_config", {}))
    ar["profile_kv_store"] = True
    config["ar_config"] = ar
    if args.seq_p_size > 1:
        config["parallel"] = {
            "seq_p_size": int(args.seq_p_size),
            "seq_p_attn_type": args.seq_p_attn_type,
        }
        if getattr(args, "stripe_full_q", False):
            config["parallel"]["stripe_full_q"] = True
            os.environ["LIGHTX2V_STRIPE_FULL_Q"] = "1"
        else:
            os.environ.pop("LIGHTX2V_STRIPE_FULL_Q", None)
        if getattr(args, "stripe_partial_out", False) or args.seq_p_attn_type == "stripe_pe":
            config["parallel"]["stripe_partial_out"] = True
            os.environ["LIGHTX2V_STRIPE_PARTIAL_OUT"] = "1"
        else:
            os.environ.pop("LIGHTX2V_STRIPE_PARTIAL_OUT", None)
        if getattr(args, "stripe_q_exchange", False) or args.seq_p_attn_type == "stripe_b2":
            config["parallel"]["stripe_q_exchange"] = True
            os.environ["LIGHTX2V_STRIPE_Q_EXCHANGE"] = "1"
        else:
            os.environ.pop("LIGHTX2V_STRIPE_Q_EXCHANGE", None)
        if getattr(args, "stripe_hier", False) or args.seq_p_attn_type == "stripe_hier":
            config["parallel"]["stripe_hier"] = True
            config["parallel"]["seq_p_attn_type"] = "stripe_hier"
            os.environ["LIGHTX2V_STRIPE_HIER"] = "1"
            # Hier is mutually exclusive with Form C / B2 flags.
            config["parallel"].pop("stripe_partial_out", None)
            config["parallel"].pop("stripe_q_exchange", None)
            os.environ.pop("LIGHTX2V_STRIPE_PARTIAL_OUT", None)
            os.environ.pop("LIGHTX2V_STRIPE_Q_EXCHANGE", None)
        else:
            os.environ.pop("LIGHTX2V_STRIPE_HIER", None)
        # Form C early-chunk → hier switch (inclusive). -1 disables.
        if getattr(args, "formc_hier_until", None) is not None:
            until = int(args.formc_hier_until)
            config["parallel"]["stripe_formc_hier_until_chunk"] = until
            os.environ["LIGHTX2V_STRIPE_FORMC_HIER_UNTIL"] = str(until)
    else:
        config["parallel"] = False
    return config


def _init_distributed(config: dict[str, Any]) -> None:
    if not config.get("parallel"):
        return
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _prepare_inputs_cache(
    config: dict[str, Any],
    cache_path: Path,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    latent_shape = _latent_shape_from_config(config)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        payload["latent_shape"] = latent_shape
        payload["inputs"]["latent_shape"] = latent_shape
        if is_main_process():
            print(f"Loaded encoder cache {cache_path}, latent_shape={latent_shape}")
        if dist.is_initialized():
            dist.barrier()
        return payload

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        payload["latent_shape"] = latent_shape
        payload["inputs"]["latent_shape"] = latent_shape
        return payload

    print("Preparing T5 encoder inputs...")
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    payload = {
        "seed": seed,
        "latent_shape": latent_shape,
        "inputs": {
            "text_encoder_output": {"context": context, "context_null": None},
            "image_encoder_output": None,
            "latent_shape": latent_shape,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    if dist.is_initialized():
        dist.barrier()
    return payload


def _reset_chunk_profiles(model: Any, stripe_profile: bool, ulysses_profile: bool) -> None:
    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()
    if stripe_profile:
        prof = get_stripe_attn_profiler()
        if prof is not None:
            prof.reset()
    if ulysses_profile:
        prof = get_ulysses_attn_profiler()
        if prof is not None:
            prof.reset()


def _finalize_chunk_profiles(model: Any, stripe_profile: bool, ulysses_profile: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ti = model.transformer_infer
    if hasattr(ti, "finalize_kv_store_profile"):
        kv = ti.finalize_kv_store_profile() or {}
        out["transformer_ms"] = round(float(kv.get("transformer_ms", 0.0)), 2)
        out["self_attn_ms"] = round(float(kv.get("self_attn_ms", 0.0)), 2)
        out["cross_attn_ms"] = round(float(kv.get("cross_attn_ms", 0.0)), 2)
        out["store_kv_ms"] = round(float(kv.get("store_kv_ms", 0.0)), 2)
        out["kv_read_ms"] = round(float(kv.get("kv_read_ms", 0.0)), 2)
        out["sp_kv_a2a_ms"] = round(float(kv.get("sp_a2a_ms", 0.0)), 2)
    if stripe_profile:
        prof = get_stripe_attn_profiler()
        if prof is not None:
            snap = prof.snapshot()
            out["stripe_self_attn"] = snap
            comm_ms = float(snap.get("comm_ms", 0.0))
            if comm_ms == 0.0:
                comm_ms = (
                    snap.get("all_gather_q_ms", 0.0)
                    + snap.get("alltoall_q_ms", 0.0)
                    + snap.get("all_gather_out_ms", 0.0)
                    + snap.get("all_gather_lse_ms", 0.0)
                    + snap.get("alltoall_out_ms", 0.0)
                    + snap.get("alltoall_lse_ms", 0.0)
                )
            out["stripe_comm_ms"] = round(comm_ms, 2)
            out["stripe_flash_ms"] = round(float(snap.get("flash_ms", 0.0)), 2)
    if ulysses_profile:
        prof = get_ulysses_attn_profiler()
        if prof is not None:
            snap = prof.snapshot()
            out["ulysses_self_attn"] = snap
            out["ulysses_comm_ms"] = round(float(snap.get("comm_ms", 0.0)), 2)
            out["ulysses_flash_ms"] = round(float(snap.get("flash_ms", 0.0)), 2)
            out["ulysses_a2a_q_ms"] = round(float(snap.get("all2all_q_ms", 0.0)), 2)
            out["ulysses_a2a_out_ms"] = round(float(snap.get("all2all_out_ms", 0.0)), 2)
    return out


def _run_per_chunk(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
    stripe_profile: bool,
    ulysses_profile: bool,
) -> dict[str, Any]:
    inputs = payload["inputs"]
    latent_shape = list(payload["latent_shape"])
    seed = int(payload["seed"])

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)

    kv_mgr = model.kv_cache_manager
    infer_steps = int(scheduler.infer_steps)
    tokens_per_chunk = kv_mgr.frame_seq_length * config["ar_config"].get("num_frame_per_chunk", 3)
    chunk_rows: list[dict[str, Any]] = []

    try:
        for seg_idx in range(num_chunks):
            _reset_chunk_profiles(model, stripe_profile, ulysses_profile)
            _sync_device()
            t0 = time.perf_counter()

            for step_index in range(infer_steps):
                kv_mgr.current_step = step_index
                scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
                model.infer(inputs)
                scheduler.step_post()

            if include_rerun:
                scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
                model.infer(inputs)

            _sync_device()
            chunk_wall_s = time.perf_counter() - t0
            prof = _finalize_chunk_profiles(model, stripe_profile, ulysses_profile)
            global_k = (seg_idx + 1) * tokens_per_chunk

            row = {
                "chunk": seg_idx,
                "global_kv_tokens": global_k,
                "q_tokens": tokens_per_chunk,
                "forwards": infer_steps + (1 if include_rerun else 0),
                "chunk_wall_ms": round(chunk_wall_s * 1000.0, 2),
                "chunk_wall_ms_per_forward": round(chunk_wall_s * 1000.0 / (infer_steps + (1 if include_rerun else 0)), 2),
                **prof,
            }
            transformer_ms = float(prof.get("transformer_ms", 0.0))
            self_attn_ms = float(prof.get("self_attn_ms", 0.0))
            if self_attn_ms > 0 and transformer_ms > 0:
                row["self_attn_share_pct"] = round(100.0 * self_attn_ms / transformer_ms, 1)
            chunk_rows.append(row)

            if is_main_process():
                comm_note = ""
                if "stripe_comm_ms" in prof:
                    comm_note = f" stripe_comm={prof['stripe_comm_ms']:.0f}ms flash={prof.get('stripe_flash_ms', 0):.0f}ms"
                elif "ulysses_comm_ms" in prof:
                    wall = row["chunk_wall_ms"]
                    comm = prof["ulysses_comm_ms"]
                    flash = prof.get("ulysses_flash_ms", 0.0)
                    comm_note = (
                        f" a2a={comm:.0f}ms ({100 * comm / wall:.0f}%wall) "
                        f"flash={flash:.0f}ms ({100 * flash / wall:.0f}%wall)"
                    )
                elif prof.get("self_attn_ms"):
                    comm_note = f" self_attn={prof['self_attn_ms']:.0f}ms"
                print(
                    f"chunk {seg_idx:2d}  K={global_k:5d}  wall={row['chunk_wall_ms']:7.1f}ms "
                    f"({row['chunk_wall_ms_per_forward']:.1f}ms/fwd){comm_note}"
                )
    finally:
        DisaggSFKVCacheManager.teardown(model)

    _sync_device()
    total_wall_s = sum(r["chunk_wall_ms"] for r in chunk_rows) / 1000.0
    return {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "include_rerun": include_rerun,
        "num_output_frames": num_output_frames,
        "tokens_per_chunk": tokens_per_chunk,
        "total_wall_s": round(total_wall_s, 3),
        "chunks": chunk_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SF transformer per-chunk SP latency bench")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_video_length", type=int, default=161, help="~10s @16fps")
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--seq_p_attn_type", default="ulysses", choices=["ulysses", "stripe", "stripe_kv", "stripe_pe", "stripe_b2", "stripe_hier", "ring_kv_cache"])
    parser.add_argument(
        "--stripe_full_q",
        action="store_true",
        help="Stripe mode: keep full Q/activation on every rank (skip all_gather_q)",
    )
    parser.add_argument(
        "--stripe_partial_out",
        action="store_true",
        help="Stripe form C: alltoall partial (out,lse) rows instead of all_gather full partials",
    )
    parser.add_argument(
        "--stripe_q_exchange",
        action="store_true",
        help="Stripe form B2: send Q_local to K owners, remote Flash, partials return",
    )
    parser.add_argument(
        "--stripe_hier",
        action="store_true",
        help="Stripe hierarchical P=4: pair-exchange KV then cross-exchange Q",
    )
    parser.add_argument(
        "--formc_hier_until",
        type=int,
        default=None,
        help="FormC path: use hier for chunk index <= N, then FormC. -1 disables (pure FormC).",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_phase1_encoder_inputs.pt")
    parser.add_argument("--output_json", default="")
    parser.add_argument(
        "--gather_skew",
        action="store_true",
        help="Record per-call Q all_gather arrival skew across ranks (Form C/stripe gather path)",
    )
    parser.add_argument("--cuda_devices", default="")
    args = parser.parse_args()

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    elif args.seq_p_size in SAFE_CUDA_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = SAFE_CUDA_DEVICES[args.seq_p_size]
    _assert_no_gpu3(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    stripe_profile = args.seq_p_size > 1 and str(args.seq_p_attn_type).startswith("stripe")
    ulysses_profile = args.seq_p_attn_type == "ulysses" and args.seq_p_size > 1
    if stripe_profile:
        os.environ["LIGHTX2V_STRIPE_PROFILE"] = "1"
        enable_stripe_attn_profiler(True)
    if os.environ.get("LIGHTX2V_STRIPE_GATHER_SKEW", "0") == "1" or getattr(args, "gather_skew", False):
        os.environ["LIGHTX2V_STRIPE_GATHER_SKEW"] = "1"
        enable_stripe_gather_skew(True)
    if ulysses_profile:
        os.environ["LIGHTX2V_ULYSSES_PROFILE"] = "1"
        enable_ulysses_attn_profiler(True)

    config = _load_config(args)
    seed_all(args.seed)
    _init_distributed(config)

    payload = _prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=args.prompt,
        seed=args.seed,
    )
    payload = _move_tensor_tree(payload, _run_device())
    include_rerun = not args.no_rerun

    model = load_wan_transformer(config)
    scheduler = load_wan_scheduler(config)
    model.set_scheduler(scheduler)
    # Align ranks after load (avoids first collective hang when load skew is large).
    _sync_device()

    all_runs: list[dict[str, Any]] = []
    for run_idx in range(args.warmup + args.measure_iters):
        is_measure = run_idx >= args.warmup
        if is_main_process():
            label = "measure" if is_measure else "warmup"
            print(f"\n=== {label} run {run_idx - args.warmup + 1 if is_measure else run_idx + 1} "
                  f"seq_p={args.seq_p_size} {args.seq_p_attn_type} video_len={args.target_video_length} ===")
        meta = _run_per_chunk(
            model, scheduler, config, payload,
            include_rerun=include_rerun,
            stripe_profile=stripe_profile,
            ulysses_profile=ulysses_profile,
        )
        if is_measure:
            all_runs.append(meta)

    def _avg_chunk_ms(chunk_idx: int, field: str) -> float:
        vals = [run["chunks"][chunk_idx][field] for run in all_runs if chunk_idx < len(run["chunks"])]
        return sum(vals) / len(vals) if vals else 0.0

    summary_chunks = []
    if all_runs:
        n_chunks = all_runs[0]["num_chunks"]
        for c in range(n_chunks):
            wall = _avg_chunk_ms(c, "chunk_wall_ms")
            per_fwd = _avg_chunk_ms(c, "chunk_wall_ms_per_forward")
            row = {
                "chunk": c,
                "global_kv_tokens": all_runs[0]["chunks"][c]["global_kv_tokens"],
                "chunk_wall_ms_mean": round(wall, 2),
                "chunk_wall_ms_per_forward_mean": round(per_fwd, 2),
            }
            if all_runs[0]["chunks"][c].get("self_attn_ms"):
                row["self_attn_ms_mean"] = round(_avg_chunk_ms(c, "self_attn_ms"), 2)
            if all_runs[0]["chunks"][c].get("sp_kv_a2a_ms"):
                row["sp_kv_a2a_ms_mean"] = round(_avg_chunk_ms(c, "sp_kv_a2a_ms"), 2)
            if all_runs[0]["chunks"][c].get("stripe_comm_ms"):
                row["stripe_comm_ms_mean"] = round(_avg_chunk_ms(c, "stripe_comm_ms"), 2)
                row["stripe_flash_ms_mean"] = round(_avg_chunk_ms(c, "stripe_flash_ms"), 2)
            if all_runs[0]["chunks"][c].get("ulysses_comm_ms"):
                row["ulysses_comm_ms_mean"] = round(_avg_chunk_ms(c, "ulysses_comm_ms"), 2)
                row["ulysses_total_comm_ms_mean"] = round(
                    row["ulysses_comm_ms_mean"] + row.get("sp_kv_a2a_ms_mean", 0.0), 2
                )
                row["ulysses_flash_ms_mean"] = round(_avg_chunk_ms(c, "ulysses_flash_ms"), 2)
                row["ulysses_a2a_q_ms_mean"] = round(_avg_chunk_ms(c, "ulysses_a2a_q_ms"), 2)
                row["ulysses_a2a_out_ms_mean"] = round(_avg_chunk_ms(c, "ulysses_a2a_out_ms"), 2)
                if wall > 0:
                    row["ulysses_comm_pct_of_wall"] = round(
                        100.0 * row["ulysses_total_comm_ms_mean"] / wall, 1
                    )
                    row["ulysses_flash_pct_of_wall"] = round(100.0 * row["ulysses_flash_ms_mean"] / wall, 1)
                    if row.get("self_attn_ms_mean"):
                        row["ulysses_comm_pct_of_self_attn"] = round(
                            100.0 * row["ulysses_total_comm_ms_mean"] / row["self_attn_ms_mean"], 1
                        )
            summary_chunks.append(row)

    result = {
        "metric": "sf_transformer_per_chunk_wall_ms",
        "description": "Per-chunk wall time (infer_steps + rerun), includes comm via sync+barrier",
        "model_cls": "wan2.1_sf",
        "target_video_length": args.target_video_length,
        "approx_duration_s": round((args.target_video_length - 1) / 16.0, 2),
        "parallel_mode": "seq_p" if args.seq_p_size > 1 else "none",
        "seq_p_size": int(args.seq_p_size),
        "seq_p_attn_type": args.seq_p_attn_type if args.seq_p_size > 1 else None,
        "stripe_full_q": bool(getattr(args, "stripe_full_q", False)),
        "stripe_partial_out": bool(getattr(args, "stripe_partial_out", False)),
        "stripe_q_exchange": bool(
            getattr(args, "stripe_q_exchange", False) or args.seq_p_attn_type == "stripe_b2"
        ),
        "stripe_hier": bool(
            getattr(args, "stripe_hier", False) or args.seq_p_attn_type == "stripe_hier"
        ),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "warmup_iters": args.warmup,
        "measure_iters": args.measure_iters,
        "include_rerun": include_rerun,
        "num_chunks": all_runs[0]["num_chunks"] if all_runs else None,
        "infer_steps_per_chunk": all_runs[0]["infer_steps_per_chunk"] if all_runs else None,
        "tokens_per_chunk": all_runs[0]["tokens_per_chunk"] if all_runs else None,
        "total_wall_s_mean": round(sum(r["total_wall_s"] for r in all_runs) / len(all_runs), 3) if all_runs else None,
        "chunk_summary": summary_chunks,
        "runs": all_runs,
        "config_json": args.config_json,
    }
    skew_rec = get_stripe_gather_skew()
    if skew_rec is not None:
        result["gather_skew"] = skew_rec.summary()

    if is_main_process():
        print("\n=== chunk summary (mean) ===")
        for row in summary_chunks:
            extra = ""
            if "stripe_comm_ms_mean" in row:
                extra = f" comm={row['stripe_comm_ms_mean']:.0f}ms flash={row['stripe_flash_ms_mean']:.0f}ms"
            elif "ulysses_comm_ms_mean" in row:
                extra = (
                    f" a2a={row['ulysses_total_comm_ms_mean']:.0f}ms "
                    f"(q+out={row['ulysses_comm_ms_mean']:.0f}, kv={row.get('sp_kv_a2a_ms_mean', 0):.0f}) "
                    f"({row.get('ulysses_comm_pct_of_wall', 0):.0f}%wall) "
                    f"flash={row['ulysses_flash_ms_mean']:.0f}ms"
                )
            elif "self_attn_ms_mean" in row:
                extra = f" self_attn={row['self_attn_ms_mean']:.0f}ms"
            print(
                f"chunk {row['chunk']:2d} K={row['global_kv_tokens']:5d}  "
                f"wall={row['chunk_wall_ms_mean']:7.1f}ms ({row['chunk_wall_ms_per_forward_mean']:.1f}ms/fwd){extra}"
            )
        print(f"total_wall_s_mean={result['total_wall_s_mean']}")
        if result.get("gather_skew"):
            print("\n=== Q all_gather arrival skew ===")
            print(json.dumps(result["gather_skew"], indent=2))
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
