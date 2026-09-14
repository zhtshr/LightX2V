#!/usr/bin/env python3
"""Pin down why stripe Q all_gather is ~5ms E2E vs ~1.5ms aligned idle.

Same process group, CVD must skip physical GPU3 (use 0,1,2,4 for P=4).
Barriers appear only in probe arms / contemporaneous check — not product path.

Verdict logic
-------------
  contemporaneous: right after a real E2E gather, barrier + probe gather.
    actual~5 & probe~1.5  →  slowdown is *inside that gather call* (wait/skew/async),
                               not chronically slow NCCL after model load.
    actual~5 & probe~5    →  NCCL itself is slow in that process/memory state;
                               then check idle_aligned vs after_model_aligned.
  sandwich_flash (no barrier) vs sandwich_flash_aligned:
    if only no-barrier rises → entry skew / unfinished peers after prior compute.
  idle vs after_model / after_hbm:
    if after_model alone ~5 → load/HBM/topology state, not sandwich.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flash_attn.flash_attn_interface as _fa_iface

from lightx2v.common.kvcache.base import BaseKVCachePool
from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_transformer, set_config
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import seed_all
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

Ql, H, D = 1170, 12, 128
flash_attn = True  # capability flag; use _fa_iface for calls


def _assert_no_gpu3(cvd: str) -> None:
    ids = {x.strip() for x in cvd.split(",") if x.strip()}
    if "3" in ids:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={cvd!r} includes physical GPU3")


def _mean_ms(samples: list[float]) -> float:
    return float(sum(samples) / max(len(samples), 1))


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _allgather_stats(local_samples: list[float], group) -> dict[str, Any]:
    """Reduce per-rank mean to max-rank; also exchange full sample vectors for skew."""
    device = torch.cuda.current_device()
    local_mean = _mean_ms(local_samples)
    t = torch.tensor([local_mean, float(len(local_samples))], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    # per-rank means
    means = torch.tensor([local_mean], device=device, dtype=torch.float64)
    gathered = [torch.empty_like(means) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, means, group=group)
    by_rank = [float(x.item()) for x in gathered]
    return {
        "ms_by_rank": [round(x, 3) for x in by_rank],
        "ms_maxrank": round(max(by_rank), 3),
        "ms_minrank": round(min(by_rank), 3),
        "ms_spread": round(max(by_rank) - min(by_rank), 3),
        "n": int(t[1].item()),
    }


def _time_gather_only(qf: torch.Tensor, q: torch.Tensor, group) -> float:
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    dist.all_gather_into_tensor(qf, q, group=group)
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e))


def _arm_aligned(
    name: str,
    qf: torch.Tensor,
    q: torch.Tensor,
    group,
    *,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        dist.all_gather_into_tensor(qf, q, group=group)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    samples = [_time_gather_only(qf, q, group) for _ in range(iters)]
    out = _allgather_stats(samples, group)
    out["arm"] = name
    return out


def _arm_sandwich(
    name: str,
    prep,
    qf: torch.Tensor,
    q: torch.Tensor,
    group,
    *,
    align: bool,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        prep()
        if align:
            dist.barrier(group=group)
        dist.all_gather_into_tensor(qf, q, group=group)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    samples: list[float] = []
    for _ in range(iters):
        prep()
        torch.cuda.synchronize()
        if align:
            dist.barrier(group=group)
            torch.cuda.synchronize()
        samples.append(_time_gather_only(qf, q, group))
    out = _allgather_stats(samples, group)
    out["arm"] = name
    out["align_before_gather"] = align
    return out


def _reduce_dict_max(d: dict[str, float], group) -> dict[str, float]:
    keys = sorted(d.keys())
    t = torch.tensor([d[k] for k in keys], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return {k: float(t[i]) for i, k in enumerate(keys)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--probe_calls", type=int, default=60, help="E2E contemporaneous samples")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_phase1_encoder_inputs.pt")
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument(
        "--output_json",
        default="save_results/optimization_study/sf_q_gather_slowdown_probe.json",
    )
    parser.add_argument("--skip_e2e", action="store_true")
    args = parser.parse_args()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _assert_no_gpu3(cvd)

    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry unavailable")
    if not dist.is_initialized():
        # Prefer platform NCCL options (high-priority stream) — same as product path
        platform_device.init_parallel_env()
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    if ws != 4:
        if rank == 0:
            print(f"expected P=4, got {ws}")
        return 1

    Q = Ql * ws
    q = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    qf = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    dist.all_gather_into_tensor(qf, q, group=group)

    results: dict[str, Any] = {
        "cvd": cvd,
        "world_size": ws,
        "payload_bytes_q_full": int(qf.numel() * qf.element_size()),
        "iters": args.iters,
    }

    # --- A: idle aligned ---
    results["A_idle_aligned"] = _arm_aligned("idle_aligned", qf, q, group, iters=args.iters, warmup=args.warmup)

    # --- B: HBM pressure (approx transformer footprint) ---
    # Allocate ~4 GiB extra to stress allocator/HBM without full model.
    blobs = [torch.empty(256 * 1024 * 1024 // 2, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    for b in blobs:
        b.zero_()
    torch.cuda.synchronize()
    results["B_after_hbm_aligned"] = _arm_aligned(
        "after_hbm_aligned", qf, q, group, iters=args.iters, warmup=args.warmup
    )
    del blobs
    torch.cuda.empty_cache()

    # --- sandwich arms ---
    if flash_attn is None:
        if rank == 0:
            print("flash_attn missing; skip sandwich flash arms")
        results["C_sandwich_flash"] = None
        results["D_sandwich_flash_aligned"] = None
    else:
        Kl = 4680
        k = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        scale = D**-0.5

        def flash_bal() -> None:
            _fa_iface._flash_attn_forward(
                qf.unsqueeze(0).contiguous(),
                k.unsqueeze(0).contiguous(),
                v.unsqueeze(0).contiguous(),
                dropout_p=0.0,
                softmax_scale=scale,
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )

        results["C_sandwich_flash"] = _arm_sandwich(
            "sandwich_flash", flash_bal, qf, q, group, align=False, iters=args.iters, warmup=args.warmup
        )
        results["D_sandwich_flash_aligned"] = _arm_sandwich(
            "sandwich_flash_aligned",
            flash_bal,
            qf,
            q,
            group,
            align=True,
            iters=args.iters,
            warmup=args.warmup,
        )

    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Ql, 1536, device="cuda", dtype=torch.bfloat16)

    def imb_gemm() -> None:
        # rank r burns extra (r) gemms → staggered entry into gather
        for _ in range(rank):
            x.copy_(x @ w)

    results["E_sandwich_imb_gemm"] = _arm_sandwich(
        "sandwich_imb_gemm", imb_gemm, qf, q, group, align=False, iters=args.iters, warmup=args.warmup
    )
    results["F_sandwich_imb_gemm_aligned"] = _arm_sandwich(
        "sandwich_imb_gemm_aligned",
        imb_gemm,
        qf,
        q,
        group,
        align=True,
        iters=args.iters,
        warmup=args.warmup,
    )

    if args.skip_e2e:
        if rank == 0:
            _write(results, args.output_json)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    # --- G/H: load model + contemporaneous probe during real Form-C gathers ---
    seed_all(42)
    config = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.config_json,
    )
    config["target_video_length"] = int(args.target_video_length)
    config["cpu_offload"] = False
    config["parallel"] = {
        "seq_p_size": 4,
        "seq_p_attn_type": "stripe_pe",
        "stripe_partial_out": True,
    }
    os.environ["LIGHTX2V_STRIPE_PARTIAL_OUT"] = "1"
    set_parallel_config(config)

    cache_path = Path(args.inputs_cache)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    latent_shape = [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]
    payload["latent_shape"] = latent_shape
    payload["inputs"]["latent_shape"] = latent_shape

    def _move(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(f"cuda:{rank}")
        if isinstance(obj, dict):
            return {k: _move(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_move(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_move(v) for v in obj)
        return obj

    payload = _move(payload)

    model = load_wan_transformer(config)
    scheduler = load_wan_scheduler(config)
    model.set_scheduler(scheduler)

    # After model load: aligned probe (same buffers)
    results["G_after_model_aligned"] = _arm_aligned(
        "after_model_aligned", qf, q, group, iters=args.iters, warmup=args.warmup
    )

    # Contemporaneous: wrap BaseKVCachePool._stripe_gather_q_full
    actual_ms: list[float] = []
    probe_ms: list[float] = []
    early_vs_late_during: list[float] = []
    state = {"n": 0, "limit": int(args.probe_calls)}
    # mutable holders so wrapped can reallocate
    holders: dict[str, torch.Tensor] = {
        "probe_buf": torch.empty_like(qf),
        "probe_local": torch.empty_like(q),
    }

    orig = BaseKVCachePool._stripe_gather_q_full

    def wrapped(self, q_in, seq_p_group, prof, *, phase: str = "all_gather_q"):
        use_probe = state["n"] < state["limit"]
        if not use_probe:
            return orig(self, q_in, seq_p_group, prof, phase=phase)

        q_local = q_in.contiguous()
        q_len = int(q_local.size(0))
        world_size = dist.get_world_size(seq_p_group)
        needed = (q_len * world_size, *q_local.shape[1:])
        buf = getattr(self, "_stripe_q_gather_buf", None)
        if (
            buf is None
            or buf.shape != needed
            or buf.dtype != q_local.dtype
            or buf.device != q_local.device
        ):
            self._stripe_q_gather_buf = torch.empty(needed, dtype=q_local.dtype, device=q_local.device)
            buf = self._stripe_q_gather_buf

        # Actual E2E gather (no pre-barrier) — mirrors product path
        if prof is not None:
            s_prof, e_prof = prof.mark()
        s0 = torch.cuda.Event(enable_timing=True)
        e0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        dist.all_gather_into_tensor(buf, q_local, group=seq_p_group)
        e0.record()
        if prof is not None:
            prof.end(s_prof, e_prof, phase)
        torch.cuda.synchronize()
        a_ms = float(s0.elapsed_time(e0))

        # Contemporaneous aligned probe (barrier then gather on scratch)
        pl = holders["probe_local"]
        if pl.shape != q_local.shape or pl.dtype != q_local.dtype:
            pl = torch.empty_like(q_local)
            holders["probe_local"] = pl
        pl.copy_(q_local)
        pb = holders["probe_buf"]
        if pb.shape != needed or pb.dtype != q_local.dtype:
            pb = torch.empty(needed, dtype=q_local.dtype, device=q_local.device)
            holders["probe_buf"] = pb
        dist.barrier(group=seq_p_group)
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s1.record()
        dist.all_gather_into_tensor(pb, pl, group=seq_p_group)
        e1.record()
        torch.cuda.synchronize()
        p_ms = float(s1.elapsed_time(e1))

        actual_ms.append(a_ms)
        probe_ms.append(p_ms)

        tdur = torch.tensor([a_ms], device=q_local.device, dtype=torch.float64)
        got = [torch.empty_like(tdur) for _ in range(world_size)]
        dist.all_gather(got, tdur, group=seq_p_group)
        vals = [float(x.item()) for x in got]
        early_vs_late_during.append(max(vals) - min(vals))

        state["n"] += 1
        return buf, [q_len] * world_size

    BaseKVCachePool._stripe_gather_q_full = wrapped  # type: ignore[method-assign]

    inputs = payload["inputs"]
    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(
        model, config, list(payload["latent_shape"])
    )
    scheduler.num_output_frames = num_output_frames
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=int(payload["seed"]), latent_shape=list(latent_shape_adj), image_encoder_output=None)
    kv_mgr = model.kv_cache_manager
    infer_steps = int(scheduler.infer_steps)

    # Run chunk 0 only until probe quota filled (still need a couple steps)
    for step_index in range(min(infer_steps, 4)):
        if state["n"] >= state["limit"]:
            break
        kv_mgr.current_step = step_index
        scheduler.step_pre(seg_index=0, step_index=step_index, is_rerun=False)
        model.infer(inputs)
        scheduler.step_post()

    BaseKVCachePool._stripe_gather_q_full = orig  # type: ignore[method-assign]

    # Summarize contemporaneous
    def _summ(local: list[float]) -> dict[str, Any]:
        if not local:
            return {"n": 0}
        st = sorted(local)
        base = {
            "n_local": len(local),
            "mean": round(_mean_ms(local), 3),
            "p50": round(_pct(st, 0.5), 3),
            "p90": round(_pct(st, 0.9), 3),
            "max": round(max(local), 3),
        }
        stats = _allgather_stats(local, group)
        base.update({"maxrank_mean": stats["ms_maxrank"], "by_rank_mean": stats["ms_by_rank"]})
        return base

    results["H_e2e_contemporaneous"] = {
        "actual_gather": _summ(actual_ms),
        "probe_barrier_then_gather": _summ(probe_ms),
        "actual_skew_ms": _summ(early_vs_late_during),
        "calls_recorded": state["n"],
    }

    # Delta on same calls (local), then max-reduce means
    if actual_ms and probe_ms and len(actual_ms) == len(probe_ms):
        deltas = [a - p for a, p in zip(actual_ms, probe_ms, strict=True)]
        results["H_e2e_contemporaneous"]["actual_minus_probe"] = _summ(deltas)

    idle = results["A_idle_aligned"]["ms_maxrank"]
    after_model = results["G_after_model_aligned"]["ms_maxrank"]
    h = results["H_e2e_contemporaneous"]
    act = h["actual_gather"].get("maxrank_mean", 0.0)
    prb = h["probe_barrier_then_gather"].get("maxrank_mean", 0.0)
    skew = h["actual_skew_ms"].get("mean", 0.0)

    verdict = {
        "idle_aligned_ms": idle,
        "after_model_aligned_ms": after_model,
        "e2e_actual_ms": act,
        "e2e_probe_aligned_ms": prb,
        "e2e_arrival_skew_ms": skew,
    }
    if act >= 3.5 and prb <= idle + 0.8:
        verdict["primary"] = (
            "entry_wait_or_call_path: contemporaneous barrier+gather stays near idle, "
            "so NCCL is not chronically 5ms; the E2E call itself is inflated (wait/skew)."
        )
    elif act >= 3.5 and prb >= 3.5:
        if after_model >= 3.5:
            verdict["primary"] = (
                "chronic_nccl_after_model: even barrier-aligned gather is slow after load; "
                "HBM/topology/NCCL state — not sandwich wait."
            )
        else:
            verdict["primary"] = (
                "in_forward_inflation: aligned gather fast after load, but both actual and "
                "in-forward probe are slow — compute sandwich / transient contention."
            )
    else:
        verdict["primary"] = "inconclusive_or_already_fast"
    # sandwich deltas
    if results.get("C_sandwich_flash") and results.get("D_sandwich_flash_aligned"):
        verdict["flash_no_barrier_ms"] = results["C_sandwich_flash"]["ms_maxrank"]
        verdict["flash_aligned_ms"] = results["D_sandwich_flash_aligned"]["ms_maxrank"]
    verdict["imb_no_barrier_ms"] = results["E_sandwich_imb_gemm"]["ms_maxrank"]
    verdict["imb_aligned_ms"] = results["F_sandwich_imb_gemm_aligned"]["ms_maxrank"]
    results["verdict"] = verdict

    if rank == 0:
        _write(results, args.output_json)
        print(json.dumps({"verdict": verdict, "key_arms": {
            "A_idle": results["A_idle_aligned"]["ms_maxrank"],
            "B_hbm": results["B_after_hbm_aligned"]["ms_maxrank"],
            "G_after_model": results["G_after_model_aligned"]["ms_maxrank"],
            "H_actual": act,
            "H_probe": prb,
        }}, indent=2))

    dist.barrier()
    dist.destroy_process_group()
    return 0


def _write(results: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    raise SystemExit(main())
