#!/usr/bin/env python3
"""Measure TP=2 GPU memory for single vs dual-request (dual-tenant prep)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _mem_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    peak = torch.cuda.max_memory_allocated(idx) / (1024**3)
    free, total = torch.cuda.mem_get_info(idx)
    return {
        "allocated_gb": alloc,
        "reserved_gb": reserved,
        "peak_allocated_gb": peak,
        "free_gb": free / (1024**3),
        "total_gb": total / (1024**3),
    }


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _gather_rank_mem(local: dict[str, float]) -> dict[str, Any]:
    if not dist.is_initialized():
        return {"rank": 0, **local}
    world = dist.get_world_size()
    payload = [local]
    gathered: list[dict[str, float] | None] = [None] * world
    dist.all_gather_object(gathered, payload[0])
    rank = dist.get_rank()
    out: dict[str, Any] = {"rank": rank, **local, "all_ranks": gathered}
    return out


def _run_single_denoise(model, scheduler, payload) -> None:
    seed = int(payload["seed"])
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]
    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    for step_index in range(scheduler.infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()


def _run_one_step(model, scheduler, payload, step_index: int = 0) -> None:
    seed = int(payload["seed"])
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]
    if step_index == 0:
        scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    scheduler.step_pre(step_index=step_index)
    model.infer(inputs)
    scheduler.step_post()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_json",
        default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_tp_fair_bench.json",
    )
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--tensor_p_size", type=int, default=2)
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--output_json", default="/root/zht/LightX2V/save_results/optimization_study/wan22_tp2_memory_profile.json")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    p1 = _load_phase1()
    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        seed=42,
        seq_p_size=1,
        tensor_p_size=args.tensor_p_size,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )

    config = p1._load_config(ns)
    seed_all(42)
    if args.tensor_p_size > 1:
        p1._init_distributed(config)

    payload = p1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=ns.prompt,
        image_path=ns.image_path,
        seed=42,
        force=False,
        task="i2v",
        negative_prompt=ns.negative_prompt,
    )
    payload = p1._prepare_payload_on_device(payload)
    payload_b = {
        "seed": 43,
        "latent_shape": payload["latent_shape"],
        "image_encoder_output": payload["image_encoder_output"],
        "inputs": payload["inputs"],
    }

    milestones: dict[str, Any] = {}

    _reset_peak()
    model = load_wan_transformer(config)
    _sync()
    milestones["after_model_load"] = _gather_rank_mem(_mem_gb())

    scheduler_a = WanScheduler(config)
    scheduler_b = WanScheduler(config)
    model.set_scheduler(scheduler_a)

    for _ in range(args.warmup):
        _run_single_denoise(model, scheduler_a, payload)

    # --- single request full denoise ---
    _reset_peak()
    scheduler_a = WanScheduler(config)
    model.set_scheduler(scheduler_a)
    _run_single_denoise(model, scheduler_a, payload)
    _sync()
    milestones["single_request_full_denoise_peak"] = _gather_rank_mem(_mem_gb())

    # --- single request one step peak ---
    _reset_peak()
    scheduler_a = WanScheduler(config)
    model.set_scheduler(scheduler_a)
    _run_one_step(model, scheduler_a, payload, step_index=0)
    _sync()
    milestones["single_request_one_step_peak"] = _gather_rank_mem(_mem_gb())

    if args.tensor_p_size > 1:
        # --- dual prepare only ---
        _reset_peak()
        scheduler_a = WanScheduler(config)
        scheduler_b = WanScheduler(config)
        scheduler_a.prepare(
            seed=int(payload["seed"]),
            latent_shape=payload["latent_shape"],
            image_encoder_output=payload["image_encoder_output"],
        )
        scheduler_b.prepare(
            seed=int(payload_b["seed"]),
            latent_shape=payload_b["latent_shape"],
            image_encoder_output=payload_b["image_encoder_output"],
        )
        _sync()
        milestones["dual_request_prepare_only_peak"] = _gather_rank_mem(_mem_gb())

        _reset_peak()
        scheduler_a = WanScheduler(config)
        scheduler_b = WanScheduler(config)
        model.set_scheduler(scheduler_a)
        _run_single_denoise(model, scheduler_a, payload)
        model.set_scheduler(scheduler_b)
        _run_single_denoise(model, scheduler_b, payload_b)
        _sync()
        milestones["dual_request_serial_b2b_peak"] = _gather_rank_mem(_mem_gb())

        _reset_peak()
        scheduler_a = WanScheduler(config)
        scheduler_b = WanScheduler(config)
        scheduler_a.prepare(
            seed=int(payload["seed"]),
            latent_shape=payload["latent_shape"],
            image_encoder_output=payload["image_encoder_output"],
        )
        scheduler_b.prepare(
            seed=int(payload_b["seed"]),
            latent_shape=payload_b["latent_shape"],
            image_encoder_output=payload_b["image_encoder_output"],
        )
        model.set_scheduler(scheduler_a)
        _run_one_step(model, scheduler_a, payload, step_index=0)
        model.set_scheduler(scheduler_b)
        _run_one_step(model, scheduler_b, payload_b, step_index=0)
        _sync()
        milestones["dual_request_one_step_b2b_with_both_latents_peak"] = _gather_rank_mem(_mem_gb())

    def _max_peak(key: str) -> float:
        m = milestones[key]
        ranks = m.get("all_ranks") or [m]
        return max(r["peak_allocated_gb"] for r in ranks if r)

    gpu_total = milestones["after_model_load"]["total_gb"]
    single_peak = _max_peak("single_request_full_denoise_peak")
    load_peak = _max_peak("after_model_load")
    dual_prepare_peak = (
        _max_peak("dual_request_prepare_only_peak") if args.tensor_p_size > 1 else load_peak
    )
    dual_b2b_peak = _max_peak("dual_request_serial_b2b_peak") if args.tensor_p_size > 1 else single_peak
    one_step_peak = _max_peak("single_request_one_step_peak")
    dual_step_proxy = (
        _max_peak("dual_request_one_step_b2b_with_both_latents_peak")
        if args.tensor_p_size > 1
        else single_peak
    )

    # Activation delta estimates
    act_full = single_peak - load_peak
    act_one_step = one_step_peak - load_peak
    latent_dual_delta = dual_prepare_peak - load_peak

    # Overlap worst case: weights + 2 * per-step activation peak (two tenants mid-forward)
    overlap_worst = load_peak + 2 * act_one_step
    # Overlap optimistic: weights + dual latents + 1x step activation + comm scratch
    comm_scratch_gb = 0.315 * 2  # one AR payload fp16 ~315MB, reserve 2 buffers
    overlap_optimistic = dual_prepare_peak + act_one_step + comm_scratch_gb

    result = {
        "tensor_p_size": args.tensor_p_size,
        "config_json": args.config_json,
        "gpu_total_gb_per_rank": gpu_total,
        "milestones_gb": milestones,
        "summary": {
            "weights_and_static_gb": round(load_peak, 3),
            "single_request_peak_gb": round(single_peak, 3),
            "single_request_headroom_gb": round(gpu_total - single_peak, 3),
            "activation_full_denoise_delta_gb": round(act_full, 3),
            "activation_one_step_delta_gb": round(act_one_step, 3),
            "dual_latents_extra_gb": round(latent_dual_delta, 3),
            "dual_serial_b2b_peak_gb": round(dual_b2b_peak, 3),
            "dual_one_step_proxy_peak_gb": round(dual_step_proxy, 3),
            "dual_overlap_worst_case_gb": round(overlap_worst, 3),
            "dual_overlap_optimistic_gb": round(overlap_optimistic, 3),
            "dual_overlap_worst_headroom_gb": round(gpu_total - overlap_worst, 3),
            "dual_overlap_optimistic_headroom_gb": round(gpu_total - overlap_optimistic, 3),
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
