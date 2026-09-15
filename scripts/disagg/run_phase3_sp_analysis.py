#!/usr/bin/env python3
"""Phase 3: seq-parallel scaling efficiency, comm/compute split, GPU util during denoise."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1_module():
    bench_path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1_bench", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {bench_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GpuUtilSampler:
    """Poll nvidia-smi GPU util for selected device indices."""

    def __init__(self, gpu_ids: list[int], interval_s: float = 0.25):
        self.gpu_ids = gpu_ids
        self.interval_s = interval_s
        self.samples: list[dict[int, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> dict[int, float]:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        out: dict[int, float] = {}
        if proc.returncode != 0:
            return out
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            idx, util = int(parts[0]), float(parts[1])
            if idx in self.gpu_ids:
                out[idx] = util
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._poll()
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuUtilSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summarize(self) -> dict[str, Any]:
        per_gpu: dict[int, list[float]] = {gid: [] for gid in self.gpu_ids}
        pooled: list[float] = []
        for sample in self.samples:
            for gid in self.gpu_ids:
                if gid in sample:
                    per_gpu[gid].append(sample[gid])
            if sample:
                pooled.append(sum(sample.values()) / len(sample))
        per_gpu_stats = {}
        for gid, vals in per_gpu.items():
            if not vals:
                continue
            per_gpu_stats[str(gid)] = {
                "avg": statistics.mean(vals),
                "max": max(vals),
                "p50": statistics.median(vals),
                "samples": len(vals),
            }
        pooled_stats = {}
        if pooled:
            pooled_stats = {
                "avg": statistics.mean(pooled),
                "max": max(pooled),
                "p50": statistics.median(pooled),
                "samples": len(pooled),
            }
        return {"per_gpu": per_gpu_stats, "pooled_active_gpus": pooled_stats}


_COMM_KEYWORDS = (
    "nccl",
    "all_to_all",
    "alltoall",
    "dist.all",
    "broadcast",
    "reduce_scatter",
    "all_gather",
    "barrier",
    "isend",
    "irecv",
    "send_recv",
    "p2p",
)


def _is_comm_event(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in _COMM_KEYWORDS)


def _profile_one_step(scheduler: WanScheduler, model: Any, payload: dict[str, Any]) -> dict[str, float]:
    seed = int(payload["seed"])
    latent_shape = payload["latent_shape"]
    image_encoder_output = payload["image_encoder_output"]
    inputs = payload["inputs"]

    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    scheduler.step_pre(step_index=0)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof:
        model.infer(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scheduler.step_post()

    comm_cuda_us = 0.0
    compute_cuda_us = 0.0
    comm_cpu_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type.name == "CUDA":
            if _is_comm_event(evt.key):
                comm_cuda_us += evt.device_time_total
            else:
                compute_cuda_us += evt.device_time_total
        elif evt.device_type.name == "CPU" and _is_comm_event(evt.key):
            comm_cpu_us += evt.self_cpu_time_total

    total_cuda = comm_cuda_us + compute_cuda_us
    return {
        "comm_cuda_us": comm_cuda_us,
        "compute_cuda_us": compute_cuda_us,
        "comm_cpu_us": comm_cpu_us,
        "comm_cuda_ratio": (comm_cuda_us / total_cuda) if total_cuda > 0 else 0.0,
        "profiled_step_index": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_p_size", type=int, required=True)
    parser.add_argument("--config_json", default="/root/zht/LightX2V/save_results/optimization_study/baseline_seqp1.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v", choices=("i2v", "t2v"))
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument(
        "--negative_prompt",
        default="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )
    parser.add_argument("--phase1_json", default="", help="Optional phase1 measure json to reuse transformer_compute_s")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    p1 = _load_phase1_module()
    cfg_path = Path(args.config_json)
    if not cfg_path.is_file():
        cfg_path = Path(
            f"/root/zht/LightX2V/save_results/optimization_study/baseline_seqp{args.seq_p_size}.json",
        )
    ns = argparse.Namespace(
        config_json=str(cfg_path),
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
            if args.task == "t2v"
            else "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
        ),
        seed=42,
        seq_p_size=args.seq_p_size,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt=args.negative_prompt,
    )

    config = p1._load_config(ns)
    seed_all(42)
    p1._init_distributed(config)

    payload = p1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=ns.prompt,
        image_path=ns.image_path,
        seed=42,
        force=False,
        task=args.task,
        negative_prompt=args.negative_prompt,
    )
    payload = p1._prepare_payload_on_device(payload)

    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)

    gpu_ids = list(range(args.seq_p_size))
    for _ in range(args.warmup):
        p1._bench_transformer_once(scheduler, model, payload)

    gpu_util: dict[str, Any] = {}
    sampler: GpuUtilSampler | None = None
    if is_main_process():
        sampler = GpuUtilSampler(gpu_ids)
        sampler.__enter__()
    try:
        transformer_compute_s = p1._bench_transformer_once(scheduler, model, payload)
    finally:
        if sampler is not None:
            sampler.__exit__(None, None, None)
            gpu_util = sampler.summarize()

    if dist.is_initialized():
        dist.barrier()

    step_profile: dict[str, float] = {}
    profile_result = _profile_one_step(scheduler, model, payload)
    if is_main_process():
        step_profile = profile_result

    if dist.is_initialized():
        dist.barrier()

    if args.phase1_json and Path(args.phase1_json).is_file() and transformer_compute_s is None:
        transformer_compute_s = json.loads(Path(args.phase1_json).read_text()).get("transformer_compute_s")

    phase1_path = Path(f"/root/zht/LightX2V/save_results/optimization_study/p1_transformer_seqp{args.seq_p_size}.json")
    if transformer_compute_s is None and phase1_path.is_file():
        transformer_compute_s = json.loads(phase1_path.read_text()).get("transformer_compute_s")

    attn_type = config.get("parallel", {}).get("seq_p_attn_type") if config.get("parallel") else None
    result = {
        "seq_p_size": args.seq_p_size,
        "seq_p_attn_type": attn_type,
        "transformer_compute_s": transformer_compute_s,
        "gpu_util_during_denoise": gpu_util,
        "one_step_profile": step_profile,
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
