#!/usr/bin/env python3
"""Exp-3b: encoder/decoder parallel efficiency under monolithic SP.

Models the monolithic multi-GPU serving pattern:
  - encoder / decoder run on rank 0 only; other SP ranks wait at barriers
  - denoise uses Ulysses SP across all ranks

Reports per-stage wall time and per-GPU utilization so we can show that
enc/dec do not benefit from SP (η ≈ 1/P) while those GPUs stay idle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.examples.wan_i2v import compute_latent_shape_from_image, get_vae_encoder_output
from lightx2v.disagg.utils import (
    load_wan_text_encoder,
    load_wan_transformer,
    load_wan_vae_decoder,
    load_wan_vae_encoder,
    read_image_input,
    set_config,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def _physical_gpu_ids() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return [0]
    return [int(x) for x in visible.split(",") if x.strip() != ""]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _query_utils(physical_ids: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for gid in physical_ids:
        cmd = [
            "nvidia-smi",
            f"-i={gid}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            text = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            out[gid] = float(text.splitlines()[0].strip())
        except Exception:
            out[gid] = 0.0
    return out


class GpuUtilSampler:
    """Background nvidia-smi sampler tagged by stage name."""

    def __init__(self, physical_ids: list[int], interval_s: float = 0.25):
        self.physical_ids = list(physical_ids)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "idle"
        self._lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []

    def start(self, stage: str) -> None:
        self.stop()
        self._stop.clear()
        self._stage = stage
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            utils = _query_utils(self.physical_ids)
            with self._lock:
                self.samples.append(
                    {
                        "stage": self._stage,
                        "ts": time.time(),
                        "util": {str(k): v for k, v in utils.items()},
                    }
                )
            self._stop.wait(self.interval_s)

    def summarize(self) -> dict[str, Any]:
        by_stage: dict[str, list[dict[str, float]]] = {}
        with self._lock:
            for sample in self.samples:
                by_stage.setdefault(sample["stage"], []).append(
                    {int(k): float(v) for k, v in sample["util"].items()}
                )

        summary: dict[str, Any] = {}
        for stage, rows in by_stage.items():
            if not rows:
                continue
            per_gpu: dict[str, Any] = {}
            for gid in self.physical_ids:
                vals = [row.get(gid, 0.0) for row in rows]
                mean_u = sum(vals) / len(vals)
                per_gpu[str(gid)] = {
                    "mean_util_pct": mean_u,
                    "max_util_pct": max(vals),
                    "min_util_pct": min(vals),
                    "n_samples": len(vals),
                    "frac_busy_gt20": sum(1 for v in vals if v > 20.0) / len(vals),
                }
            means = [per_gpu[str(g)]["mean_util_pct"] for g in self.physical_ids]
            busiest = max(means) if means else 0.0
            avg_all = sum(means) / len(means) if means else 0.0
            idleish = sum(1 for m in means if m < 20.0)
            summary[stage] = {
                "per_gpu": per_gpu,
                "mean_util_all_gpus_pct": avg_all,
                "busiest_gpu_mean_util_pct": busiest,
                "idle_gpu_count_mean_lt20": idleish,
                "util_balance_ratio": (avg_all / busiest) if busiest > 1e-6 else 0.0,
                "n_samples": len(rows),
            }
        return summary


def _init_distributed(seq_p_size: int, config: dict[str, Any]) -> None:
    if seq_p_size <= 1:
        config["parallel"] = False
        return
    # vae_parallel must stay False: enc/dec run rank0-only; enabling it deadlocks vs our barriers.
    config["parallel"] = {
        "seq_p_size": seq_p_size,
        "seq_p_attn_type": "ulysses",
        "vae_parallel": False,
    }
    platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _broadcast_tensor(t: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if not dist.is_initialized():
        return t
    has = torch.tensor([0 if t is None else 1], device=device, dtype=torch.int64)
    dist.broadcast(has, src=0)
    if int(has.item()) == 0:
        return None
    # Fixed-size meta: [ndim, d0, d1, d2, d3, d4, d5, dtype_code] — must match on all ranks.
    meta = torch.zeros(8, device=device, dtype=torch.int64)
    if dist.get_rank() == 0:
        assert t is not None
        meta[0] = int(t.ndim)
        for i, d in enumerate(t.shape):
            meta[1 + i] = int(d)
        # 0=bf16, 1=fp16, 2=fp32 (enough for this bench)
        if t.dtype == torch.bfloat16:
            meta[7] = 0
        elif t.dtype == torch.float16:
            meta[7] = 1
        else:
            meta[7] = 2
    dist.broadcast(meta, src=0)
    ndim = int(meta[0].item())
    shape = tuple(int(x) for x in meta[1 : 1 + ndim].tolist())
    dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[int(meta[7].item())]
    if dist.get_rank() != 0:
        t = torch.empty(shape, device=device, dtype=dtype)
    assert t is not None
    dist.broadcast(t.contiguous(), src=0)
    return t


def _run_encoder(
    *,
    config: dict[str, Any],
    text_encoder: Any,
    vae_encoder: Any,
    prompt: str,
    image_path: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[int], float]:
    text_len = int(config.get("text_len", 512))
    _sync()
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()

    if is_main_process():
        context = text_encoder.infer([prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
        img, _ = read_image_input(image_path)
        latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)
        vae_out = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)
        context = context.to(device=device, dtype=GET_DTYPE())
        vae_out = vae_out.to(device=device, dtype=GET_DTYPE())
    else:
        context = None
        vae_out = None
        latent_shape = [0, 0, 0, 0]

    if dist.is_initialized():
        shape_t = torch.tensor(latent_shape if is_main_process() else [0, 0, 0, 0], device=device, dtype=torch.int64)
        dist.broadcast(shape_t, src=0)
        latent_shape = [int(x) for x in shape_t.tolist()]
        context = _broadcast_tensor(context, device)
        vae_out = _broadcast_tensor(vae_out, device)
        dist.barrier()

    _sync()
    elapsed = time.perf_counter() - start
    inputs = {
        "text_encoder_output": {"context": context, "context_null": None},
        "image_encoder_output": {"clip_encoder_out": None, "vae_encoder_out": vae_out},
    }
    return inputs, latent_shape, elapsed


def _run_denoise(
    *,
    scheduler: WanScheduler,
    model: Any,
    inputs: dict[str, Any],
    seed: int,
    latent_shape: list[int],
    image_encoder_output: dict[str, Any] | None,
) -> tuple[torch.Tensor, float]:
    _sync()
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)
    for step_index in range(scheduler.infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()
    _sync()
    if dist.is_initialized():
        dist.barrier()
    elapsed = time.perf_counter() - start
    return scheduler.latents, elapsed


def _run_decoder(*, vae_decoder: Any, latents: torch.Tensor) -> float:
    _sync()
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    if is_main_process():
        gen_video = vae_decoder.decode(latents.to(GET_DTYPE()))
        _ = wan_vae_to_comfy(gen_video)
    if dist.is_initialized():
        dist.barrier()
    _sync()
    return time.perf_counter() - start


def _efficiency(t_p1: float | None, t_p: float, p: int) -> float | None:
    if t_p1 is None or t_p <= 0 or p <= 0:
        return None
    return t_p1 / (t_p * p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_baseline.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    parser.add_argument(
        "--prompt",
        default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_p_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="Warmup full E2E passes before measure")
    parser.add_argument("--sample_interval_s", type=float, default=0.25)
    parser.add_argument(
        "--baseline_json",
        default="",
        help="Optional P=1 result json for efficiency ratios when this run is P>1",
    )
    parser.add_argument(
        "--output_json",
        default="/root/zht/LightX2V/save_results/optimization_study/p3_encdec_sp_efficiency.json",
    )
    args = parser.parse_args()

    physical_ids = _physical_gpu_ids()
    if len(physical_ids) != args.seq_p_size and args.seq_p_size > 1:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES has {len(physical_ids)} ids {physical_ids}, "
            f"but --seq_p_size={args.seq_p_size}"
        )
    # Safety: never touch physical GPU 1 or 3.
    for gid in physical_ids:
        if gid in (1, 3):
            raise RuntimeError(f"Refusing to use unreliable physical GPU {gid}; got {physical_ids}")

    config = set_config(
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=args.config_json,
    )
    seed_all(args.seed)
    _init_distributed(args.seq_p_size, config)

    device = torch.device(f"{AI_DEVICE}:{dist.get_rank()}" if dist.is_initialized() else AI_DEVICE)
    # Only rank0 samples nvidia-smi (avoids multi-rank query contention).
    sampler = GpuUtilSampler(physical_ids, interval_s=args.sample_interval_s) if is_main_process() else None

    if is_main_process():
        print(
            f"[Exp-3b] seq_p={args.seq_p_size} physical_gpus={physical_ids} "
            f"cpu_offload={config.get('cpu_offload')} offload_granularity={config.get('offload_granularity')} "
            f"vae_parallel={False if args.seq_p_size > 1 else 'n/a'}"
        )

    load_t0 = time.perf_counter()
    text_encoder = load_wan_text_encoder(config)[0] if is_main_process() else None
    vae_encoder = load_wan_vae_encoder(config) if is_main_process() else None
    vae_decoder = load_wan_vae_decoder(config) if is_main_process() else None
    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)
    if dist.is_initialized():
        dist.barrier()
    model_load_s = time.perf_counter() - load_t0
    if is_main_process():
        print(f"[Exp-3b] model_load_s={model_load_s:.2f} (excluded from stage metrics)")

    def _one_pass(measure: bool) -> dict[str, float]:
        if measure and sampler is not None:
            sampler.start("encoder")
        inputs, latent_shape, enc_s = _run_encoder(
            config=config,
            text_encoder=text_encoder,
            vae_encoder=vae_encoder,
            prompt=args.prompt,
            image_path=args.image_path,
            device=device,
        )
        if measure and sampler is not None:
            sampler.stop()
            sampler.start("denoise")
        latents, denoise_s = _run_denoise(
            scheduler=scheduler,
            model=model,
            inputs=inputs,
            seed=args.seed,
            latent_shape=latent_shape,
            image_encoder_output=inputs["image_encoder_output"],
        )
        if measure and sampler is not None:
            sampler.stop()
            sampler.start("decoder")
        dec_s = _run_decoder(vae_decoder=vae_decoder, latents=latents)
        if measure and sampler is not None:
            sampler.stop()
        return {"encoder_s": enc_s, "denoise_s": denoise_s, "decoder_s": dec_s}

    for i in range(max(0, args.warmup)):
        if is_main_process():
            print(f"[Exp-3b] warmup {i + 1}/{args.warmup}")
        _one_pass(measure=False)

    if is_main_process():
        print("[Exp-3b] measuring one E2E pass with GPU util sampling...")
    times = _one_pass(measure=True)
    util_summary = sampler.summarize() if sampler is not None else {}

    baseline = None
    if args.baseline_json:
        bpath = Path(args.baseline_json)
        if bpath.is_file():
            baseline = json.loads(bpath.read_text(encoding="utf-8"))

    p = int(args.seq_p_size)
    t_enc = times["encoder_s"]
    t_dec = times["decoder_s"]
    t_den = times["denoise_s"]
    t_e2e = t_enc + t_den + t_dec

    b_stages = (baseline or {}).get("stages_s", {}) if isinstance(baseline, dict) else {}
    result = {
        "metric": "encdec_sp_efficiency",
        "description": (
            "Monolithic SP: enc/dec on rank0 only + barrier; denoise Ulysses SP. "
            "Shows enc/dec wall time does not scale and non-rank0 GPUs stay idle."
        ),
        "seq_p_size": p,
        "physical_gpus": physical_ids,
        "cpu_offload": bool(config.get("cpu_offload", False)),
        "offload_granularity": config.get("offload_granularity"),
        "target_hw": [config.get("target_height"), config.get("target_width")],
        "infer_steps": config.get("infer_steps"),
        "model_load_s_excluded": model_load_s,
        "stages_s": {
            "encoder": t_enc,
            "denoise": t_den,
            "decoder": t_dec,
            "e2e_sum": t_e2e,
        },
        "stage_fraction": {
            "encoder": t_enc / t_e2e if t_e2e > 0 else None,
            "denoise": t_den / t_e2e if t_e2e > 0 else None,
            "decoder": t_dec / t_e2e if t_e2e > 0 else None,
        },
        "gpu_util_by_stage": util_summary,
        "parallel_efficiency_vs_p1": {
            "encoder": _efficiency(b_stages.get("encoder"), t_enc, p),
            "decoder": _efficiency(b_stages.get("decoder"), t_dec, p),
            "denoise": _efficiency(b_stages.get("denoise"), t_den, p),
            "e2e_sum": _efficiency(b_stages.get("e2e_sum"), t_e2e, p),
            "ideal_if_encdec_rank0_only": 1.0 / p,
        },
    "narrative": {
        "enc_dec_should_not_scale": "encoder/decoder η near 1/P means SP GPUs buy no enc/dec speedup",
        "idle_gpus_during_enc_dec": (
            "non-rank0 ranks wait on barrier/broadcast; wall time unchanged. "
            "nvidia-smi util can stay high due to NCCL wait kernels — use η not util"
        ),
        "denoise_does_use_sp": "denoise η ≫ 1/P and util_balance_ratio high (all ranks do real SP work)",
    },
}

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["stages_s"], indent=2))
        print(f"[Exp-3b] wrote {out}")
        for stage in ("encoder", "denoise", "decoder"):
            st = util_summary.get(stage, {})
            print(
                f"  {stage}: wall={result['stages_s'][stage]:.2f}s "
                f"mean_util={st.get('mean_util_all_gpus_pct', 0):.1f}% "
                f"busiest={st.get('busiest_gpu_mean_util_pct', 0):.1f}% "
                f"idle_lt20={st.get('idle_gpu_count_mean_lt20', '?')} "
                f"balance={st.get('util_balance_ratio', 0):.2f}"
            )
            if p > 1 and result["parallel_efficiency_vs_p1"].get(stage) is not None:
                print(f"    η vs P=1 = {result['parallel_efficiency_vs_p1'][stage]:.3f} (ideal if rank0-only ≈ {1.0 / p:.3f})")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
