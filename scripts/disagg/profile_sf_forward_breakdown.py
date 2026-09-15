#!/usr/bin/env python3
"""Profile SF wan2.1_sf single-forward and full-run compute vs overhead breakdown."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.utils.utils import seed_all

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)

KERNEL_CATEGORIES = (
    ("gemm", re.compile(r"cublas|cutlass|gemm|_mm_|matmul|linear", re.I)),
    ("flash_attn", re.compile(r"flash|fmha|attention.*fwd|scaled_dot", re.I)),
    ("rope", re.compile(r"rope|rotary|triton.*causal", re.I)),
    ("norm_modulate", re.compile(r"norm|layer_norm|rms|modulate|scale_shift", re.I)),
    ("kv_memory", re.compile(r"copy|cat|slice|index|gather|scatter|memcpy|contiguous", re.I)),
    ("elementwise", re.compile(r"elementwise|mul|add|silu|gelu|softmax", re.I)),
)


def _event_cuda_ms(event) -> float:
    for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total", "self_device_time_total"):
        val = getattr(event, attr, None)
        if val:
            return float(val) / 1000.0
    return 0.0


def _categorize_kernel(name: str) -> str:
    for cat, pat in KERNEL_CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


def _latent_shape(cfg: dict[str, Any]) -> list[int]:
    vae_stride = cfg["vae_stride"]
    return [
        16,
        (cfg["target_video_length"] - 1) // vae_stride[0] + 1,
        cfg["target_height"] // vae_stride[1],
        cfg["target_width"] // vae_stride[2],
    ]


def _prepare_inputs(cfg: dict[str, Any], prompt: str) -> dict[str, Any]:
    text_encoder = load_wan_text_encoder(cfg)[0]
    text_len = int(cfg.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    return {
        "text_encoder_output": {"context": context.cuda(), "context_null": None},
        "image_encoder_output": None,
    }


def _sync() -> None:
    torch.cuda.synchronize()


def _record_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.record()
    _sync()
    return float(start.elapsed_time(end))


class PhaseTimer:
    def __init__(self) -> None:
        self.times_ms: dict[str, float] = defaultdict(float)

    def wrap_pre_infer(self, model: Any, inputs: dict[str, Any]) -> Any:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = model.pre_infer.infer(model.pre_weight, inputs)
        self.times_ms["pre_infer"] += _record_elapsed(s, e)
        return out

    def wrap_transformer(self, model: Any, pre_infer_out: Any) -> Any:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        x = model.transformer_infer.infer(model.transformer_weights, pre_infer_out)
        self.times_ms["transformer"] += _record_elapsed(s, e)
        return x

    def wrap_post_infer(self, model: Any, x: Any, pre_infer_out: Any) -> Any:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = model.post_infer.infer(x, pre_infer_out)
        self.times_ms["post_infer"] += _record_elapsed(s, e)
        return out


def _patch_model_for_phase_timing(model: Any, timer: PhaseTimer) -> None:
    orig = model._infer_cond_uncond.__func__ if hasattr(model._infer_cond_uncond, "__func__") else None

    def timed_infer(self, inputs, infer_condition=True):
        pre_infer_out = timer.wrap_pre_infer(self, inputs)
        x = timer.wrap_transformer(self, pre_infer_out)
        noise_pred = timer.wrap_post_infer(self, x, pre_infer_out)[0]
        return noise_pred

    import types

    model._infer_cond_uncond = types.MethodType(timed_infer, model)


def _warmup_to(model, scheduler, cfg, inputs, latent_shape, seg_idx: int, step_index: int) -> None:
    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
    scheduler.num_output_frames = num_out
    scheduler.prepare(seed=42, latent_shape=list(ls_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    try:
        for s in range(seg_idx):
            for st in range(infer_steps):
                model.kv_cache_manager.current_step = st
                scheduler.step_pre(seg_index=s, step_index=st, is_rerun=False)
                model.infer(inputs)
                scheduler.step_post()
            scheduler.step_pre(seg_index=s, step_index=infer_steps - 1, is_rerun=True)
            model.infer(inputs)
        for st in range(step_index):
            model.kv_cache_manager.current_step = st
            scheduler.step_pre(seg_index=seg_idx, step_index=st, is_rerun=False)
            model.infer(inputs)
            scheduler.step_post()
    finally:
        pass
    return ls_adj, num_out, num_chunks


def profile_single_forward(model, scheduler, cfg, inputs, latent_shape, seg_idx: int, step_index: int) -> dict[str, Any]:
    DisaggSFKVCacheManager.teardown(model)
    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    ls_adj, num_out, num_chunks = _warmup_to(model, scheduler, cfg, inputs, latent_shape, seg_idx, step_index)
    scheduler.num_output_frames = num_out
    scheduler.prepare(seed=42, latent_shape=list(ls_adj), image_encoder_output=None)

    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    q_len = 3 * model.kv_cache_manager.frame_seq_length
    k_len = (seg_idx + 1) * q_len

    phase_timer = PhaseTimer()
    _patch_model_for_phase_timing(model, phase_timer)

    model.kv_cache_manager.current_step = step_index
    scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)

    _sync()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_flops=True,
    ) as prof:
        model.infer(inputs)
    _sync()
    scheduler.step_post()

    kv_prof = ti.finalize_kv_store_profile() if hasattr(ti, "finalize_kv_store_profile") else {}
    wall_ms = sum(phase_timer.times_ms.values())

    cat_ms: dict[str, float] = defaultdict(float)
    total_kernel_ms = 0.0
    total_flops = 0
    top_kernels: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        ms = _event_cuda_ms(evt)
        if ms <= 0:
            continue
        total_kernel_ms += ms
        cat_ms[_categorize_kernel(evt.key)] += ms
        if evt.flops:
            total_flops += int(evt.flops)
        top_kernels.append({"name": evt.key, "cuda_ms": round(ms, 3)})
    top_kernels.sort(key=lambda x: -x["cuda_ms"])
    top_kernels = top_kernels[:12]

    sa = float(kv_prof.get("self_attn_ms", 0))
    ca = float(kv_prof.get("cross_attn_ms", 0))
    store = float(kv_prof.get("store_kv_ms", 0))
    read = float(kv_prof.get("kv_read_ms", 0))
    cross_store = float(kv_prof.get("cross_kv_store_ms", 0))
    transformer_ms = phase_timer.times_ms["transformer"]
    attn_total = sa + ca
    ffn_other_ms = max(0.0, transformer_ms - attn_total)

    return {
        "seg_index": seg_idx,
        "step_index": step_index,
        "q_tokens": q_len,
        "k_tokens": k_len,
        "phase_ms": dict(phase_timer.times_ms),
        "phase_pct": {k: round(100 * v / wall_ms, 2) for k, v in phase_timer.times_ms.items()},
        "transformer_breakdown_ms": {
            "self_attn": round(sa, 3),
            "cross_attn": round(ca, 3),
            "ffn_and_block_overhead": round(ffn_other_ms, 3),
            "self_attn_store_kv": round(store, 3),
            "self_attn_kv_read": round(read, 3),
            "cross_kv_store": round(cross_store, 3),
        },
        "transformer_breakdown_pct": {
            "self_attn": round(100 * sa / transformer_ms, 2) if transformer_ms else 0,
            "cross_attn": round(100 * ca / transformer_ms, 2) if transformer_ms else 0,
            "ffn_and_block_overhead": round(100 * ffn_other_ms / transformer_ms, 2) if transformer_ms else 0,
            "kv_io_of_transformer": round(100 * (store + read + cross_store) / transformer_ms, 2) if transformer_ms else 0,
        },
        "kernel_category_ms": {k: round(v, 3) for k, v in sorted(cat_ms.items(), key=lambda x: -x[1])},
        "kernel_category_pct": {
            k: round(100 * v / total_kernel_ms, 2) for k, v in sorted(cat_ms.items(), key=lambda x: -x[1])
        },
        "profiler_flops_g": round(total_flops / 1e9, 2),
        "profiler_kernel_ms": round(total_kernel_ms, 3),
        "top_kernels": top_kernels,
        "wall_ms": round(wall_ms, 3),
    }


def profile_full_run(model, scheduler, cfg, inputs, latent_shape) -> dict[str, Any]:
    ti = model.transformer_infer
    if hasattr(ti, "reset_kv_store_profile"):
        ti.reset_kv_store_profile()

    phase_timer = PhaseTimer()
    _patch_model_for_phase_timing(model, phase_timer)

    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent_shape)
    scheduler.num_output_frames = num_out
    scheduler.prepare(seed=42, latent_shape=list(ls_adj), image_encoder_output=None)
    infer_steps = int(scheduler.infer_steps)
    num_forwards = num_chunks * infer_steps + num_chunks

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
        DisaggSFKVCacheManager.teardown(model)
    _sync()
    elapsed = time.perf_counter() - t0

    kv_prof = ti.finalize_kv_store_profile() if hasattr(ti, "finalize_kv_store_profile") else {}
    phase_ms = dict(phase_timer.times_ms)
    total_phase_ms = sum(phase_ms.values())
    sa = float(kv_prof.get("self_attn_ms", 0))
    ca = float(kv_prof.get("cross_attn_ms", 0))
    store = float(kv_prof.get("store_kv_ms", 0))
    read = float(kv_prof.get("kv_read_ms", 0))
    cross_store = float(kv_prof.get("cross_kv_store_ms", 0))
    transformer_ms = phase_ms.get("transformer", 0)
    attn_ms = sa + ca
    ffn_other_ms = max(0.0, transformer_ms - attn_ms)
    unaccounted_ms = elapsed * 1000 - total_phase_ms

    ideal_compute_ms = 1036.659 / 125.0 * 1000 if cfg["target_height"] == 480 else 4097.885 / 125.0 * 1000

    return {
        "num_forwards": num_forwards,
        "wall_s": round(elapsed, 4),
        "per_forward_ms": round(elapsed * 1000 / num_forwards, 2),
        "phase_total_ms": {k: round(v, 2) for k, v in phase_ms.items()},
        "phase_pct_of_gpu_measured": {k: round(100 * v / total_phase_ms, 2) for k, v in phase_ms.items()},
        "transformer_sub_ms": {
            "self_attn": round(sa, 2),
            "cross_attn": round(ca, 2),
            "ffn_and_block_overhead": round(ffn_other_ms, 2),
            "kv_store_read": round(store + read + cross_store, 2),
        },
        "transformer_sub_pct": {
            "self_attn": round(100 * sa / transformer_ms, 2) if transformer_ms else 0,
            "cross_attn": round(100 * ca / transformer_ms, 2) if transformer_ms else 0,
            "ffn_and_block_overhead": round(100 * ffn_other_ms / transformer_ms, 2) if transformer_ms else 0,
            "kv_store_read": round(100 * (store + read + cross_store) / transformer_ms, 2) if transformer_ms else 0,
        },
        "compute_vs_overhead_ms": {
            "gemm_attn_ffn_compute_proxy": round(sa + ca + ffn_other_ms, 2),
            "kv_io": round(store + read + cross_store, 2),
            "pre_post_embed": round(phase_ms.get("pre_infer", 0) + phase_ms.get("post_infer", 0), 2),
            "unaccounted_cpu_sync": round(max(0.0, unaccounted_ms), 2),
        },
        "compute_vs_overhead_pct": {},
        "ideal_compute_ms_at_125tflops": round(ideal_compute_ms, 2),
        "achieved_vs_ideal_compute_pct": round(100 * ideal_compute_ms / (elapsed * 1000), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--output_json", default="save_results/sf_forward_breakdown.json")
    parser.add_argument("--output_md", default="save_results/sf_forward_breakdown.md")
    args = parser.parse_args()

    cfg = set_config(model_path=args.model_path, task="t2v", model_cls="wan2.1_sf", config_path=args.config_json)
    cfg["cpu_offload"] = False
    cfg["parallel"] = False
    ar = dict(cfg.get("ar_config", {}))
    ar["profile_kv_store"] = True
    cfg["ar_config"] = ar

    seed_all(42)
    inputs = _prepare_inputs(cfg, DEFAULT_PROMPT)
    latent_shape = _latent_shape(cfg)

    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)

    single_forwards = []
    for seg_idx in (0, 3, 6):
        single_forwards.append(profile_single_forward(model, scheduler, cfg, inputs, latent_shape, seg_idx, 0))
        DisaggSFKVCacheManager.teardown(model)

    full_run = profile_full_run(model, scheduler, cfg, inputs, latent_shape)

    cvo = full_run["compute_vs_overhead_ms"]
    cvo_total = sum(cvo.values())
    full_run["compute_vs_overhead_pct"] = {k: round(100 * v / cvo_total, 2) for k, v in cvo.items()}

    result = {
        "config": {
            "resolution": f"{cfg['target_width']}x{cfg['target_height']}",
            "cpu_offload": cfg["cpu_offload"],
            "infer_steps": cfg.get("infer_steps", 4),
            "num_layers": cfg.get("num_layers"),
            "dim": cfg.get("dim"),
        },
        "single_forward_profiles": single_forwards,
        "full_run": full_run,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fr = full_run
    sf_mid = single_forwards[1]
    md = [
        "# SF Forward Breakdown Profile",
        "",
        f"- Resolution: {result['config']['resolution']}, cpu_offload={result['config']['cpu_offload']}",
        "",
        "## Full run (35 forwards)",
        "",
        f"- Total: **{fr['wall_s']} s**, per-forward **{fr['per_forward_ms']} ms**",
        f"- Ideal compute @125 TFLOPS: **{fr['ideal_compute_ms_at_125tflops']} ms** ({fr['achieved_vs_ideal_compute_pct']}% of wall)",
        "",
        "### Phase breakdown (CUDA events)",
        "",
        "| Phase | ms | % |",
        "|-------|-----|---|",
    ]
    for k, v in fr["phase_total_ms"].items():
        md.append(f"| {k} | {v} | {fr['phase_pct_of_gpu_measured'][k]}% |")

    md += [
        "",
        "### Transformer sub-components (instrumented)",
        "",
        "| Component | ms | % of transformer |",
        "|-----------|-----|-------------------|",
    ]
    for k, v in fr["transformer_sub_ms"].items():
        md.append(f"| {k} | {v} | {fr['transformer_sub_pct'][k]}% |")

    md += [
        "",
        "### Compute vs overhead (full run)",
        "",
        "| Category | ms | % |",
        "|----------|-----|---|",
    ]
    for k, v in fr["compute_vs_overhead_ms"].items():
        md.append(f"| {k} | {v} | {fr['compute_vs_overhead_pct'][k]}% |")

    md += [
        "",
        f"## Single forward @ chunk {sf_mid['seg_index']} (K={sf_mid['k_tokens']} tokens)",
        "",
        f"- Wall: **{sf_mid['wall_ms']} ms**",
        "",
        "### Phase",
        "",
        "| Phase | ms | % |",
        "|-------|-----|---|",
    ]
    for k, v in sf_mid["phase_ms"].items():
        md.append(f"| {k} | {round(v,2)} | {sf_mid['phase_pct'][k]}% |")

    md += [
        "",
        "### Transformer sub-components",
        "",
        "| Component | ms | % |",
        "|-----------|-----|---|",
    ]
    for k, v in sf_mid["transformer_breakdown_ms"].items():
        pct_key = k if k in sf_mid["transformer_breakdown_pct"] else None
        pct = sf_mid["transformer_breakdown_pct"].get("self_attn" if k == "self_attn" else k.replace("self_attn_", "").replace("cross_", "cross_"), 0)
        if k == "self_attn":
            pct = sf_mid["transformer_breakdown_pct"]["self_attn"]
        elif k == "cross_attn":
            pct = sf_mid["transformer_breakdown_pct"]["cross_attn"]
        elif k == "ffn_and_block_overhead":
            pct = sf_mid["transformer_breakdown_pct"]["ffn_and_block_overhead"]
        else:
            pct = "-"
        md.append(f"| {k} | {v} | {pct}% |" if pct != "-" else f"| {k} | {v} | - |")

    md += [
        "",
        "### CUDA kernel categories (torch profiler)",
        "",
        "| Category | ms | % |",
        "|----------|-----|---|",
    ]
    for k, v in sf_mid["kernel_category_ms"].items():
        md.append(f"| {k} | {v} | {sf_mid['kernel_category_pct'][k]}% |")

    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
