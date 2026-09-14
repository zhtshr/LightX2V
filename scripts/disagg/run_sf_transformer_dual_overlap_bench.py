#!/usr/bin/env python3
"""SF (wan2.1_sf) dual-request comm/compute overlap benchmark with Ulysses SP.

Measures full AR denoise (all chunks x infer_steps + rerun) for two in-flight requests:
  - single_request_s
  - dual_back_to_back_s (serial, model.infer)
  - dual_a2a_serial_s / dual_a2a_overlap_s (layer decomposed, P>1 only)

Reuses Phase-3 A2A overlap orchestration from run_phase3_dual_overlap_bench.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from lightx2v.common.kvcache import KVCacheManager
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


def _load_phase3_module():
    import sys

    bench_path = Path(__file__).with_name("run_phase3_dual_overlap_bench.py")
    spec = importlib.util.spec_from_file_location("phase3_dual_overlap", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {bench_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
    return torch.device(AI_DEVICE)


def _sync() -> None:
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
    if args.seq_p_size > 1:
        config["cpu_offload"] = False
        config["parallel"] = {
            "seq_p_size": int(args.seq_p_size),
            "seq_p_attn_type": args.seq_p_attn_type,
        }
    else:
        config["parallel"] = False
    return config


def _init_distributed(config: dict[str, Any]) -> None:
    if not config.get("parallel"):
        return
    platform_device = PLATFORM_DEVICE_REGISTER.get(__import__("os").getenv("PLATFORM", "cuda"), None)
    if platform_device is None:
        raise RuntimeError("platform device registry is unavailable")
    platform_device.init_parallel_env()
    set_parallel_config(config)


def _validate_seq_p(config: dict[str, Any], seq_p_size: int) -> None:
    if seq_p_size <= 1:
        return
    num_heads = int(config.get("num_heads", 0))
    if num_heads > 0 and num_heads % seq_p_size != 0:
        raise ValueError(
            f"Ulysses SP requires num_heads ({num_heads}) % seq_p_size ({seq_p_size}) == 0"
        )


def _prepare_inputs_cache(
    config: dict[str, Any],
    cache_path: Path,
    prompt: str,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    if cache_path.is_file() and not force:
        if is_main_process():
            print(f"Loaded encoder inputs cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if dist.is_initialized():
            dist.barrier()
        return payload

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Preparing T5 encoder inputs (excluded from bench timing)...")
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder

    latent_shape = _latent_shape_from_config(config)
    inputs = {
        "text_encoder_output": {"context": context, "context_null": None},
        "image_encoder_output": None,
        "latent_shape": latent_shape,
    }
    payload = {"seed": seed, "latent_shape": latent_shape, "inputs": inputs}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"Wrote encoder inputs cache: {cache_path}")
    _sync()
    if dist.is_initialized():
        dist.barrier()
    return payload


@dataclass
class SFTenantCtx:
    name: str
    scheduler: Any
    inputs: dict[str, Any]
    kv_cache_manager: KVCacheManager
    pre_infer_out: Any = None
    x: torch.Tensor | None = None
    ti_snap: Any = None
    cross_kv_len: int | None = None
    block_cache: dict[int, Any] | None = None

    def __post_init__(self) -> None:
        if self.block_cache is None:
            self.block_cache = {}


def _sf_capture_ti_extra(p3: Any, ti: Any, tenant: SFTenantCtx) -> None:
    tenant.ti_snap = p3._capture_ti_snap(ti)
    tenant.cross_kv_len = getattr(ti, "_cross_kv_len", None)


def _sf_apply_ti_extra(p3: Any, ti: Any, tenant: SFTenantCtx) -> None:
    p3._apply_ti_snap(ti, tenant.ti_snap)
    if tenant.cross_kv_len is not None:
        ti._cross_kv_len = tenant.cross_kv_len


def _sf_run_self_attn_block(p3: Any, wan: Any, ti: Any, tenant: SFTenantCtx, block_idx: int) -> Any:
    block = p3._ensure_block(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    x = tenant.x
    pre = tenant.pre_infer_out
    if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
        x = block.compute_phases[0].before_proj.apply(x) + pre.x
    mods = ti.pre_process(block.compute_phases[0].modulation, pre.embed0)
    y_out = ti.infer_self_attn_with_kvcache(
        block.compute_phases[0],
        pre.grid_sizes.tensor,
        x,
        pre.seq_lens,
        pre.freqs,
        mods[0],
        mods[1],
    )
    tenant.x = x
    _sf_capture_ti_extra(p3, ti, tenant)
    return p3.BlockMid(mods=mods, y_out=y_out)


def _sf_run_cross_ffn_block(p3: Any, wan: Any, ti: Any, tenant: SFTenantCtx, block_idx: int, mid: Any) -> None:
    block = p3._ensure_block(tenant, wan, ti, block_idx)
    ti.block_idx = block_idx
    gate_msa = mid.mods[2]
    x, attn_out = ti.infer_cross_attn_with_kvcache(
        block.compute_phases[1], tenant.x, tenant.pre_infer_out.context, mid.y_out, gate_msa,
    )
    y = ti.infer_ffn(block.compute_phases[2], x, attn_out, mid.mods[3], mid.mods[4])
    tenant.x = ti.post_process(x, y, mid.mods[5], tenant.pre_infer_out)
    _sf_capture_ti_extra(p3, ti, tenant)


def _sf_a2a_overlap_main_blocks(
    p3: Any,
    model: Any,
    tenant_a: SFTenantCtx,
    tenant_b: SFTenantCtx,
    orch: Any,
    *,
    overlap: bool,
) -> None:
    """SF KV-cache layer schedule with optional a2a-window overlap."""
    wan_a, ti_a = p3._bind_tenant(model, tenant_a)
    num_blocks = len(wan_a.transformer_weights.blocks)
    p3._preload_blocks(tenant_a, wan_a, ti_a, num_blocks)
    wan_b, ti_b = p3._bind_tenant(model, tenant_b)
    p3._preload_blocks(tenant_b, wan_b, ti_b, num_blocks)

    orch.enabled = True
    orch.overlap_enabled = False
    orch.other_compute_cb = None
    orch.pending_other_self_fn = None

    mid_a = _sf_run_self_attn_block(p3, wan_a, ti_a, tenant_a, 0)

    for k in range(num_blocks):
        if overlap:
            orch.overlap_enabled = True
            orch.pending_other_self_fn = None
            orch.other_compute_cb = p3._OnceCompute(
                lambda k=k, m=mid_a: _sf_run_cross_ffn_block(
                    p3, *p3._bind_tenant(model, tenant_a), tenant_a, k, m,
                ),
            )
        else:
            orch.overlap_enabled = False
            orch.other_compute_cb = None
            orch.pending_other_self_fn = None

        wan_b, ti_b = p3._bind_tenant(model, tenant_b)
        _sf_apply_ti_extra(p3, ti_b, tenant_b)
        if overlap:
            orch.ti_ref = ti_b
            orch.reset_self_attn_counters()
        try:
            mid_b = _sf_run_self_attn_block(p3, wan_b, ti_b, tenant_b, k)
        finally:
            if overlap:
                orch.ti_ref = None

        if not overlap:
            wan_a, ti_a = p3._bind_tenant(model, tenant_a)
            _sf_apply_ti_extra(p3, ti_a, tenant_a)
            _sf_run_cross_ffn_block(p3, wan_a, ti_a, tenant_a, k, mid_a)

        if k + 1 < num_blocks:
            if overlap:
                orch.overlap_enabled = True
                orch.pending_other_self_fn = None
                orch.other_compute_cb = p3._OnceCompute(
                    lambda k=k, mb=mid_b: _sf_run_cross_ffn_block(
                        p3, *p3._bind_tenant(model, tenant_b), tenant_b, k, mb,
                    ),
                )
                wan_a, ti_a = p3._bind_tenant(model, tenant_a)
                _sf_apply_ti_extra(p3, ti_a, tenant_a)
                if overlap:
                    orch.ti_ref = ti_a
                    orch.reset_self_attn_counters()
                try:
                    mid_a = _sf_run_self_attn_block(p3, wan_a, ti_a, tenant_a, k + 1)
                finally:
                    if overlap:
                        orch.ti_ref = None
            else:
                orch.overlap_enabled = False
                orch.other_compute_cb = None
                wan_a, ti_a = p3._bind_tenant(model, tenant_a)
                _sf_apply_ti_extra(p3, ti_a, tenant_a)
                mid_a = _sf_run_self_attn_block(p3, wan_a, ti_a, tenant_a, k + 1)
                wan_b, ti_b = p3._bind_tenant(model, tenant_b)
                _sf_apply_ti_extra(p3, ti_b, tenant_b)
                _sf_run_cross_ffn_block(p3, wan_b, ti_b, tenant_b, k, mid_b)
        else:
            orch.overlap_enabled = False
            orch.other_compute_cb = None
            wan_b, ti_b = p3._bind_tenant(model, tenant_b)
            _sf_apply_ti_extra(p3, ti_b, tenant_b)
            _sf_run_cross_ffn_block(p3, wan_b, ti_b, tenant_b, k, mid_b)

    orch.overlap_enabled = False
    orch.other_compute_cb = None
    orch.pending_other_self_fn = None
    orch.enabled = False
    orch.compute_stream.synchronize()
    orch.comm_stream.synchronize()
    orch.other_self_stream.synchronize()
    _sync()


def _create_tenant_kv(model: Any, config: dict[str, Any], latent_shape_adj: list[int]) -> KVCacheManager:
    kv_manager = KVCacheManager(
        config=config,
        device=_run_device(),
        sp_group=getattr(model, "seq_p_group", None),
    )
    kv_manager._create_kv_caches(list(latent_shape_adj))
    return kv_manager


def _bind_sf_kv(model: Any, kv_manager: KVCacheManager, step_index: int) -> None:
    model.kv_cache_manager = kv_manager
    kv_manager.current_step = step_index
    if hasattr(model, "transformer_infer"):
        model.transformer_infer.kv_cache_manager = kv_manager


def _sf_finish_step_tenant(p3: Any, model: Any, tenant: SFTenantCtx) -> None:
    wan, ti = p3._bind_tenant(model, tenant)
    x = ti.infer_non_blocks(wan.transformer_weights, tenant.x, tenant.pre_infer_out.embed)
    if wan.config["seq_parallel"]:
        x = wan._seq_parallel_post_process(x)
    noise_pred = wan.post_infer.infer(x, tenant.pre_infer_out)[0]
    seg_start = tenant.scheduler.seg_index * tenant.scheduler.num_frame_per_chunk
    seg_end = min(
        (tenant.scheduler.seg_index + 1) * tenant.scheduler.num_frame_per_chunk,
        tenant.scheduler.num_output_frames,
    )
    tenant.scheduler.noise_pred[:, seg_start:seg_end] = noise_pred


def _sf_finish_rerun_tenant(p3: Any, model: Any, tenant: SFTenantCtx) -> None:
    """Rerun forward only; SF smoke path does not call step_post."""
    wan, ti = p3._bind_tenant(model, tenant)
    x = ti.infer_non_blocks(wan.transformer_weights, tenant.x, tenant.pre_infer_out.embed)
    if wan.config["seq_parallel"]:
        x = wan._seq_parallel_post_process(x)
    noise_pred = wan.post_infer.infer(x, tenant.pre_infer_out)[0]
    seg_start = tenant.scheduler.seg_index * tenant.scheduler.num_frame_per_chunk
    seg_end = min(
        (tenant.scheduler.seg_index + 1) * tenant.scheduler.num_frame_per_chunk,
        tenant.scheduler.num_output_frames,
    )
    tenant.scheduler.noise_pred[:, seg_start:seg_end] = noise_pred


@contextmanager
def _kv_bind_patch(p3: Any, model: Any):
    orig_bind = p3._bind_tenant

    def _bind_tenant_with_kv(m: Any, tenant: SFTenantCtx) -> tuple[Any, Any]:
        _bind_sf_kv(m, tenant.kv_cache_manager, int(tenant.scheduler.step_index))
        return orig_bind(m, tenant)

    p3._bind_tenant = _bind_tenant_with_kv
    try:
        yield
    finally:
        p3._bind_tenant = orig_bind


def _sf_one_forward_decomposed(
    p3: Any,
    model: Any,
    tenant_a: SFTenantCtx,
    tenant_b: SFTenantCtx,
    orch: Any,
    *,
    overlap: bool,
    is_rerun: bool,
) -> None:
    seg_idx = tenant_a.scheduler.seg_index
    step_index = tenant_a.scheduler.step_index
    tenant_a.scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=is_rerun)
    tenant_b.scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=is_rerun)
    tenant_a.ti_snap = None
    tenant_b.ti_snap = None
    tenant_a.cross_kv_len = None
    tenant_b.cross_kv_len = None

    _bind_sf_kv(model, tenant_a.kv_cache_manager, step_index)
    p3._pre_infer_tenant(model, tenant_a)
    _bind_sf_kv(model, tenant_b.kv_cache_manager, step_index)
    p3._pre_infer_tenant(model, tenant_b)

    _sf_a2a_overlap_main_blocks(p3, model, tenant_a, tenant_b, orch, overlap=overlap)

    _bind_sf_kv(model, tenant_a.kv_cache_manager, step_index)
    if is_rerun:
        _sf_finish_rerun_tenant(p3, model, tenant_a)
    else:
        _sf_finish_step_tenant(p3, model, tenant_a)

    _bind_sf_kv(model, tenant_b.kv_cache_manager, step_index)
    if is_rerun:
        _sf_finish_rerun_tenant(p3, model, tenant_b)
    else:
        _sf_finish_step_tenant(p3, model, tenant_b)

    if not is_rerun:
        tenant_a.scheduler.step_post()
        tenant_b.scheduler.step_post()


def _run_sf_serial(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
) -> dict[str, Any]:
    inputs = payload["inputs"]
    latent_shape = list(payload["latent_shape"])
    seed = int(payload["seed"])

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)

    infer_steps = int(scheduler.infer_steps)
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
    return {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "include_rerun": include_rerun,
        "num_output_frames": num_output_frames,
        "latent_shape": latent_shape_adj,
    }


def _prepare_tenants(
    model: Any,
    config: dict[str, Any],
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
) -> tuple[SFTenantCtx, SFTenantCtx, dict[str, Any]]:
    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(
        model, config, list(payload_a["latent_shape"]),
    )
    DisaggSFKVCacheManager.teardown(model)

    meta = {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": int(config.get("infer_steps", 4)),
        "num_output_frames": num_output_frames,
        "latent_shape": latent_shape_adj,
    }

    scheduler_a = load_wan_scheduler(config)
    scheduler_b = load_wan_scheduler(config)
    scheduler_a.num_output_frames = num_output_frames
    scheduler_b.num_output_frames = num_output_frames

    kv_a = _create_tenant_kv(model, config, latent_shape_adj)
    kv_b = _create_tenant_kv(model, config, latent_shape_adj)

    tenant_a = SFTenantCtx("A", scheduler_a, payload_a["inputs"], kv_a)
    tenant_b = SFTenantCtx("B", scheduler_b, payload_b["inputs"], kv_b)
    return tenant_a, tenant_b, meta


def _run_sf_dual_a2a(
    p3: Any,
    model: Any,
    tenant_a: SFTenantCtx,
    tenant_b: SFTenantCtx,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    meta: dict[str, Any],
    *,
    overlap: bool,
    include_rerun: bool,
) -> tuple[float, dict[str, Any]]:
    device = _run_device()
    orch = p3.A2AOrchestrator(device)
    orch.install()

    tenant_a.scheduler.prepare(
        seed=int(payload_a["seed"]),
        latent_shape=meta["latent_shape"],
        image_encoder_output=None,
    )
    tenant_b.scheduler.prepare(
        seed=int(payload_b["seed"]),
        latent_shape=meta["latent_shape"],
        image_encoder_output=None,
    )

    num_chunks = int(meta["num_chunks"])
    infer_steps = int(meta["infer_steps_per_chunk"])

    def _body() -> None:
        with _kv_bind_patch(p3, model):
            for seg_idx in range(num_chunks):
                for step_index in range(infer_steps):
                    tenant_a.scheduler.seg_index = seg_idx
                    tenant_b.scheduler.seg_index = seg_idx
                    tenant_a.scheduler.step_index = step_index
                    tenant_b.scheduler.step_index = step_index
                    _sf_one_forward_decomposed(
                        p3, model, tenant_a, tenant_b, orch,
                        overlap=overlap, is_rerun=False,
                    )
                if include_rerun:
                    tenant_a.scheduler.seg_index = seg_idx
                    tenant_b.scheduler.seg_index = seg_idx
                    rerun_step = infer_steps - 1
                    tenant_a.scheduler.step_index = rerun_step
                    tenant_b.scheduler.step_index = rerun_step
                    _sf_one_forward_decomposed(
                        p3, model, tenant_a, tenant_b, orch,
                        overlap=overlap, is_rerun=True,
                    )

    try:
        wall = p3._time_fn(_body)
        stats = orch.stats_snapshot()
        stats["overlap"] = overlap
        return wall, stats
    finally:
        orch.restore()


def _time_fn(fn: Callable[[], None]) -> float:
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="SF dual-request SP overlap benchmark")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed_a", type=int, default=42)
    parser.add_argument("--seed_b", type=int, default=43)
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--seq_p_attn_type", default="ulysses")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument(
        "--inputs_cache",
        default="save_results/optimization_study/sf_phase1_encoder_inputs.pt",
    )
    parser.add_argument("--refresh_inputs_cache", action="store_true")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--baseline_p1_json",
        default="",
        help="Optional P=1 single-request json for overlap efficiency vs ideal N/T1",
    )
    args = parser.parse_args()

    p3 = _load_phase3_module()
    config = _load_config(args)
    _validate_seq_p(config, args.seq_p_size)
    seed_all(args.seed_a)
    _init_distributed(config)
    include_rerun = not args.no_rerun

    payload_a = _prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=args.prompt,
        seed=args.seed_a,
        force=args.refresh_inputs_cache,
    )
    payload_b = {
        "seed": args.seed_b,
        "latent_shape": payload_a["latent_shape"],
        "inputs": payload_a["inputs"],
    }
    payload_a = _move_tensor_tree(payload_a, _run_device())
    payload_b = _move_tensor_tree(payload_b, _run_device())

    model = load_wan_transformer(config)
    scheduler_single = load_wan_scheduler(config)
    model.set_scheduler(scheduler_single)

    run_meta: dict[str, Any] = {}
    for _ in range(args.warmup):
        _run_sf_serial(model, scheduler_single, config, payload_a, include_rerun=include_rerun)

    single_s = _time_fn(
        lambda: _run_sf_serial(model, scheduler_single, config, payload_a, include_rerun=include_rerun),
    )

    def _dual_b2b() -> None:
        _run_sf_serial(model, scheduler_single, config, payload_a, include_rerun=include_rerun)
        _run_sf_serial(model, scheduler_single, config, payload_b, include_rerun=include_rerun)

    dual_b2b_s = _time_fn(_dual_b2b)

    dual_a2a_serial_s: float | None = None
    dual_a2a_overlap_s: float | None = None
    a2a_serial_stats: dict[str, Any] = {}
    a2a_overlap_stats: dict[str, Any] = {}
    overlap_err: str | None = None

    if args.seq_p_size > 1 and dist.is_initialized():
        tenant_a, tenant_b, run_meta = _prepare_tenants(model, config, payload_a, payload_b)
        try:
            dual_a2a_serial_s, a2a_serial_stats = _run_sf_dual_a2a(
                p3, model, tenant_a, tenant_b, payload_a, payload_b, run_meta,
                overlap=False, include_rerun=include_rerun,
            )
            dual_a2a_overlap_s, a2a_overlap_stats = _run_sf_dual_a2a(
                p3, model, tenant_a, tenant_b, payload_a, payload_b, run_meta,
                overlap=True, include_rerun=include_rerun,
            )
        except Exception as exc:  # noqa: BLE001
            overlap_err = str(exc)
            if is_main_process():
                print(f"SF dual a2a overlap failed: {exc}")
    else:
        tenant_a, tenant_b, run_meta = _prepare_tenants(model, config, payload_a, payload_b)

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    baseline_p1_s: float | None = None
    if args.baseline_p1_json:
        p1_path = Path(args.baseline_p1_json)
        if p1_path.is_file():
            baseline_p1_s = json.loads(p1_path.read_text()).get("transformer_compute_s")

    def rps(n: float, t: float | None) -> float | None:
        return n / t if t and t > 0 else None

    overlap_vs_b2b = (
        dual_b2b_s / dual_a2a_overlap_s
        if dual_b2b_s and dual_a2a_overlap_s and dual_a2a_overlap_s > 0
        else None
    )
    overlap_vs_serial = (
        dual_a2a_serial_s / dual_a2a_overlap_s
        if dual_a2a_serial_s and dual_a2a_overlap_s and dual_a2a_overlap_s > 0
        else None
    )
    overlap_eff_vs_p1 = None
    if baseline_p1_s and dual_a2a_overlap_s and dual_a2a_overlap_s > 0:
        overlap_eff_vs_p1 = 2.0 * baseline_p1_s / (dual_a2a_overlap_s * args.seq_p_size)
    elif baseline_p1_s and dual_b2b_s and dual_b2b_s > 0 and args.seq_p_size == 1:
        overlap_eff_vs_p1 = 2.0 * baseline_p1_s / dual_b2b_s

    result = {
        "model_cls": "wan2.1_sf",
        "metric": "sf_dual_request_overlap",
        "description": "Two SF AR denoise requests; overlap = a2a||cross_ffn (P>1)",
        "seq_p_size": int(args.seq_p_size),
        "world_size": world_size,
        "include_rerun": include_rerun,
        "single_request_s": round(single_s, 4),
        "dual_back_to_back_s": round(dual_b2b_s, 4),
        "dual_a2a_serial_s": round(dual_a2a_serial_s, 4) if dual_a2a_serial_s else None,
        "dual_a2a_overlap_s": round(dual_a2a_overlap_s, 4) if dual_a2a_overlap_s else None,
        "baseline_p1_single_s": baseline_p1_s,
        "num_chunks": run_meta.get("num_chunks"),
        "infer_steps_per_chunk": run_meta.get("infer_steps_per_chunk"),
        "a2a_serial_stats": a2a_serial_stats,
        "a2a_overlap_stats": a2a_overlap_stats,
        "overlap_error": overlap_err,
        "speedup": {
            "overlap_vs_back_to_back": round(overlap_vs_b2b, 4) if overlap_vs_b2b else None,
            "overlap_vs_a2a_serial": round(overlap_vs_serial, 4) if overlap_vs_serial else None,
        },
        "throughput": {
            "single_rps": rps(1, single_s),
            "dual_back_to_back_rps": rps(2, dual_b2b_s),
            "dual_a2a_serial_rps": rps(2, dual_a2a_serial_s),
            "dual_a2a_overlap_rps": rps(2, dual_a2a_overlap_s),
        },
        "overlap_efficiency_vs_p1_ideal": round(overlap_eff_vs_p1, 4) if overlap_eff_vs_p1 else None,
        "config_json": args.config_json,
    }

    if is_main_process():
        print(
            f"seq_p={args.seq_p_size}: single={result['single_request_s']}s "
            f"dual_b2b={result['dual_back_to_back_s']}s "
            f"dual_a2a_serial={result['dual_a2a_serial_s']} "
            f"dual_a2a_overlap={result['dual_a2a_overlap_s']}"
        )
        if result["overlap_efficiency_vs_p1_ideal"] is not None:
            print(f"  overlap_eff_vs_p1_ideal={result['overlap_efficiency_vs_p1_ideal']:.1%}")
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
