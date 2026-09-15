#!/usr/bin/env python3
"""Compare SF transformer bench: baseline vs FlowCache."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from lightx2v.common.flowcache import SFFlowCacheManager
from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_transformer, set_config
from lightx2v.utils.utils import seed_all

DEFAULT_PROMPT_CACHE = Path("save_results/optimization_study/sf_phase1_encoder_inputs.pt")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_compute(
    model: Any,
    scheduler: Any,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    include_rerun: bool,
    flowcache: SFFlowCacheManager | None,
) -> dict[str, Any]:
    inputs = payload["inputs"]
    latent_shape = list(payload["latent_shape"])
    seed = int(payload["seed"])

    if flowcache is not None:
        flowcache.reset()
        model.flowcache_manager = flowcache
        model.transformer_infer.flowcache_manager = flowcache

    latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(model, config, latent_shape)
    scheduler.num_output_frames = num_output_frames
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=None)

    infer_steps = int(scheduler.infer_steps)
    try:
        for seg_idx in range(num_chunks):
            if flowcache is not None:
                flowcache.begin_chunk(seg_idx)
            for step_index in range(infer_steps):
                model.kv_cache_manager.current_step = step_index
                scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)

                if flowcache is not None and flowcache.uses_feature_cache:
                    metric = flowcache.compute_metric(model, inputs)
                    if flowcache.should_skip_transformer(seg_idx, step_index, metric, is_rerun=False):
                        flowcache.apply_cached_noise_pred(scheduler, seg_idx)
                    else:
                        model.infer(inputs)
                        seg_start = seg_idx * scheduler.num_frame_per_chunk
                        seg_end = min((seg_idx + 1) * scheduler.num_frame_per_chunk, scheduler.num_output_frames)
                        noise_pred = scheduler.noise_pred[:, seg_start:seg_end]
                        flowcache.on_forward(seg_idx, step_index, metric, noise_pred)
                else:
                    model.infer(inputs)

                scheduler.step_post()

            if include_rerun:
                scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
                model.infer(inputs)
                if flowcache is not None and flowcache.uses_kv_compress:
                    flowcache.mark_chunk_completed(seg_idx)
                    flowcache.maybe_compress_kv(model, seg_idx)
    finally:
        DisaggSFKVCacheManager.teardown(model)

    _sync()
    stats = flowcache.feature_cache.stats() if flowcache is not None and flowcache.feature_cache is not None else {}
    return {
        "num_chunks": num_chunks,
        "infer_steps_per_chunk": infer_steps,
        "flowcache_stats": stats,
    }


def _bench(label: str, model, scheduler, config, payload, flowcache, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        _run_compute(model, scheduler, config, payload, include_rerun=True, flowcache=flowcache)

    samples: list[float] = []
    last_meta: dict[str, Any] = {}
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        last_meta = _run_compute(model, scheduler, config, payload, include_rerun=True, flowcache=flowcache)
        samples.append(time.perf_counter() - t0)

    avg = sum(samples) / len(samples)
    print(f"[{label}] transformer_s={avg:.3f} samples={[round(s, 3) for s in samples]} stats={last_meta.get('flowcache_stats', {})}")
    return {
        "label": label,
        "transformer_compute_s": round(avg, 4),
        "samples_s": [round(s, 4) for s in samples],
        "flowcache_stats": last_meta.get("flowcache_stats", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--baseline_config", default="configs/self_forcing/wan_t2v_sf_sp_bench.json")
    parser.add_argument("--flowcache_config", default="configs/self_forcing/wan_t2v_sf_flowcache.json")
    parser.add_argument("--inputs_cache", default=str(DEFAULT_PROMPT_CACHE))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=2)
    parser.add_argument("--output_json", default="save_results/sf_flowcache/bench_compare.json")
    parser.add_argument("--flowcache_log", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.inputs_cache)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing inputs cache: {cache_path}. Run run_sf_transformer_sp_bench.py --encoder_only first.")

    payload = torch.load(cache_path, map_location="cuda", weights_only=False)
    seed_all(int(payload["seed"]))

    base_cfg = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.baseline_config,
    )
    fc_cfg = set_config(
        model_path=args.model_path,
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=args.flowcache_config,
    )
    if args.flowcache_log:
        fc_cfg.setdefault("ar_config", {}).setdefault("flowcache", {})["log"] = True

    print("Loading model...")
    load_t0 = time.perf_counter()
    model = load_wan_transformer(base_cfg)
    scheduler = load_wan_scheduler(base_cfg)
    model.set_scheduler(scheduler)
    print(f"Model load: {time.perf_counter() - load_t0:.2f}s")

    flowcache_mgr = SFFlowCacheManager(fc_cfg)

    baseline = _bench("baseline", model, scheduler, base_cfg, payload, None, args.warmup, args.measure_iters)
    flowcache = _bench("flowcache", model, scheduler, fc_cfg, payload, flowcache_mgr, args.warmup, args.measure_iters)

    speedup = baseline["transformer_compute_s"] / flowcache["transformer_compute_s"] if flowcache["transformer_compute_s"] else 0.0
    summary = {
        "baseline": baseline,
        "flowcache": flowcache,
        "speedup_transformer": round(speedup, 4),
        "baseline_config": args.baseline_config,
        "flowcache_config": args.flowcache_config,
    }
    print(f"\n=== Summary ===")
    print(f"baseline:  {baseline['transformer_compute_s']:.3f}s")
    print(f"flowcache: {flowcache['transformer_compute_s']:.3f}s")
    print(f"speedup:   {speedup:.3f}x")
    print(f"reuse:     {flowcache['flowcache_stats']}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
