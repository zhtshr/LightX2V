#!/usr/bin/env python3
"""Probe whether kv_offload prefetch DMA stalls compute (5s quick check).

Patches RollingKVCachePool / KIVI begin_layer / end_layer / _prefetch_layer to
record CUDA-event timings:
  - begin_wait_ms: time current stream waits for _load_done (true stall)
  - layer_compute_ms: time from after begin_layer wait to end_layer call
  - prefetch_dma_ms: H2D copy of one compressed layer on prefetch_stream
"""

from __future__ import annotations

import argparse
import json
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


class PrefetchProbe:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.begin_wait_ms: dict[int, float] = defaultdict(float)
        self.layer_compute_ms: dict[int, float] = defaultdict(float)
        self.prefetch_dma_ms: dict[int, float] = defaultdict(float)
        self.begin_calls = 0
        self.prefetch_calls = 0
        self.miss_reloads = 0  # begin_layer had to sync-load (loaded_layer mismatch)
        self._pending: list[tuple[torch.cuda.Event, torch.cuda.Event, str, int]] = []
        self._layer_start: dict[int, torch.cuda.Event] = {}

    def mark(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        return s, e

    def end(self, s: torch.cuda.Event, e: torch.cuda.Event, kind: str, layer: int) -> None:
        e.record()
        self._pending.append((s, e, kind, layer))

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for s, e, kind, layer in self._pending:
            ms = float(s.elapsed_time(e))
            if kind == "begin_wait":
                self.begin_wait_ms[layer] += ms
            elif kind == "layer_compute":
                self.layer_compute_ms[layer] += ms
            elif kind == "prefetch_dma":
                self.prefetch_dma_ms[layer] += ms
        self._pending.clear()

    def snapshot(self) -> dict[str, Any]:
        self.finalize()
        wait_total = sum(self.begin_wait_ms.values())
        comp_total = sum(self.layer_compute_ms.values())
        dma_total = sum(self.prefetch_dma_ms.values())
        # Per-layer mean over calls: approx by / (calls/num_layers) later
        layers = sorted(set(self.begin_wait_ms) | set(self.layer_compute_ms) | set(self.prefetch_dma_ms))
        per_layer = []
        for L in layers:
            w = self.begin_wait_ms.get(L, 0.0)
            c = self.layer_compute_ms.get(L, 0.0)
            d = self.prefetch_dma_ms.get(L, 0.0)
            per_layer.append({
                "layer": L,
                "begin_wait_ms": round(w, 3),
                "layer_compute_ms": round(c, 3),
                "prefetch_dma_ms": round(d, 3),
                "wait_gt_compute": w > c,
                "dma_gt_compute": d > c,
            })
        stall_layers = sum(1 for r in per_layer if r["begin_wait_ms"] > 0.05)
        return {
            "begin_calls": self.begin_calls,
            "prefetch_calls": self.prefetch_calls,
            "miss_reloads": self.miss_reloads,
            "begin_wait_ms_total": round(wait_total, 3),
            "layer_compute_ms_total": round(comp_total, 3),
            "prefetch_dma_ms_total": round(dma_total, 3),
            "begin_wait_pct_of_compute": round(100.0 * wait_total / comp_total, 2) if comp_total > 0 else None,
            "layers_with_nonzero_wait": stall_layers,
            "per_layer": per_layer,
        }


def _patch_kv_offload_probe(kv_cache: Any, probe: PrefetchProbe) -> None:
    orig_begin = kv_cache.begin_layer
    orig_end = kv_cache.end_layer
    orig_prefetch = kv_cache._prefetch_layer
    orig_copy = kv_cache._copy_layer_to_gpu

    def begin_layer(layer_id: int) -> None:
        if not kv_cache._kv_offload:
            return orig_begin(layer_id)
        layer_id = int(layer_id)
        probe.begin_calls += 1
        if kv_cache._loaded_layer != layer_id:
            probe.miss_reloads += 1
            # same as original: sync prefetch then wait
            kv_cache._prefetch_layer(layer_id)
        # Measure wait on compute stream for prefetch completion
        s, e = probe.mark()
        torch.cuda.current_stream().wait_event(kv_cache._load_done)
        # force a no-op on compute stream so elapsed_time covers the wait
        # (wait_event itself is async; record end after a tiny op)
        _ = torch.empty(1, device=kv_cache._device)
        probe.end(s, e, "begin_wait", layer_id)
        # start layer compute timer
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        probe._layer_start[layer_id] = start

    def end_layer(layer_id: int, next_prefetch: int | None = None) -> None:
        layer_id = int(layer_id)
        start = probe._layer_start.pop(layer_id, None)
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            probe._pending.append((start, end, "layer_compute", layer_id))
        if not kv_cache._kv_offload:
            return orig_end(layer_id, next_prefetch=next_prefetch)
        next_layer = layer_id + 1 if next_prefetch is None else int(next_prefetch)
        kv_cache._prefetch_layer(next_layer)

    def _prefetch_layer(layer_id: int) -> None:
        if layer_id >= kv_cache._num_layers:
            return
        probe.prefetch_calls += 1
        with torch.cuda.stream(kv_cache._prefetch_stream):
            kv_cache._prefetch_stream.wait_event(kv_cache._cpu_update_event(layer_id))
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(kv_cache._prefetch_stream)
            orig_copy(layer_id)
            e.record(kv_cache._prefetch_stream)
            probe._pending.append((s, e, "prefetch_dma", int(layer_id)))
            kv_cache._load_done.record(kv_cache._prefetch_stream)
        kv_cache._loaded_layer = int(layer_id)

    kv_cache.begin_layer = begin_layer  # type: ignore[method-assign]
    kv_cache.end_layer = end_layer  # type: ignore[method-assign]
    kv_cache._prefetch_layer = _prefetch_layer  # type: ignore[method-assign]


def _latent_shape(cfg: dict[str, Any]) -> list[int]:
    return [
        cfg.get("num_channels_latents", 16),
        (cfg["target_video_length"] - 1) // cfg["vae_stride"][0] + 1,
        cfg["target_height"] // cfg["vae_stride"][1],
        cfg["target_width"] // cfg["vae_stride"][2],
    ]


def _load_payload(cfg: dict[str, Any], cache: Path, prompt: str) -> dict[str, Any]:
    latent = _latent_shape(cfg)
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        payload["latent_shape"] = latent
        payload["inputs"]["latent_shape"] = latent
        return payload
    text_encoder = load_wan_text_encoder(cfg)[0]
    text_len = int(cfg.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    payload = {
        "seed": 42,
        "latent_shape": latent,
        "inputs": {
            "text_encoder_output": {"context": context.cuda(), "context_null": None},
            "image_encoder_output": None,
            "latent_shape": latent,
        },
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    # move context to cpu for save already done via torch.save snapshot; keep on cuda for run
    payload["inputs"]["text_encoder_output"]["context"] = payload["inputs"]["text_encoder_output"]["context"].cuda()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--config_json", default="configs/self_forcing/wan_t2v_sf_14b_int8_sp_5s_kivi_offload.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/sf_14b_int8_encoder_inputs_f81.pt")
    parser.add_argument("--output_json", default="save_results/optimization_study/sf_14b_kv_offload_prefetch_probe_5s.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = set_config(model_path=args.model_path, task="t2v", model_cls="wan2.1_sf", config_path=args.config_json)
    cfg["cpu_offload"] = False
    cfg["parallel"] = False
    ar = dict(cfg.get("ar_config", {}))
    ar["kv_offload"] = True
    cfg["ar_config"] = ar

    seed_all(args.seed)
    payload = _load_payload(cfg, Path(args.inputs_cache), args.prompt)
    if not payload["inputs"]["text_encoder_output"]["context"].is_cuda:
        payload["inputs"]["text_encoder_output"]["context"] = payload["inputs"]["text_encoder_output"]["context"].cuda()

    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)

    probe = PrefetchProbe()
    inputs = payload["inputs"]
    latent = list(payload["latent_shape"])

    ls_adj, num_out, num_chunks = DisaggSFKVCacheManager.setup(model, cfg, latent)
    scheduler.num_output_frames = num_out
    scheduler.num_chunks = num_chunks
    scheduler.prepare(seed=args.seed, latent_shape=list(ls_adj), image_encoder_output=None)

    kv = model.kv_cache_manager.self_attn_kv_cache
    assert bool(getattr(kv, "_kv_offload", False)), "kv_offload not enabled on cache pool"
    _patch_kv_offload_probe(kv, probe)

    # optional: also time cross-attn pool if it has offload (usually no)
    infer_steps = int(scheduler.infer_steps)
    torch.cuda.synchronize()
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
        snap = probe.snapshot()
        DisaggSFKVCacheManager.teardown(model)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    # Summarize: average wait vs compute per begin call
    n_begin = max(snap["begin_calls"], 1)
    avg_wait = snap["begin_wait_ms_total"] / n_begin
    avg_comp = snap["layer_compute_ms_total"] / n_begin
    avg_dma = snap["prefetch_dma_ms_total"] / max(snap["prefetch_calls"], 1)

    result = {
        "metric": "sf_kv_offload_prefetch_probe",
        "config_json": args.config_json,
        "target_video_length": cfg.get("target_video_length"),
        "num_chunks": num_chunks,
        "infer_steps": infer_steps,
        "num_layers": int(cfg.get("num_layers", 0)),
        "wall_s": round(wall_s, 3),
        "probe": snap,
        "summary": {
            "avg_begin_wait_ms": round(avg_wait, 3),
            "avg_layer_compute_ms": round(avg_comp, 3),
            "avg_prefetch_dma_ms": round(avg_dma, 3),
            "prefetch_slower_than_compute": avg_dma > avg_comp,
            "stall_significant": avg_wait > 0.05 and avg_wait > 0.1 * avg_comp,
            "verdict": (
                "PREFETCH_BOTTLENECK"
                if avg_wait > 0.1 * avg_comp and avg_wait > 0.05
                else "PREFETCH_HIDDEN_OK"
                if avg_dma <= avg_comp
                else "DMA_SLOWER_BUT_OVERLAPPED"
            ),
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], indent=2))
    print(
        f"wall={wall_s:.2f}s begin_calls={snap['begin_calls']} miss_reloads={snap['miss_reloads']} "
        f"wait_total={snap['begin_wait_ms_total']:.1f}ms "
        f"({snap['begin_wait_pct_of_compute']}% of layer_compute) "
        f"dma_total={snap['prefetch_dma_ms_total']:.1f}ms"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
