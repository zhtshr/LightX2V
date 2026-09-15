#!/usr/bin/env python3
"""Smoke tests for Wan Self-Forcing disaggregated inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import (
    DisaggSFKVCacheManager,
    get_phase2_send_mode,
    is_per_chunk_phase2,
    is_sf_model,
)
from lightx2v.disagg.utils import (
    estimate_transformer_buffer_sizes,
    load_wan_scheduler,
    load_wan_text_encoder,
    load_wan_transformer,
    set_config,
)
from lightx2v.utils.set_config import set_config as set_config_base
from lightx2v.utils.utils import seed_all


DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights "
    "and animated city signs casting vibrant reflections."
)


def _build_config(model_path: str, config_json: str, phase2_send_mode: str) -> dict:
    args = Namespace(
        model_cls="wan2.1_sf",
        task="t2v",
        model_path=model_path,
        config_json=config_json,
        seed=42,
        support_tasks=[],
        parallel=False,
        num_iterations=None,
    )
    config = set_config_base(args)
    config["model_cls"] = "wan2.1_sf"
    config["phase2_send_mode"] = phase2_send_mode
    disagg_cfg = dict(config.get("disagg_config") or {})
    disagg_cfg["phase2_send_mode"] = phase2_send_mode
    config["disagg_config"] = disagg_cfg
    return config


def test_helpers():
    cfg = {"model_cls": "wan2.1_sf", "phase2_send_mode": "per_chunk"}
    assert is_sf_model(cfg)
    assert get_phase2_send_mode(cfg) == "per_chunk"
    assert is_per_chunk_phase2(cfg)
    cfg["phase2_send_mode"] = "full"
    assert not is_per_chunk_phase2(cfg)
    print("[ok] sf_support helpers")


def test_scheduler_and_model_load(config: dict):
    scheduler = load_wan_scheduler(config)
    assert scheduler.__class__.__name__ == "WanSFScheduler"
    transformer = load_wan_transformer(config)
    assert transformer.__class__.__name__ == "WanSFModel"
    transformer.set_scheduler(scheduler)
    del transformer
    del scheduler
    torch.cuda.empty_cache()
    print("[ok] load_wan_scheduler + load_wan_transformer")


def _encode_prompt(config: dict, prompt: str):
    text_encoder = load_wan_text_encoder(config)[0]
    text_len = int(config.get("text_len", 512))
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    del text_encoder
    torch.cuda.empty_cache()
    latent_shape = _latent_shape(config)
    return {
        "text_encoder_output": {"context": context, "context_null": None},
        "image_encoder_output": None,
        "latent_shape": latent_shape,
    }


def _latent_shape(config: dict):
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def _run_sf_transformer(config: dict, inputs: dict, latent_shape: list[int], phase2_send_mode: str):
    from lightx2v.disagg.services.transformer import TransformerService

    config = dict(config)
    config["phase2_send_mode"] = phase2_send_mode
    service = TransformerService(config)
    service.load_models()

    image_encoder_output = None
    seed = int(config.get("seed", 42))
    chunks = []

    if phase2_send_mode == "per_chunk":
        room = 0
        service.rdma_buffer2[room] = [
            torch.empty(estimate_transformer_buffer_sizes(config)[0], dtype=torch.uint8, device="cuda"),
            torch.empty(4096, dtype=torch.uint8, device="cuda"),
        ]
        service.data_sender[room] = None
        service._phase2_remote_rooms = set()

        latent_shape_adj, num_output_frames, num_chunks = DisaggSFKVCacheManager.setup(
            service.transformer, config, latent_shape
        )
        service.scheduler.num_output_frames = num_output_frames
        service.scheduler.num_chunks = num_chunks
        service.scheduler.prepare(seed=seed, latent_shape=list(latent_shape_adj), image_encoder_output=image_encoder_output)
        infer_steps = service.scheduler.infer_steps
        try:
            for seg_idx in range(num_chunks):
                for step_index in range(infer_steps):
                    service.transformer.kv_cache_manager.current_step = step_index
                    service.scheduler.step_pre(seg_index=seg_idx, step_index=step_index, is_rerun=False)
                    service.transformer.infer(inputs)
                    service.scheduler.step_post()
                service.scheduler.step_pre(seg_index=seg_idx, step_index=infer_steps - 1, is_rerun=True)
                service.transformer.infer(inputs)
                chunk_latents = service.scheduler.stream_output
                chunks.append(chunk_latents.detach().clone())
        finally:
            DisaggSFKVCacheManager.teardown(service.transformer)
        return None, chunks

    latents = service._run_sf_denoise(inputs, latent_shape, image_encoder_output, config, seed)
    return latents, chunks


def _decode_and_save(config: dict, latents_or_chunks, phase2_send_mode: str, save_path: Path):
    from lightx2v.disagg.services.decoder import DecoderService

    decode_cfg_path = (
        "configs/disagg/wan/wan_t2v_sf_disagg_decode_per_chunk.json"
        if phase2_send_mode == "per_chunk"
        else "configs/disagg/wan/wan_t2v_sf_disagg_decode.json"
    )
    decode_cfg = set_config(
        model_path=config["model_path"],
        task="t2v",
        model_cls="wan2.1_sf",
        config_path=str(REPO_ROOT / decode_cfg_path),
    )
    decode_cfg["phase2_send_mode"] = phase2_send_mode
    decode_cfg["save_path"] = str(save_path)
    decode_cfg["seed"] = config.get("seed", 42)

    service = DecoderService(decode_cfg)
    room = 0
    service.data_receiver[room] = object()
    buffer_sizes = estimate_transformer_buffer_sizes(decode_cfg)
    service._rdma_buffers[room] = []
    for nbytes in buffer_sizes:
        service._rdma_buffers[room].append(torch.empty(nbytes, dtype=torch.uint8, device="cuda"))

    def _write_chunk(latents, meta: dict):
        latents = latents.detach().to(torch.float16).contiguous()
        latents_buf = service._rdma_buffers[room][0]
        latents_nbytes = latents.numel() * latents.element_size()
        view = torch.empty(0, dtype=latents.dtype, device=latents_buf.device)
        view.set_(latents_buf.untyped_storage(), 0, tuple(latents.shape))
        view.copy_(latents)
        meta_payload = {
            "version": 1,
            "latents_shape": list(latents.shape),
            "latents_dtype": str(latents.dtype),
            "latents_hash": None,
            **meta,
        }
        meta_bytes = json.dumps(meta_payload).encode("utf-8")
        meta_buf = service._rdma_buffers[room][1]
        meta_view = torch.empty(0, dtype=torch.uint8, device=meta_buf.device)
        meta_view.set_(meta_buf.untyped_storage(), 0, (meta_buf.numel(),))
        meta_view.zero_()
        if meta_bytes:
            meta_copy = bytearray(meta_bytes)
            meta_view[: len(meta_copy)].copy_(torch.tensor(meta_copy, dtype=torch.uint8, device=meta_buf.device))

    if phase2_send_mode == "per_chunk":
        num_chunks = len(latents_or_chunks)
        for idx, chunk_latents in enumerate(latents_or_chunks):
            _write_chunk(
                chunk_latents,
                {
                    "phase2_send_mode": "per_chunk",
                    "chunk_index": idx,
                    "num_chunks": num_chunks,
                    "is_last": idx == num_chunks - 1,
                },
            )
            result = service.process(decode_cfg)
            assert isinstance(result, dict)
            if idx < num_chunks - 1:
                assert result.get("pending_more") is True
            else:
                assert result.get("pending_more") is False
                assert Path(result["save_path"]).exists()
    else:
        _write_chunk(latents_or_chunks, {"phase2_send_mode": "full"})
        result = service.process(decode_cfg)
        assert isinstance(result, dict)
        assert result.get("pending_more") is False
        assert Path(result["save_path"]).exists()

    service.release()
    print(f"[ok] decoder saved {save_path}")


def run_integration(model_path: str, encoder_cfg: str, phase2_send_mode: str, save_path: Path):
    config = _build_config(model_path, encoder_cfg, phase2_send_mode)
    seed_all(42)
    config["seed"] = 42

    inputs = _encode_prompt(config, DEFAULT_PROMPT)
    latent_shape = inputs["latent_shape"]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    latents, chunks = _run_sf_transformer(config, inputs, latent_shape, phase2_send_mode)
    torch.cuda.synchronize()
    transformer_s = time.perf_counter() - t0

    payload = chunks if phase2_send_mode == "per_chunk" else latents
    assert payload is not None
    _decode_and_save(config, payload, phase2_send_mode, save_path)

    return {
        "phase2_send_mode": phase2_send_mode,
        "transformer_latency_s": round(transformer_s, 3),
        "output": str(save_path),
        "success": save_path.exists(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--encoder_cfg", default="configs/disagg/wan/wan_t2v_sf_disagg_encoder.json")
    parser.add_argument("--modes", default="full,per_chunk", help="Comma-separated: full,per_chunk")
    parser.add_argument("--out_dir", default="save_results/sf_disagg_smoke")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_cfg = str(REPO_ROOT / args.encoder_cfg)
    config = _build_config(args.model_path, encoder_cfg, "full")

    test_helpers()
    test_scheduler_and_model_load(config)

    summaries = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        save_path = out_dir / f"wan_t2v_sf_disagg_{mode}.mp4"
        print(f"\n=== integration: phase2_send_mode={mode} ===")
        summary = run_integration(args.model_path, encoder_cfg, mode, save_path)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    summary_path = out_dir / "smoke_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
