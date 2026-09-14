"""Self-Forcing helpers for disaggregated Wan inference."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from lightx2v_platform.base.global_var import AI_DEVICE


def is_sf_model(config: Dict[str, Any]) -> bool:
    return str(config.get("model_cls", "")) == "wan2.1_sf"


def get_phase2_send_mode(config: Dict[str, Any]) -> str:
    """Return ``full`` (entire latents) or ``per_chunk`` (SF chunk streaming)."""
    disagg_cfg = config.get("disagg_config") or {}
    mode = config.get("phase2_send_mode", disagg_cfg.get("phase2_send_mode", "full"))
    mode = str(mode).strip().lower()
    if mode not in {"full", "per_chunk"}:
        raise ValueError(f"Unsupported phase2_send_mode={mode!r}, expected 'full' or 'per_chunk'")
    return mode


def is_per_chunk_phase2(config: Dict[str, Any]) -> bool:
    return is_sf_model(config) and get_phase2_send_mode(config) == "per_chunk"


class DisaggSFKVCacheManager:
    """Initialize and attach Self-Forcing KV caches on a WanSFModel instance."""

    @staticmethod
    def setup(transformer, config: Dict[str, Any], latent_shape: List[int]) -> Tuple[List[int], int, int]:
        from lightx2v.common.kvcache import KVCacheManager

        device = torch.device(AI_DEVICE)
        # PP owns a layer shard: SF block_idx is local 0..len(blocks)-1, so KV
        # must be sized to local blocks (not full num_layers) to avoid 2× HBM.
        num_layers_orig = config.get("num_layers")
        if config.get("pipeline_parallel"):
            local_layers = len(transformer.transformer_weights.blocks)
            config["num_layers"] = local_layers
        try:
            kv_manager = KVCacheManager(
                config=config,
                device=device,
                sp_group=getattr(transformer, "seq_p_group", None),
            )
            kv_manager._create_kv_caches(list(latent_shape))
        finally:
            if num_layers_orig is not None:
                config["num_layers"] = num_layers_orig

        transformer.kv_cache_manager = kv_manager
        transformer.transformer_infer.kv_cache_manager = kv_manager

        adjusted_shape = list(latent_shape)
        adjusted_shape[1] = int(kv_manager.num_output_frames)

        num_frame_per_chunk = int(config.get("ar_config", {}).get("num_frame_per_chunk", 3))
        num_chunks = int(kv_manager.num_output_frames) // num_frame_per_chunk
        return adjusted_shape, int(kv_manager.num_output_frames), num_chunks

    @staticmethod
    def teardown(transformer) -> None:
        kv_manager = getattr(transformer, "kv_cache_manager", None)
        if kv_manager is not None and hasattr(kv_manager, "save_calibration"):
            kv_manager.save_calibration()
        if hasattr(transformer, "transformer_infer"):
            transformer.transformer_infer.kv_cache_manager = None
        transformer.kv_cache_manager = None


def refresh_sf_scheduler(scheduler, config: Dict[str, Any]) -> None:
    scheduler.refresh_from_config(config)
    ar = config.get("ar_config", {})
    scheduler.num_frame_per_chunk = int(ar.get("num_frame_per_chunk", 3))
    if "denoising_step_list" in ar:
        scheduler.denoising_step_list = [float(t) for t in ar["denoising_step_list"]]
        scheduler.infer_steps = len(scheduler.denoising_step_list)
    elif "timesteps_index" in ar:
        scheduler.timesteps_index = list(ar["timesteps_index"])
        scheduler.infer_steps = len(scheduler.timesteps_index)
        scheduler._mode = "index"
    scheduler.num_output_frames = int(config.get("num_output_frames", config.get("target_video_length", 81)))
