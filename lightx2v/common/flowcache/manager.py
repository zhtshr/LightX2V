"""FlowCache manager for Wan Self-Forcing."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from .chunkwise_cache import SFChunkwiseFeatureCache
from .kv_r1kv import FlowCacheR1KV


class SFFlowCacheManager:
    """Coordinates chunkwise feature caching and KV compression for ``wan2.1_sf``."""

    def __init__(self, config: dict[str, Any]) -> None:
        ar = config.get("ar_config", {})
        fc = ar.get("flowcache", {})
        self.enabled = bool(fc.get("enable", False))
        self.enable_feature_cache = bool(fc.get("enable_feature_cache", True))
        self.enable_kv_compress = bool(fc.get("enable_kv_compress", True))
        self.kv_compress_once = bool(fc.get("kv_compress_once", True))
        self.log = bool(fc.get("log", False))

        infer_steps = int(config.get("infer_steps", 4))
        self.feature_cache = SFChunkwiseFeatureCache(
            rel_l1_thresh=float(fc.get("rel_l1_thresh", 0.01)),
            warmup_steps=int(fc.get("warmup_steps", 0)),
            infer_steps=infer_steps,
            log=self.log,
        ) if self.enabled and self.enable_feature_cache else None

        self.kv_budget_chunks = int(fc.get("kv_budget_chunks", 4))
        self.mix_lambda = float(fc.get("mix_lambda", 0.07))
        self.compress_kernel_size = int(fc.get("compress_kernel_size", 7))
        self.similarity_mode = str(fc.get("similarity_mode", "blocked"))
        self.similarity_block_size = int(fc.get("similarity_block_size", 256))
        self.score_chunk_size = int(fc.get("score_chunk_size", 4096))
        self._kv_compressor: FlowCacheR1KV | None = None
        self._completed_chunks: set[int] = set()
        self._last_query_states: dict[int, torch.Tensor] = {}
        self._kv_prefix_compressed = False
        self.kv_compress_count = 0

    @property
    def uses_feature_cache(self) -> bool:
        return self.enabled and self.enable_feature_cache and self.feature_cache is not None

    @property
    def uses_kv_compress(self) -> bool:
        return self.enabled and self.enable_kv_compress

    def reset(self) -> None:
        if self.feature_cache is not None:
            self.feature_cache.reset()
        self._completed_chunks.clear()
        self._last_query_states.clear()
        self._kv_prefix_compressed = False
        self.kv_compress_count = 0

    @torch.no_grad()
    def compute_metric(self, model, inputs) -> torch.Tensor:
        """Patch-embedded latent metric (FlowCache feature proxy for Wan SF)."""
        scheduler = model.scheduler
        x = scheduler.latents_input
        weights = model.pre_weight
        embedded = weights.patch_embedding.apply(x.unsqueeze(0))
        embedded = embedded.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        return embedded.float()

    def begin_chunk(self, chunk_id: int) -> None:
        if self.feature_cache is not None:
            self.feature_cache.begin_chunk(chunk_id)

    def should_skip_transformer(self, chunk_id: int, step_index: int, metric: torch.Tensor, *, is_rerun: bool) -> bool:
        if not self.enabled or self.feature_cache is None or is_rerun:
            return False
        return self.feature_cache.should_reuse(chunk_id, step_index, metric)

    def apply_cached_noise_pred(self, scheduler, chunk_id: int) -> None:
        cached = self.feature_cache.cached_noise_pred(chunk_id)
        if cached is None:
            raise RuntimeError(f"FlowCache reuse requested but no cached noise_pred for chunk {chunk_id}")
        seg_start = chunk_id * scheduler.num_frame_per_chunk
        seg_end = min((chunk_id + 1) * scheduler.num_frame_per_chunk, scheduler.num_output_frames)
        scheduler.noise_pred[:, seg_start:seg_end] = cached

    def on_forward(
        self,
        chunk_id: int,
        step_index: int,
        metric: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> None:
        if self.feature_cache is not None:
            self.feature_cache.on_forward(chunk_id, step_index, metric, noise_pred)

    def store_query_state(self, layer_id: int, query: torch.Tensor) -> None:
        if self.uses_kv_compress:
            self._last_query_states[int(layer_id)] = query.detach()

    def mark_chunk_completed(self, chunk_id: int) -> None:
        if self.uses_kv_compress:
            self._completed_chunks.add(int(chunk_id))

    def maybe_compress_kv(self, model, chunk_id: int) -> bool:
        """Compress denoised clean KV prefix after chunk ``chunk_id`` completes."""
        if not self.uses_kv_compress:
            return False

        if self.kv_compress_once and self._kv_prefix_compressed:
            return False

        mgr = getattr(model, "kv_cache_manager", None)
        if mgr is None:
            return False

        ar = model.config.get("ar_config", {})
        num_frame_per_chunk = int(ar.get("num_frame_per_chunk", 3))
        tokens_per_chunk = mgr.frame_seq_length * num_frame_per_chunk
        if tokens_per_chunk <= 0:
            return False

        completed = sorted(self._completed_chunks)
        keep_recent = max(1, self.kv_budget_chunks - 1)
        if len(completed) <= keep_recent:
            return False

        compress_chunk_count = len(completed) - keep_recent
        clean_tokens = compress_chunk_count * tokens_per_chunk
        budget_tokens = max(tokens_per_chunk, (self.kv_budget_chunks - 1) * tokens_per_chunk)
        if clean_tokens <= budget_tokens:
            return False

        kv_cache = mgr.self_attn_kv_cache
        if not hasattr(kv_cache, "compress_prefix"):
            logger.warning("FlowCache KV compression requires RollingKVCachePool.compress_prefix")
            return False

        query_layer0 = self._last_query_states.get(0)
        if query_layer0 is None:
            logger.warning("FlowCache KV compression skipped: missing layer-0 query states")
            return False

        if self._kv_compressor is None or self._kv_compressor.budget != budget_tokens:
            self._kv_compressor = FlowCacheR1KV(
                budget=budget_tokens,
                mix_lambda=self.mix_lambda,
                kernel_size=self.compress_kernel_size,
                similarity_mode=self.similarity_mode,
                similarity_block_size=self.similarity_block_size,
                score_chunk_size=self.score_chunk_size,
            )

        active_start = clean_tokens
        compressed = kv_cache.compress_prefix(
            clean_end=clean_tokens,
            active_start=active_start,
            compressor=self._kv_compressor,
            query_states=query_layer0,
        )
        if compressed:
            self._kv_prefix_compressed = True
            self.kv_compress_count += 1
            if self.log:
                logger.info(
                    "[FlowCache] KV compressed after chunk {}: clean_tokens {} -> budget {} (layer0 score, broadcast)",
                    chunk_id,
                    clean_tokens,
                    budget_tokens,
                )
        return compressed

    def log_stats(self) -> None:
        if self.feature_cache is not None:
            stats = self.feature_cache.stats()
            logger.info(
                "[FlowCache] feature cache reuse {}/{} ({:.1%})",
                stats["reuse_count"],
                stats["total_steps"],
                stats["reuse_ratio"],
            )
