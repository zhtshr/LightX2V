"""Chunk-wise adaptive feature cache for Self-Forcing (FlowCache Phase 1)."""

from __future__ import annotations

import torch


class SFChunkwiseFeatureCache:
    """Per-chunk denoise-step cache with relative-L1 similarity (FlowCache ChunkWiseCache)."""

    def __init__(
        self,
        *,
        rel_l1_thresh: float = 0.01,
        warmup_steps: int = 0,
        infer_steps: int = 4,
        log: bool = False,
    ) -> None:
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.warmup_steps = int(warmup_steps)
        self.infer_steps = int(infer_steps)
        self.log = bool(log)
        self.reset()

    def reset(self) -> None:
        self._prev_metric: dict[int, torch.Tensor] = {}
        self._accumulated_l1: dict[int, float] = {}
        self._prev_noise_pred: dict[int, torch.Tensor] = {}
        self._chunk_step_count: dict[int, int] = {}
        self.reuse_count = 0
        self.total_steps = 0

    def begin_chunk(self, chunk_id: int) -> None:
        self._accumulated_l1.setdefault(chunk_id, 0.0)
        self._chunk_step_count.setdefault(chunk_id, 0)

    def should_reuse(self, chunk_id: int, step_index: int, metric: torch.Tensor) -> bool:
        """Return True when transformer forward can be skipped for this chunk/step."""
        if step_index == 0 or step_index >= self.infer_steps - 1:
            self._accumulated_l1[chunk_id] = 0.0
            return False

        if self._chunk_step_count.get(chunk_id, 0) < self.warmup_steps:
            self._accumulated_l1[chunk_id] = 0.0
            return False

        prev = self._prev_metric.get(chunk_id)
        if prev is None or chunk_id not in self._prev_noise_pred:
            return False

        rel_l1 = ((metric - prev).abs().mean() / (prev.abs().mean() + 1e-8)).item()
        accumulated = self._accumulated_l1.get(chunk_id, 0.0) + rel_l1
        if accumulated < self.rel_l1_thresh:
            self._accumulated_l1[chunk_id] = accumulated
            self.reuse_count += 1
            if self.log:
                print(f"[FlowCache] reuse chunk={chunk_id} step={step_index} acc_l1={accumulated:.5f}")
            return True

        self._accumulated_l1[chunk_id] = 0.0
        if self.log:
            print(f"[FlowCache] compute chunk={chunk_id} step={step_index} acc_l1={accumulated:.5f}")
        return False

    def on_forward(
        self,
        chunk_id: int,
        step_index: int,
        metric: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> None:
        self._prev_metric[chunk_id] = metric.detach().clone()
        self._prev_noise_pred[chunk_id] = noise_pred.detach().clone()
        self._chunk_step_count[chunk_id] = self._chunk_step_count.get(chunk_id, 0) + 1
        self.total_steps += 1

    def cached_noise_pred(self, chunk_id: int) -> torch.Tensor | None:
        return self._prev_noise_pred.get(chunk_id)

    def stats(self) -> dict:
        return {
            "reuse_count": self.reuse_count,
            "total_steps": self.total_steps,
            "reuse_ratio": (self.reuse_count / self.total_steps) if self.total_steps else 0.0,
        }
