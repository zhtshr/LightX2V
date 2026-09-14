"""CUDA Graph helpers for TP phase overlap (pump tail during row NCCL)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class PumpTailCudaGraph:
    """Captured compute-stream subgraph from heavy step (e.g. sa_v_attn) through phase end."""

    graph: torch.cuda.CUDAGraph | None = None
    phase: int = 0
    start_name: str = ""
    captured: bool = False
    capture_error: str | None = None

    def reset(self) -> None:
        self.graph = None
        self.captured = False
        self.capture_error = None

    def try_capture(
        self,
        *,
        phase: int,
        start_name: str,
        run_tail: Callable[[], None],
        compute_stream: torch.cuda.Stream,
        warmup_iters: int = 3,
        prep: Callable[[], None] | None = None,
    ) -> bool:
        self.reset()
        self.phase = phase
        self.start_name = start_name
        try:
            for _ in range(warmup_iters):
                if prep is not None:
                    prep()
                with torch.cuda.stream(compute_stream):
                    run_tail()
            compute_stream.synchronize()
            g = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            if prep is not None:
                prep()
            with torch.cuda.graph(g, pool=pool):
                with torch.cuda.stream(compute_stream):
                    run_tail()
            compute_stream.synchronize()
            self.graph = g
            self.captured = True
            return True
        except Exception as exc:  # noqa: BLE001 — experiment path
            self.capture_error = f"{type(exc).__name__}: {exc}"
            self.captured = False
            return False

    def replay(self, compute_stream: torch.cuda.Stream) -> None:
        if not self.captured or self.graph is None:
            raise RuntimeError("pump tail cuda graph not captured")
        with torch.cuda.stream(compute_stream):
            self.graph.replay()


@dataclass
class OverlapOnceCudaGraph:
    """Best-effort capture of full overlap iteration (often fails with P2P/gloo barriers)."""

    graph: torch.cuda.CUDAGraph | None = None
    captured: bool = False
    capture_error: str | None = None

    def try_capture(self, fn: Callable[[], None], *, warmup_iters: int = 3) -> bool:
        self.graph = None
        self.captured = False
        self.capture_error = None
        try:
            for _ in range(warmup_iters):
                fn()
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            g = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(g, pool=pool):
                fn()
            torch.cuda.synchronize()
            self.graph = g
            self.captured = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.capture_error = f"{type(exc).__name__}: {exc}"
            return False

    def replay(self) -> None:
        if not self.captured or self.graph is None:
            raise RuntimeError("overlap cuda graph not captured")
        self.graph.replay()
