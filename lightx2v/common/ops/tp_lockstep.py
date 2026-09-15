"""CPU lockstep barriers for TP=2 infer micro-steps (P2P norm / overlap pump)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_GLOO_GROUP: dist.ProcessGroup | None = None


def _gloo_barrier_group() -> dist.ProcessGroup:
    global _GLOO_GROUP
    if _GLOO_GROUP is None:
        ranks = list(range(dist.get_world_size()))
        _GLOO_GROUP = dist.new_group(ranks=ranks, backend="gloo")
    return _GLOO_GROUP


def tp_micro_step_barrier(*, enabled: bool | None = None) -> None:
    """Align TP ranks before each infer micro-step.

    P2P norm exchange (``TpP2pScalarExchange``) requires both ranks to enter
    ``sum_reduce`` together. Without lockstep, async GPU work lets ranks drift
    and the IPC handshake deadlocks. The old workaround was an implicit barrier
    via ``get_current_model_index``'s CUDA-scalar ``if`` (~7 ms/step). We now
    barrier only at ``sum_reduce`` entry (2×/phase) plus ``stream.sync`` before
    ``record_ready`` inside P2P norm.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    if enabled is None:
        enabled = os.environ.get("LIGHTX2V_TP_MICRO_BARRIER", "1") != "0"
    if not enabled:
        return
    dist.barrier(group=_gloo_barrier_group())
