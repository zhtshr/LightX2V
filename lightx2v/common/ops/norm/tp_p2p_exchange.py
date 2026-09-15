"""CUDA IPC + P2P copy for TP=2 RMS norm scalar exchange (no NCCL at norm)."""

from __future__ import annotations

import torch
import torch.distributed as dist

from lightx2v.common.ops.tp_p2p_ipc import IpcEventPair, open_peer_buffer

_MAX_ELEMS = 256 * 1024
_EXCHANGES: dict[int, TpP2pScalarExchange] = {}


class TpP2pScalarExchange:
    """Pairwise sum via IPC peer read + CUDA event sync (TP=2 only)."""

    def __init__(self, tp_group: dist.ProcessGroup) -> None:
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        if self.tp_size != 2:
            raise ValueError(f"TpP2pScalarExchange requires tp_size=2, got {self.tp_size}")
        self.local_dev = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.local_dev}")
        peer_local = 1 - self.tp_rank
        self.peer_dev = peer_local
        self.local_buf = torch.zeros(_MAX_ELEMS, dtype=torch.float32, device=self.local_dev)
        self.staging = torch.zeros(_MAX_ELEMS, dtype=torch.float32, device=self.local_dev)
        handle = self.local_buf.untyped_storage()._share_cuda_()
        handles: list[tuple | None] = [None, None]
        dist.all_gather_object(handles, handle, group=tp_group)
        peer_handle = handles[peer_local]
        assert peer_handle is not None
        self._peer_buf = open_peer_buffer(self.local_buf, peer_handle, self.peer_dev)
        self._events = IpcEventPair(tp_group, self.device)

    def sum_reduce(self, local_sum: torch.Tensor) -> torch.Tensor:
        n = local_sum.numel()
        if n > _MAX_ELEMS:
            raise RuntimeError(f"tp p2p exchange needs n<={_MAX_ELEMS}, got {n}")
        flat = local_sum.contiguous().view(-1)
        stream = torch.cuda.current_stream()
        from lightx2v.common.ops.tp_lockstep import tp_micro_step_barrier

        tp_micro_step_barrier()
        with torch.cuda.stream(stream):
            if flat.dtype == torch.float32:
                self.local_buf[:n].copy_(flat, non_blocking=True)
            else:
                self.local_buf[:n].copy_(flat.float(), non_blocking=True)
            stream.synchronize()
            self._events.record_ready(stream)
            self._events.wait_peer(stream)
            self.staging[:n].copy_(self._peer_buf[:n], non_blocking=True)
            out = self.local_buf[:n] + self.staging[:n]
        return out.view_as(local_sum).to(dtype=local_sum.dtype)


def get_tp_p2p_exchange(tp_group: dist.ProcessGroup) -> TpP2pScalarExchange:
    key = id(tp_group)
    if key not in _EXCHANGES:
        _EXCHANGES[key] = TpP2pScalarExchange(tp_group)
    return _EXCHANGES[key]


def init_tp_norm_p2p(tp_group: dist.ProcessGroup) -> TpP2pScalarExchange:
    return get_tp_p2p_exchange(tp_group)
