"""TP=2 row all-reduce via CUDA IPC P2P copy + IPC events (no NCCL, no gloo in hot path)."""

from __future__ import annotations

import torch
import torch.distributed as dist

from lightx2v.common.ops.tp_p2p_ipc import CudaEventWork, IpcEventPair, open_peer_buffer

_INIT_ELEMS = 8192 * 2560
_EXCHANGES: dict[int, TpP2pAllReduce] = {}


class TpP2pAllReduce:
    """Pairwise sum all-reduce: CE/DMA data path on comm_stream, event sync only."""

    def __init__(self, tp_group: dist.ProcessGroup) -> None:
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        if self.tp_size != 2:
            raise ValueError(f"TpP2pAllReduce requires tp_size=2, got {self.tp_size}")
        self.local_dev = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.local_dev}")
        peer_local = 1 - self.tp_rank
        self.peer_dev = peer_local
        self._cap = _INIT_ELEMS
        self.local_buf = torch.zeros(self._cap, dtype=torch.float32, device=self.local_dev)
        self.staging = torch.zeros(self._cap, dtype=torch.float32, device=self.local_dev)
        self._peer_buf: torch.Tensor | None = None
        self._events = IpcEventPair(tp_group, self.device)
        self._setup_ipc(peer_local)

    def _setup_ipc(self, peer_local: int) -> None:
        handle = self.local_buf.untyped_storage()._share_cuda_()
        handles: list[tuple | None] = [None, None]
        dist.all_gather_object(handles, handle, group=self.tp_group)
        peer_handle = handles[peer_local]
        assert peer_handle is not None
        self._peer_buf = open_peer_buffer(self.local_buf, peer_handle, self.peer_dev)

    def _ensure_capacity(self, n: int) -> None:
        if n <= self._cap:
            return
        cap = 1
        while cap < n:
            cap *= 2
        self._cap = cap
        self.local_buf = torch.zeros(cap, dtype=torch.float32, device=self.local_dev)
        self.staging = torch.zeros(cap, dtype=torch.float32, device=self.local_dev)
        self._setup_ipc(1 - self.tp_rank)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        *,
        async_op: bool = False,
    ) -> CudaEventWork | None:
        flat = tensor.contiguous().view(-1)
        n = flat.numel()
        self._ensure_capacity(n)
        assert self._peer_buf is not None

        done = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(stream):
            if flat.dtype == torch.float32:
                self.local_buf[:n].copy_(flat, non_blocking=True)
            else:
                self.local_buf[:n].copy_(flat.float(), non_blocking=True)
            self._events.record_ready(stream)
            self._events.wait_peer(stream)
            self.staging[:n].copy_(self._peer_buf[:n], non_blocking=True)
            reduced = self.local_buf[:n] + self.staging[:n]
            if flat.dtype == torch.float32:
                flat.copy_(reduced, non_blocking=True)
            else:
                flat.copy_(reduced.to(dtype=flat.dtype), non_blocking=True)
            done.record(stream)

        if not async_op:
            done.synchronize()
            return None
        return CudaEventWork(done)


def get_tp_p2p_allreduce(tp_group: dist.ProcessGroup) -> TpP2pAllReduce:
    key = id(tp_group)
    if key not in _EXCHANGES:
        _EXCHANGES[key] = TpP2pAllReduce(tp_group)
    return _EXCHANGES[key]
