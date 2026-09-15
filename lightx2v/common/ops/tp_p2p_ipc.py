"""CUDA IPC helpers for TP=2 P2P exchange (buffers + cross-GPU events)."""

from __future__ import annotations

import torch
import torch.distributed as dist


def unpack_share_handle(handle: tuple) -> tuple:
    return (
        handle[0],
        handle[1],
        handle[2],
        handle[3],
        handle[4],
        handle[5],
        handle[6],
        handle[7],
    )


def open_peer_buffer(local_buf: torch.Tensor, peer_handle: tuple, peer_device: int) -> torch.Tensor:
    (
        _storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = unpack_share_handle(peer_handle)
    peer_storage = torch.UntypedStorage._new_shared_cuda(
        peer_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    )
    n = local_buf.numel()
    return torch.tensor([], dtype=local_buf.dtype, device=f"cuda:{peer_device}").set_(peer_storage, 0, (n,))


class IpcEventPair:
    """Inter-process CUDA event for pairwise write-ready handshake (TP=2)."""

    def __init__(self, tp_group: dist.ProcessGroup, device: torch.device) -> None:
        self.device = device
        self._local_ready = torch.cuda.Event(blocking=False, interprocess=True)
        handles: list[bytes | None] = [None, None]
        dist.all_gather_object(handles, self._local_ready.ipc_handle(), group=tp_group)
        peer_local = 1 - dist.get_rank(tp_group)
        peer_handle = handles[peer_local]
        assert peer_handle is not None
        self._peer_ready = torch.cuda.Event.from_ipc_handle(device, peer_handle)

    def record_ready(self, stream: torch.cuda.Stream) -> None:
        self._local_ready.record(stream)

    def wait_peer(self, stream: torch.cuda.Stream) -> None:
        stream.wait_event(self._peer_ready)


class CudaEventWork:
    """dist.Work-compatible waiter for async collectives on a CUDA stream."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event

    def wait(self) -> None:
        self._event.synchronize()

    def is_completed(self) -> bool:
        return self._event.query()
