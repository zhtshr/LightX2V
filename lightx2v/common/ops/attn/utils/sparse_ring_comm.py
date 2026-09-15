"""Ring P2P helpers for SLA block-sparse KV exchange."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist

from .ring_comm import RingComm
from .sla_util import mean_pool


def block_map_from_pooled(pooled_qblocks: torch.Tensor, pooled_kblocks: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    """Return bool mask [B, H, Qb, Kb] of selected K blocks."""
    num_q_heads = pooled_qblocks.size(1)
    num_kv_heads = pooled_kblocks.size(1)
    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
        pooled_kblocks = pooled_kblocks.repeat_interleave(repeat_factor, dim=1)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)
    k_blocks = pooled_score.shape[-1]
    topk = min(k_blocks, max(1, int(topk_ratio * k_blocks)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    sparse_map.scatter_(-1, lut, True)
    return sparse_map


def needed_k_block_ids(sparse_map: torch.Tensor) -> torch.Tensor:
    """Union of selected K-block ids across batch/heads/query blocks."""
    active = sparse_map.any(dim=(0, 1, 2))
    return torch.nonzero(active, as_tuple=False).squeeze(-1).to(dtype=torch.int32)


def pack_kv_blocks(k: torch.Tensor, v: torch.Tensor, block_ids: torch.Tensor, blkk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack selected K/V token spans. k/v: [1, L, H, D]. Returns packed k,v and block_ids."""
    if block_ids.numel() == 0:
        empty = k.new_zeros((1, 0, k.shape[2], k.shape[3]))
        return empty, empty.clone(), block_ids
    chunks_k: list[torch.Tensor] = []
    chunks_v: list[torch.Tensor] = []
    seq_len = k.shape[1]
    for bid in block_ids.tolist():
        start = int(bid) * blkk
        end = min(start + blkk, seq_len)
        if start >= seq_len:
            continue
        chunks_k.append(k[:, start:end])
        chunks_v.append(v[:, start:end])
    if not chunks_k:
        empty = k.new_zeros((1, 0, k.shape[2], k.shape[3]))
        return empty, empty.clone(), block_ids
    return torch.cat(chunks_k, dim=1), torch.cat(chunks_v, dim=1), block_ids


def k_block_means(k: torch.Tensor, blkk: int) -> torch.Tensor:
    """k: [1, L, H, D] -> [1, H, Kb, D]."""
    return mean_pool(k.transpose(1, 2).contiguous(), blkk)


class SparseRingComm(RingComm):
    """Ring communicator with fixed-size sparse KV metadata + payload buffers."""

    def __init__(self, process_group: dist.ProcessGroup, *, max_k_blocks: int, blkk: int, heads: int, hidden: int, dtype: torch.dtype):
        super().__init__(process_group)
        self.max_k_blocks = max_k_blocks
        self.blkk = blkk
        self.heads = heads
        self.hidden = hidden
        self.dtype = dtype
        meta_elems = 2 + max_k_blocks  # packed_len, n_blocks, block_ids...
        self._meta_buf = torch.zeros(meta_elems, device=torch.cuda.current_device(), dtype=torch.int32)
        payload_elems = max_k_blocks * blkk * heads * hidden
        self._k_buf = torch.zeros(payload_elems, device=torch.cuda.current_device(), dtype=dtype)
        self._v_buf = torch.zeros(payload_elems, device=torch.cuda.current_device(), dtype=dtype)
        self._kmeans_buf: torch.Tensor | None = None

    def ensure_kmeans_buf(self, k_blocks: int) -> torch.Tensor:
        elems = k_blocks * self.heads * self.hidden
        if self._kmeans_buf is None or self._kmeans_buf.numel() != elems:
            self._kmeans_buf = torch.zeros(elems, device=torch.cuda.current_device(), dtype=self.dtype)
        return self._kmeans_buf

    def exchange_kmeans(self, k_means: torch.Tensor) -> torch.Tensor:
        """Symmetric ring exchange of pooled K block means [1,H,Kb,D]."""
        flat = k_means.contiguous().view(-1)
        recv = self.ensure_kmeans_buf(flat.numel() // (self.heads * self.hidden))
        recv_flat = recv.view(-1)
        assert flat.numel() == recv_flat.numel()
        out = self.send_recv(flat, recv_flat)
        self.commit()
        self.wait()
        return out.view_as(k_means)

    def exchange_sparse_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        send_block_ids: torch.Tensor,
        recv_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange variable-size sparse KV packs on the ring."""
        _ = recv_block_ids
        k_send, v_send, _ = pack_kv_blocks(k, v, send_block_ids, self.blkk)
        packed_send = int(k_send.shape[1])

        meta_send = self._meta_buf
        meta_send.zero_()
        meta_send[0] = packed_send
        meta_send[1] = send_block_ids.numel()
        if send_block_ids.numel():
            meta_send[2 : 2 + send_block_ids.numel()] = send_block_ids
        meta_recv = torch.empty_like(meta_send)

        group = self._process_group
        ops = [
            dist.P2POp(dist.isend, meta_send, self.send_rank, group=group),
            dist.P2POp(dist.irecv, meta_recv, self.recv_rank, group=group),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

        packed_recv = int(meta_recv[0].item())
        n_recv_ids = int(meta_recv[1].item())
        recv_ids = meta_recv[2 : 2 + n_recv_ids].clone() if n_recv_ids else recv_block_ids.new_zeros(0)

        def _flat(tensor: torch.Tensor, packed_len: int) -> torch.Tensor:
            if packed_len == 0:
                return tensor.new_zeros(1)
            return tensor.reshape(-1)

        send_elems = max(1, packed_send * self.heads * self.hidden)
        recv_elems = max(1, packed_recv * self.heads * self.hidden)
        k_send_flat = _flat(k_send, packed_send)
        v_send_flat = _flat(v_send, packed_send)
        k_recv_flat = k.new_zeros(recv_elems)
        v_recv_flat = v.new_zeros(recv_elems)

        ops = [
            dist.P2POp(dist.isend, k_send_flat, self.send_rank, group=group),
            dist.P2POp(dist.irecv, k_recv_flat, self.recv_rank, group=group),
            dist.P2POp(dist.isend, v_send_flat, self.send_rank, group=group),
            dist.P2POp(dist.irecv, v_recv_flat, self.recv_rank, group=group),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

        if packed_recv == 0:
            empty = k.new_zeros((1, 0, self.heads, self.hidden))
            return empty, empty.clone(), recv_ids
        elems = packed_recv * self.heads * self.hidden
        k_pack = k_recv_flat[:elems].view(1, packed_recv, self.heads, self.hidden)
        v_pack = v_recv_flat[:elems].view(1, packed_recv, self.heads, self.hidden)
        return k_pack, v_pack, recv_ids
