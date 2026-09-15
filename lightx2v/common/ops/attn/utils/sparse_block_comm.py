"""Block-sparse K/V helpers for Ulysses sequence parallel."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .sla_util import mean_pool


def block_map_from_pooled(
    pooled_qblocks: torch.Tensor, pooled_kblocks: torch.Tensor, topk_ratio: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_q_heads = pooled_qblocks.size(1)
    num_kv_heads = pooled_kblocks.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, (
            f"Q heads ({num_q_heads}) must be divisible by KV heads ({num_kv_heads})"
        )
        repeat_factor = num_q_heads // num_kv_heads
        pooled_kblocks = pooled_kblocks.repeat_interleave(repeat_factor, dim=1)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)
    k_blocks = pooled_score.shape[-1]
    topk = min(k_blocks, max(1, int(topk_ratio * k_blocks)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def build_sla_block_map_from_precomputed(
    q: torch.Tensor,
    k: torch.Tensor,
    img_sparse_map: torch.Tensor,
    img_lut: torch.Tensor,
    img_topk: int,
    img_seqlen: int,
    *,
    topk_ratio: float,
    BLKQ: int,
    BLKK: int,
    k_is_compact: bool = False,
    k_means_shard: torch.Tensor | None = None,
    img_pooled_kblocks: torch.Tensor | None = None,
    img_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Merge image precomputed block-map with text Q-tail block-map."""
    from .sla_util import block_map_from_pooled_tensors, get_block_map, get_block_map_pooled, mean_pool

    B, H, L, D = q.shape
    M_img = (img_seqlen + BLKQ - 1) // BLKQ
    M_total = (L + BLKQ - 1) // BLKQ
    K_total = (L + BLKK - 1) // BLKK
    K_img = (img_seqlen + BLKK - 1) // BLKK

    if M_total <= M_img:
        sm = img_sparse_map.to(torch.int8)
        if sm.shape[-1] < K_total:
            pad = torch.zeros(*sm.shape[:-1], K_total - sm.shape[-1], dtype=torch.int8, device=sm.device)
            sm = torch.cat([sm, pad], dim=-1)
        return sm, img_lut, img_topk

    q_tail = q[:, :, img_seqlen:, :]
    if img_pooled_kblocks is not None and not k_is_compact:
        _, pooled_k_full = get_block_map_pooled(q, k, BLKQ=BLKQ, BLKK=BLKK)
        pooled_q_tail = mean_pool(q_tail, BLKQ)
        sm_tail, lut_tail, topk_tail = block_map_from_pooled_tensors(
            pooled_q_tail, pooled_k_full, topk_ratio
        )
    elif k_is_compact:
        txt_len = L - img_seqlen
        if img_first:
            k_tail = k[:, :, k.shape[2] - txt_len :, :]
        else:
            k_tail = k[:, :, :txt_len, :]
        k_for_map = k_tail.new_zeros(B, H, L, D)
        k_for_map[:, :, img_seqlen:, :] = k_tail
        if k_means_shard is not None:
            kb = k_means_shard.shape[2]
            for bid in range(kb):
                s = bid * BLKK
                e = min(s + BLKK, img_seqlen)
                if s >= img_seqlen:
                    break
                mean = k_means_shard[:, :, bid, :].unsqueeze(2)
                k_for_map[:, :, s:e, :] = mean.expand(B, H, e - s, D)
        sm_tail, lut_tail, topk_tail = get_block_map(
            q_tail, k_for_map, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK
        )
    else:
        sm_tail, lut_tail, topk_tail = get_block_map(
            q_tail, k, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK
        )
    real_topk = max(img_topk, topk_tail)

    sm = torch.zeros((B, H, M_total, K_total), dtype=torch.int8, device=q.device)
    lut = torch.zeros((B, H, M_total, real_topk), dtype=torch.int64, device=q.device)
    sm[:, :, :M_img, :K_img] = img_sparse_map.to(torch.int8)
    if K_total > K_img:
        sm[:, :, :M_img, K_img:K_total] = 0
    sm[:, :, M_img:, :] = sm_tail
    lut[:, :, :M_img, :img_topk] = img_lut
    lut[:, :, M_img:, :topk_tail] = lut_tail
    return sm, lut, real_topk


def build_kv_block_table(
    active_blocks: torch.Tensor,
    *,
    global_img_seqlen: int,
    total_seqlen: int,
    txt_seqlen: int,
    img_first: bool,
    blkk: int = 64,
) -> torch.Tensor:
    """Map global K block id -> compact token offset in [img_active|txt] or [txt|img_active] layout."""
    kb_global = (total_seqlen + blkk - 1) // blkk
    kb_img = (global_img_seqlen + blkk - 1) // blkk
    kb_txt = (txt_seqlen + blkk - 1) // blkk if txt_seqlen > 0 else 0
    device = active_blocks.device
    table = torch.full((kb_global,), -1, dtype=torch.int32, device=device)
    active_set = set(int(x) for x in active_blocks.tolist())

    if img_first:
        compact_off = 0
        for bid in sorted(active_set):
            if bid >= kb_img:
                continue
            token_len = min(blkk, global_img_seqlen - bid * blkk)
            table[bid] = compact_off
            compact_off += token_len
        for bid in range(kb_img, kb_global):
            g_start = bid * blkk
            token_len = min(blkk, total_seqlen - g_start)
            table[bid] = compact_off
            compact_off += token_len
    else:
        for bid in range(kb_txt):
            table[bid] = bid * blkk
        img_compact_base = txt_seqlen
        compact_off = img_compact_base
        for bid in sorted(active_set):
            if bid >= kb_img:
                continue
            global_bid = kb_txt + bid
            token_len = min(blkk, global_img_seqlen - bid * blkk)
            table[global_bid] = compact_off
            compact_off += token_len
    return table


def global_active_k_blocks(sparse_map: torch.Tensor) -> torch.Tensor:
    active = sparse_map.any(dim=(0, 1, 2))
    return torch.nonzero(active, as_tuple=False).squeeze(-1)


def gather_global_kmeans(
    img_k: torch.Tensor,
    *,
    world_size: int,
    blkk: int,
    seq_p_group: dist.ProcessGroup,
) -> torch.Tensor:
    """img_k: [shard_seqlen, kv_heads, D]. Returns [1, kv_heads, Kb_global, D]."""
    k_means = mean_pool(img_k.transpose(0, 1).unsqueeze(0).contiguous(), blkk)
    gathered = [torch.empty_like(k_means) for _ in range(world_size)]
    dist.all_gather(gathered, k_means.contiguous(), group=seq_p_group)
    return torch.cat(gathered, dim=2)


def _local_active_seq_mask(active_blocks: torch.Tensor, cur_rank: int, shard_seqlen: int, blkk: int) -> torch.Tensor:
    kb_shard = (shard_seqlen + blkk - 1) // blkk
    block_start = cur_rank * kb_shard
    block_end = block_start + kb_shard
    mask = torch.zeros(shard_seqlen, dtype=torch.bool, device=active_blocks.device)
    for bid in active_blocks.tolist():
        if block_start <= bid < block_end:
            local_b = int(bid) - block_start
            s = local_b * blkk
            e = min(s + blkk, shard_seqlen)
            mask[s:e] = True
    return mask


def sparse_kv_ulysses_all2all(
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    active_blocks: torch.Tensor,
    *,
    world_size: int,
    q_shard_heads: int,
    kv_shard_heads: int,
    hidden_dims: int,
    seq_p_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse seq gather + head shard for image K/V.

    Input per rank: img_k/img_v [shard_seqlen, kv_heads, D] with kv_heads == world_size * kv_shard_heads.
    Output: [global_active_len, kv_shard_heads, D] scattered in global block order (active tokens only).
    """
    cur_rank = dist.get_rank(seq_p_group)
    shard_seqlen = img_k.shape[0]
    blkk = 64
    kb_shard = (shard_seqlen + blkk - 1) // blkk

    active_blocks = active_blocks.to(dtype=torch.long, device=img_k.device)
    if active_blocks.numel() == 0:
        empty = img_k.new_zeros((0, kv_shard_heads, hidden_dims))
        return empty, empty.clone()

    # Per-destination head shard: pack active local tokens.
    send_tensors: list[torch.Tensor] = []
    for dest in range(world_size):
        h0 = dest * kv_shard_heads
        h1 = h0 + kv_shard_heads
        k_slice = img_k[:, h0:h1, :]
        v_slice = img_v[:, h0:h1, :]
        seq_mask = _local_active_seq_mask(active_blocks, cur_rank, shard_seqlen, blkk)
        if seq_mask.any():
            send_tensors.append(k_slice[seq_mask].contiguous())
        else:
            send_tensors.append(k_slice.new_zeros((0, kv_shard_heads, hidden_dims)))

    send_counts = torch.tensor([t.shape[0] for t in send_tensors], device=img_k.device, dtype=torch.int64)
    counts_matrix = torch.empty(world_size, world_size, device=img_k.device, dtype=torch.int64)
    dist.all_gather(
        list(counts_matrix.unbind(0)),
        send_counts,
        group=seq_p_group,
    )
    recv_counts = counts_matrix[:, cur_rank].tolist()
    recv_tensors = [
        img_k.new_zeros((int(recv_counts[src]), kv_shard_heads, hidden_dims))
        for src in range(world_size)
    ]
    dist.all_to_all(recv_tensors, send_tensors, group=seq_p_group)

    total_len = 0
    for bid in active_blocks.tolist():
        src_rank = int(bid) // kb_shard
        local_b = int(bid) - src_rank * kb_shard
        token_len = min(blkk, shard_seqlen - local_b * blkk)
        total_len += token_len

    out_k = img_k.new_zeros((total_len, kv_shard_heads, hidden_dims))
    out_v = img_v.new_zeros((total_len, kv_shard_heads, hidden_dims))
    offset = 0
    recv_by_src = {src: recv_tensors[src] for src in range(world_size)}
    recv_offsets = {src: 0 for src in range(world_size)}
    for bid in active_blocks.tolist():
        src_rank = int(bid) // kb_shard
        local_b = int(bid) - src_rank * kb_shard
        token_len = min(blkk, shard_seqlen - local_b * blkk)
        chunk_k = recv_by_src[src_rank][recv_offsets[src_rank] : recv_offsets[src_rank] + token_len]
        chunk_v = recv_tensors[src_rank][recv_offsets[src_rank] : recv_offsets[src_rank] + token_len]
        out_k[offset : offset + token_len] = chunk_k
        out_v[offset : offset + token_len] = chunk_v
        recv_offsets[src_rank] += token_len
        offset += token_len
    return out_k, out_v


def expand_sparse_kv_to_dense(
    sparse_k: torch.Tensor,
    sparse_v: torch.Tensor,
    active_blocks: torch.Tensor,
    *,
    global_seqlen: int,
    world_size: int,
    shard_seqlen: int,
    blkk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse active tokens back to dense [global_seqlen, H, D] for SLA kernel."""
    kv_shard_heads = sparse_k.shape[1]
    hidden_dims = sparse_k.shape[2]
    dense_k = sparse_k.new_zeros((global_seqlen, kv_shard_heads, hidden_dims))
    dense_v = sparse_v.new_zeros((global_seqlen, kv_shard_heads, hidden_dims))
    kb_shard = (shard_seqlen + blkk - 1) // blkk
    offset = 0
    for bid in active_blocks.tolist():
        src_rank = int(bid) // kb_shard
        local_b = int(bid) - src_rank * kb_shard
        token_len = min(blkk, shard_seqlen - local_b * blkk)
        g_start = int(bid) * blkk
        g_end = min(g_start + token_len, global_seqlen)
        n = g_end - g_start
        dense_k[g_start:g_end] = sparse_k[offset : offset + n]
        dense_v[g_start:g_end] = sparse_v[offset : offset + n]
        offset += token_len
    return dense_k, dense_v


def fill_inactive_from_kmeans(
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    k_means_global: torch.Tensor,
    active_blocks: torch.Tensor,
    *,
    global_seqlen: int,
    cur_rank: int,
    kv_shard_heads: int,
    blkk: int = 64,
) -> None:
    """Fill non-active K blocks with per-block means so SLA block-map stays consistent."""
    active_set = set(int(x) for x in active_blocks.tolist())
    kb_global = k_means_global.shape[2]
    heads, dim = dense_k.shape[1], dense_k.shape[2]
    h0 = cur_rank * kv_shard_heads
    h1 = h0 + kv_shard_heads
    for bid in range(kb_global):
        if bid in active_set:
            continue
        s = bid * blkk
        e = min(s + blkk, global_seqlen)
        if s >= global_seqlen:
            break
        mean = k_means_global[0, h0:h1, bid, :].unsqueeze(0)
        dense_k[s:e] = mean.expand(e - s, heads, dim)
        dense_v[s:e] = mean.expand(e - s, heads, dim)
