"""Importance–redundancy KV compression (FlowCache / R1KV, Phase 2)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _cal_similarity_full(key_states: torch.Tensor) -> torch.Tensor:
    """Per-token redundancy from full N×N key cosine similarity."""
    k = key_states.permute(1, 0, 2)
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    for h in range(similarity_cos.shape[0]):
        similarity_cos[h].fill_diagonal_(0.0)
    return similarity_cos.mean(dim=1).softmax(dim=-1)


def _cal_similarity_blocked(key_states: torch.Tensor, block_size: int) -> torch.Tensor:
    """Blocked redundancy proxy: O(heads × num_blocks × block_size²) instead of O(heads × N²)."""
    k = key_states.permute(1, 0, 2)
    n_heads, n_tokens, _ = k.shape
    block_size = max(32, int(block_size))
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    scores = torch.empty(n_heads, n_tokens, device=k.device, dtype=k.dtype)
    for start in range(0, n_tokens, block_size):
        end = min(start + block_size, n_tokens)
        blk = k_norm[:, start:end]
        sim = torch.matmul(blk, blk.transpose(-1, -2))
        for h in range(n_heads):
            sim[h].fill_diagonal_(0.0)
        scores[:, start:end] = sim.mean(dim=-1)
    return scores.softmax(dim=-1)


def _cal_similarity(key_states: torch.Tensor, *, mode: str = "blocked", block_size: int = 256) -> torch.Tensor:
    if mode == "full":
        return _cal_similarity_full(key_states)
    if mode == "blocked":
        return _cal_similarity_blocked(key_states, block_size)
    raise ValueError(f"unknown similarity mode: {mode!r}")


def _compute_attention_scores(query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
    """Attention logits [kv_heads, q_len, kv_len] (full materialization, for tests)."""
    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states.shape
    query_group_size = q_heads // kv_heads
    if query_group_size != 1:
        raise NotImplementedError("FlowCache KV compression requires query_group_size == 1 for Wan SF.")
    attn_chunks = []
    chunk_size = max(q_len, 1)
    q = query_states.transpose(0, 1)
    for i in range(0, kv_len, chunk_size):
        end_i = min(i + chunk_size, kv_len)
        k_chunk = key_states[i:end_i]
        attn_chunk = torch.bmm(q, k_chunk.permute(1, 2, 0)) / math.sqrt(head_dim)
        attn_chunks.append(attn_chunk)
    return torch.cat(attn_chunks, dim=2)


def _attention_weights_sum_full(query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
    """Reference: mean_q softmax(QK^T/sqrt(d)) per K position."""
    kv_len = key_states.shape[0]
    attn_weights = _compute_attention_scores(query_states, key_states)
    return (
        F.softmax(attn_weights[:, :, :kv_len], dim=-1)
        .mean(dim=-2)
        .to(query_states.dtype)
    )


def _attention_weights_sum_chunked(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    score_chunk_size: int,
) -> torch.Tensor:
    """Chunked equivalent of ``_attention_weights_sum_full`` with O(heads × q_len × chunk) peak memory."""
    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states.shape
    if q_heads // kv_heads != 1:
        raise NotImplementedError("FlowCache KV compression requires query_group_size == 1 for Wan SF.")

    q = query_states.transpose(0, 1).float()
    chunk_size = max(256, int(score_chunk_size))

    max_logits = torch.full((kv_heads, q_len), float("-inf"), device=q.device, dtype=torch.float32)
    for i in range(0, kv_len, chunk_size):
        end_i = min(i + chunk_size, kv_len)
        logits = torch.bmm(q, key_states[i:end_i].permute(1, 2, 0).float()) / math.sqrt(head_dim)
        max_logits = torch.maximum(max_logits, logits.max(dim=-1).values)

    denom = torch.zeros((kv_heads, q_len), device=q.device, dtype=torch.float32)
    for i in range(0, kv_len, chunk_size):
        end_i = min(i + chunk_size, kv_len)
        logits = torch.bmm(q, key_states[i:end_i].permute(1, 2, 0).float()) / math.sqrt(head_dim)
        exp_logits = torch.exp(logits - max_logits.unsqueeze(-1))
        denom += exp_logits.sum(dim=-1)

    attn_sum = torch.zeros((kv_heads, kv_len), device=q.device, dtype=torch.float32)
    inv_q_len = 1.0 / max(q_len, 1)
    denom_safe = denom.clamp(min=1e-8)
    for i in range(0, kv_len, chunk_size):
        end_i = min(i + chunk_size, kv_len)
        logits = torch.bmm(q, key_states[i:end_i].permute(1, 2, 0).float()) / math.sqrt(head_dim)
        exp_logits = torch.exp(logits - max_logits.unsqueeze(-1))
        attn_sum[:, i:end_i] = (exp_logits / denom_safe.unsqueeze(-1)).sum(dim=1) * inv_q_len

    return attn_sum.to(query_states.dtype)


class FlowCacheR1KV:
    def __init__(
        self,
        *,
        budget: int,
        mix_lambda: float = 0.07,
        kernel_size: int = 7,
        similarity_mode: str = "blocked",
        similarity_block_size: int = 256,
        score_chunk_size: int = 4096,
    ) -> None:
        self.budget = int(budget)
        self.mix_lambda = float(mix_lambda)
        self.kernel_size = int(kernel_size)
        self.similarity_mode = str(similarity_mode)
        self.similarity_block_size = int(similarity_block_size)
        self.score_chunk_size = int(score_chunk_size)

    def select_token_indices(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        clean_tokens: int,
    ) -> torch.Tensor | None:
        """Score clean prefix on layer-0 K; return gather indices or None if no compression."""
        if clean_tokens <= self.budget:
            return None

        head_dim = query_states.shape[-1]
        k_clean = key_states[:clean_tokens]

        attn_weights_sum = _attention_weights_sum_chunked(
            query_states,
            k_clean,
            score_chunk_size=self.score_chunk_size,
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        similarity_cos = _cal_similarity(
            k_clean,
            mode=self.similarity_mode,
            block_size=self.similarity_block_size,
        )
        final_score = attn_cache * self.mix_lambda - similarity_cos * (1.0 - self.mix_lambda)

        num_to_keep = min(self.budget, clean_tokens)
        indices = final_score.topk(num_to_keep, dim=-1).indices
        return indices.unsqueeze(-1).expand(-1, -1, head_dim).permute(1, 0, 2)

    def apply_token_indices(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        indices: torch.Tensor,
        clean_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V with precomputed indices; keep suffix ``[clean_tokens:]`` unchanged."""
        k_comp = key_states[:clean_tokens].gather(dim=0, index=indices)
        v_comp = value_states[:clean_tokens].gather(dim=0, index=indices)
        k_out = torch.cat([k_comp, key_states[clean_tokens:]], dim=0)
        v_out = torch.cat([v_comp, value_states[clean_tokens:]], dim=0)
        return k_out, v_out

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        clean_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress prefix ``[:clean_tokens]`` to ``budget`` tokens; keep suffix unchanged."""
        indices = self.select_token_indices(key_states, query_states, clean_tokens)
        if indices is None:
            return key_states, value_states, torch.arange(clean_tokens, device=key_states.device)
        k_out, v_out = self.apply_token_indices(key_states, value_states, indices, clean_tokens)
        return k_out, v_out, indices[:, 0, 0]
