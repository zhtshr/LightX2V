"""Unit tests for FlowCache Self-Forcing helpers."""

import torch

from lightx2v.common.flowcache.chunkwise_cache import SFChunkwiseFeatureCache
from lightx2v.common.flowcache.kv_r1kv import FlowCacheR1KV


def test_chunkwise_cache_reuse_and_reset():
    cache = SFChunkwiseFeatureCache(rel_l1_thresh=0.05, warmup_steps=0, infer_steps=4)
    metric = torch.ones(8, 16)
    noise = torch.zeros(1, 3, 4, 8)

    cache.begin_chunk(0)
    assert not cache.should_reuse(0, 0, metric)
    cache.on_forward(0, 0, metric, noise)

    metric_close = metric + 1e-4
    assert cache.should_reuse(0, 1, metric_close)
    assert cache.cached_noise_pred(0) is not None

    metric_far = metric * 2.0
    assert not cache.should_reuse(0, 2, metric_far)


def test_r1kv_select_and_apply():
    q_len, kv_len, heads, dim = 32, 128, 4, 16
    q = torch.randn(q_len, heads, dim)
    k = torch.randn(kv_len, heads, dim)
    v = torch.randn(kv_len, heads, dim)
    clean = 96

    comp = FlowCacheR1KV(budget=48, mix_lambda=0.07, kernel_size=7)
    indices = comp.select_token_indices(k, q, clean)
    assert indices is not None
    k_out, v_out = comp.apply_token_indices(k, v, indices, clean)
    assert k_out.shape[0] == comp.budget + (kv_len - clean)
    assert v_out.shape[0] == k_out.shape[0]

    k_full, v_full, _ = comp.compress(k, v, q, clean)
    assert k_full.shape == k_out.shape


def test_blocked_similarity_matches_shape():
    k = torch.randn(256, 4, 16)
    full = _cal_similarity(k, mode="full")
    blocked = _cal_similarity(k, mode="blocked", block_size=64)
    assert full.shape == blocked.shape == (4, 256)


def test_chunked_attention_matches_full():
    from lightx2v.common.flowcache.kv_r1kv import (
        _attention_weights_sum_chunked,
        _attention_weights_sum_full,
    )

    torch.manual_seed(0)
    q_len, kv_len, heads, dim = 64, 512, 4, 32
    q = torch.randn(q_len, heads, dim)
    k = torch.randn(kv_len, heads, dim)
    full = _attention_weights_sum_full(q, k)
    chunked = _attention_weights_sum_chunked(q, k, score_chunk_size=128)
    assert torch.allclose(full, chunked, rtol=1e-4, atol=1e-5)


def test_r1kv_chunked_indices_match_full():
    torch.manual_seed(1)
    q_len, kv_len, heads, dim = 48, 256, 4, 16
    q = torch.randn(q_len, heads, dim)
    k = torch.randn(kv_len, heads, dim)
    clean = 200
    full_comp = FlowCacheR1KV(budget=96, score_chunk_size=kv_len + 1)
    chunk_comp = FlowCacheR1KV(budget=96, score_chunk_size=64)
    idx_full = full_comp.select_token_indices(k, q, clean)
    idx_chunk = chunk_comp.select_token_indices(k, q, clean)
    assert idx_full is not None and idx_chunk is not None
    assert torch.equal(idx_full[:, 0, 0], idx_chunk[:, 0, 0])


if __name__ == "__main__":
    from lightx2v.common.flowcache.kv_r1kv import _cal_similarity

    test_chunkwise_cache_reuse_and_reset()
    test_r1kv_select_and_apply()
    test_blocked_similarity_matches_shape()
    test_chunked_attention_matches_full()
    test_r1kv_chunked_indices_match_full()
    print("ok")
