"""Opt-in, lossless layout experiments for dense image-only Wan Ulysses."""

import torch
import torch.distributed as dist


def head2seq_direct(x, group=None):
    p = dist.get_world_size(group)
    seq, heads, dim = x.shape
    assert seq % p == 0
    # Preserve sequence-major layout across transport; only reorder on receipt.
    send = x.reshape(p, seq // p, heads, dim).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.transpose(0, 1).contiguous().reshape(seq // p, p * heads, dim)


_ORIGINALS = None


def install(variant):
    from lightx2v.common.ops.attn import ulysses_attn as module
    from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head

    global _ORIGINALS
    if _ORIGINALS is None:
        _ORIGINALS = (module.all2all_head2seq, module.UlyssesAttnWeight.apply)
    module.all2all_head2seq, module.UlyssesAttnWeight.apply = _ORIGINALS
    if variant in ("return", "both"):
        module.all2all_head2seq = head2seq_direct
    if variant not in ("notext", "both"):
        return
    original = module.UlyssesAttnWeight.apply

    def apply(self, q, k, v, slice_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, **kwargs):
        enabled = (
            q.ndim == 3
            and q.shape == k.shape == v.shape
            and q.shape[0] == slice_qkv_len
            and len(cu_seqlens_qkv) == 2
            and kwargs.get("img_first", True)
            and not any(kwargs.get(key, False) for key in ("use_fp8_comm", "use_fp4_comm", "use_tensor_fusion", "use_async_comm", "enable_head_parallel", "q_only_img"))
            and not getattr(self, "sparse_kv_comm", False)
            and attention_module.__class__.__name__ == "TorchSDPAWeight"
        )
        if not enabled:
            return original(self, q, k, v, slice_qkv_len, cu_seqlens_qkv, attention_module=attention_module, seq_p_group=seq_p_group, **kwargs)
        heads, dim = q.shape[-2:]
        p = dist.get_world_size(seq_p_group)
        q, k, v = [all2all_seq2head(x, seq_p_group) for x in (q, k, v)]
        seq = q.shape[0]
        cu = torch.tensor([0, seq], dtype=torch.int32)
        attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=seq, max_seqlen_kv=seq)
        out = module.all2all_head2seq(attn.reshape(seq, heads // p, dim), seq_p_group)
        return out.reshape(slice_qkv_len, heads * dim)

    module.UlyssesAttnWeight.apply = apply
