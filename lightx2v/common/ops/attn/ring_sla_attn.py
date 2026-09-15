"""Ring sequence-parallel self-attention with optional SLA block-sparse KV comm."""

from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .ring_attn import RingAttnHelper, _update_out_and_lse
from .template import AttnWeightTemplate
from .utils.sparse_ring_comm import (
    SparseRingComm,
    block_map_from_pooled,
    k_block_means,
    needed_k_block_ids,
)
from .utils.sla_util import mean_pool

try:
    import flash_attn
except ImportError:
    logger.info("flash_attn not found for ring_sla")
    flash_attn = None


@ATTN_WEIGHT_REGISTER("ring_sla")
class RingSlaAttnWeight(AttnWeightTemplate):
    """Ring SP with SLA-guided sparse KV exchange."""

    sparse_comm = True
    sparsity_ratio = 0.8
    BLKQ = 64
    BLKK = 64

    def __init__(self) -> None:
        self.config = {}
        self.helper = RingAttnHelper()

    def _topk_ratio(self, attention_module) -> float:
        if attention_module is not None and hasattr(attention_module, "topk"):
            return float(attention_module.topk)
        return 1.0 - self.sparsity_ratio

    def _ring_flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if k.shape[1] == 0:
            out = torch.zeros_like(q)
            lse = torch.full((q.shape[0], q.shape[1], q.shape[2]), float("-inf"), device=q.device, dtype=torch.float32)
            return out, lse
        block_out, block_lse, _, _ = flash_attn.flash_attn_interface._flash_attn_forward(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        return block_out, block_lse

    def _update(self, out, lse, block_out, block_lse):
        if out is None:
            return block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        return _update_out_and_lse(out, lse, block_out, block_lse)

    def _exchange_need_indices(self, comm: SparseRingComm, need_from_left: torch.Tensor) -> torch.Tensor:
        world = comm.world_size
        send_need = torch.full((comm.max_k_blocks + 2,), -1, device=need_from_left.device, dtype=torch.int32)
        send_need[0] = need_from_left.numel()
        if need_from_left.numel():
            send_need[1 : 1 + need_from_left.numel()] = need_from_left
        gathered = [torch.empty_like(send_need) for _ in range(world)]
        dist.all_gather(gathered, send_need, group=comm._process_group)
        recv_need = gathered[(comm.rank + 1) % world]
        n_right = int(recv_need[0].item())
        if n_right <= 0:
            return need_from_left.new_zeros((0,), dtype=torch.int32)
        return recv_need[1 : 1 + n_right].clone()

    def _sparse_rotate_kv(
        self,
        comm: SparseRingComm,
        q: torch.Tensor,
        k_hold: torch.Tensor,
        v_hold: torch.Tensor,
        topk_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recv_means = comm.exchange_kmeans(k_block_means(k_hold, self.BLKK))
        q_means = mean_pool(q.transpose(1, 2).contiguous(), self.BLKQ)
        sparse_map = block_map_from_pooled(q_means, recv_means, topk_ratio)
        need_from_left = needed_k_block_ids(sparse_map).clamp(min=0, max=comm.max_k_blocks - 1)
        send_block_ids = self._exchange_need_indices(comm, need_from_left)
        k_recv, v_recv, _ = comm.exchange_sparse_kv(k_hold, v_hold, send_block_ids, need_from_left)
        return k_recv, v_recv

    def apply(
        self,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        seq_p_group=None,
        use_fp8_comm=False,
        use_fp4_comm=False,
        use_tensor_fusion=False,
        enable_head_parallel=False,
        **kwargs,
    ):
        assert flash_attn is not None, "ring_sla requires flash_attn"
        assert not enable_head_parallel, "ring_sla does not support head parallel"
        assert not (use_fp8_comm or use_fp4_comm), "ring_sla sparse path does not support fp8/fp4 comm yet"

        topk_ratio = self._topk_ratio(attention_module)
        world_size = dist.get_world_size(seq_p_group)

        img_qkv_len = slice_qkv_len
        txt_qkv_len, txt_mask_len = self.helper._get_text_lengths(cu_seqlens_qkv, img_qkv_len)

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        heads, hidden_dims = k.shape[-2], k.shape[-1]
        k_blocks = (img_qkv_len + self.BLKK - 1) // self.BLKK

        img_q, img_k, img_v = q[:, :img_qkv_len], k[:, :img_qkv_len], v[:, :img_qkv_len]
        txt_q, txt_k, txt_v = (
            q[:, img_qkv_len : img_qkv_len + txt_qkv_len],
            k[:, img_qkv_len : img_qkv_len + txt_qkv_len],
            v[:, img_qkv_len : img_qkv_len + txt_qkv_len],
        )

        if len(cu_seqlens_qkv) == 3:
            q = torch.cat((img_q, txt_q), dim=1)
        else:
            q = img_q
        k_local, v_local = img_k, img_v

        comm = SparseRingComm(
            seq_p_group,
            max_k_blocks=max(k_blocks, 1),
            blkk=self.BLKK,
            heads=heads,
            hidden=hidden_dims,
            dtype=k_local.dtype,
        )

        k_hold, v_hold = k_local, v_local
        out, lse = None, None

        for step in range(world_size):
            is_last = step + 1 == world_size
            if is_last and txt_qkv_len:
                k_step = torch.cat((k_hold, txt_k), dim=1)
                v_step = torch.cat((v_hold, txt_v), dim=1)
            else:
                k_step, v_step = k_hold, v_hold

            block_out, block_lse = self._ring_flash(q, k_step, v_step)
            if k_step.shape[1] > 0:
                out, lse = self._update(out, lse, block_out, block_lse)

            if is_last:
                break

            if self.sparse_comm:
                k_hold, v_hold = self._sparse_rotate_kv(comm, q, k_hold, v_hold, topk_ratio)
            else:
                next_k = comm.send_recv(k_hold)
                next_v = comm.send_recv(v_hold)
                comm.commit()
                comm.wait()
                k_hold, v_hold = next_k, next_v

        attn1 = out.to(GET_DTYPE()).squeeze(0).reshape(img_qkv_len + txt_qkv_len, -1)

        if txt_mask_len > 0:
            attn2, *_ = flash_attn.flash_attn_interface._flash_attn_forward(
                q[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
                k_step[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
                v_step[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
                dropout_p=0.0,
                softmax_scale=q.shape[-1] ** (-0.5),
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
            attn2 = attn2.to(GET_DTYPE()).squeeze(0).reshape((txt_mask_len - txt_qkv_len), -1)
            attn1 = torch.cat([attn1, attn2], dim=0)

        return attn1
