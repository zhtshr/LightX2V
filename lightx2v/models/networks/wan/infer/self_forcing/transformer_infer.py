import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.triton_ops import causal_rope_apply_triton
from lightx2v.models.networks.wan.infer.utils import causal_rope_apply
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class _KVStoreProfiler:
    """CUDA event accumulator for self/cross-attn KV path breakdown."""

    def __init__(self) -> None:
        self.store_kv_ms = 0.0
        self.sp_a2a_ms = 0.0
        self.kv_read_ms = 0.0
        self.self_attn_ms = 0.0
        self.cross_attn_ms = 0.0
        self.cross_kv_store_ms = 0.0
        self.store_kv_calls = 0
        self.skipped_store_kv_calls = 0
        self.cross_attn_calls = 0
        self.cross_kv_store_calls = 0
        self._pending: list[tuple[torch.cuda.Event, torch.cuda.Event, str]] = []

    def record(self, start: torch.cuda.Event, end: torch.cuda.Event, kind: str) -> None:
        self._pending.append((start, end, kind))

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for start, end, kind in self._pending:
            ms = float(start.elapsed_time(end))
            if kind == "store_kv":
                self.store_kv_ms += ms
                self.store_kv_calls += 1
            elif kind == "sp_a2a":
                self.sp_a2a_ms += ms
            elif kind == "kv_read":
                self.kv_read_ms += ms
            elif kind == "self_attn":
                self.self_attn_ms += ms
            elif kind == "cross_attn":
                self.cross_attn_ms += ms
                self.cross_attn_calls += 1
            elif kind == "cross_kv_store":
                self.cross_kv_store_ms += ms
                self.cross_kv_store_calls += 1
        self._pending.clear()

    def reset(self) -> None:
        self.__init__()

    def snapshot(self) -> dict:
        return {
            "store_kv_ms": round(self.store_kv_ms, 3),
            "sp_a2a_ms": round(self.sp_a2a_ms, 3),
            "kv_read_ms": round(self.kv_read_ms, 3),
            "self_attn_ms": round(self.self_attn_ms, 3),
            "cross_attn_ms": round(self.cross_attn_ms, 3),
            "cross_kv_store_ms": round(self.cross_kv_store_ms, 3),
            "store_kv_calls": self.store_kv_calls,
            "skipped_store_kv_calls": self.skipped_store_kv_calls,
            "cross_attn_calls": self.cross_attn_calls,
            "cross_kv_store_calls": self.cross_kv_store_calls,
        }


class WanSFTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        ar = config.get("ar_config", {})
        self.num_frame_per_chunk = ar.get("num_frame_per_chunk", 3)
        self._ar_kv_offload: bool = bool(ar.get("kv_offload", False))
        self._profile_kv_store = bool(ar.get("profile_kv_store", False))
        self._kv_store_profiler = _KVStoreProfiler() if self._profile_kv_store else None
        self.flowcache_manager = None
        if self._ar_kv_offload:
            self.infer_block_func = self.infer_block_with_kvoffload
        else:
            self.infer_block_func = self.infer_block_with_kvcache

        # Weight CPU↔GPU block streaming (WeightAsyncStreamManager) — independent of
        # ``infer_block_func`` (KV cache CPU offload vs on-GPU).
        self._weight_offload_block_compute = False
        cpu_off = self.config.get("cpu_offload", False)
        gran = self.config.get("offload_granularity", "block")
        if cpu_off and gran == "block":
            self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")
            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load:
                self.offload_manager.init_lazy_load(
                    num_workers=self.config.get("num_disk_workers", 4),
                )
            self.infer_func = self.infer_with_kvcache_blocks_offload
            self._weight_offload_block_compute = True
        elif cpu_off:
            logger.warning(
                "[WanLingbotFastTransformerInfer] cpu_offload with offload_granularity={!r} does not use "
                "block weight streaming; falling back to infer_with_kvcache. Use offload_granularity='block' "
                "to enable infer_with_kvcache_blocks_offload (WeightAsyncStreamManager).",
                gran,
            )
            self.infer_func = self.infer_with_kvcache
        else:
            self.infer_func = self.infer_with_kvcache

        if self.config.get("causal_rope_type", "torch") == "triton":
            self.causal_rope_apply_func = causal_rope_apply_triton
        else:
            self.causal_rope_apply_func = causal_rope_apply

    def _calculate_q_k_len(self, q, k_lens):
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k

    def reset_kv_store_profile(self) -> None:
        if self._kv_store_profiler is not None:
            self._kv_store_profiler.reset()

    def finalize_kv_store_profile(self) -> dict | None:
        if self._kv_store_profiler is None:
            return None
        self._kv_store_profiler.finalize()
        return self._kv_store_profiler.snapshot()

    def _cuda_mark(self, kind: str):
        if self._kv_store_profiler is None:
            return None, None, kind
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end, kind

    def _cuda_end(self, start, end, kind: str) -> None:
        if self._kv_store_profiler is None or start is None or end is None:
            return
        end.record()
        self._kv_store_profiler.record(start, end, kind)

    def _should_persist_self_attn_kv(self) -> bool:
        ar = self.config.get("ar_config", {})
        if ar.get("store_kv_only_on_rerun"):
            return bool(getattr(self.scheduler, "is_rerun", False))
        return True

    def _sp_stripe_kv(self) -> bool:
        mgr = getattr(self, "kv_cache_manager", None)
        return bool(mgr is not None and getattr(mgr, "sp_stripe_kv", False))

    def _stripe_full_q(self) -> bool:
        """Full activation + striped KV: skip Q all_gather in stripe attention."""
        import os

        if not self._sp_stripe_kv():
            return False
        parallel = self.config.get("parallel") or {}
        if isinstance(parallel, dict) and parallel.get("stripe_full_q"):
            return True
        return os.environ.get("LIGHTX2V_STRIPE_FULL_Q", "0") == "1"

    def _stripe_hier(self) -> bool:
        """P=4 hierarchical: pair-exchange KV, then cross-exchange Q."""
        import os

        if not self._sp_stripe_kv() or self._stripe_full_q():
            return False
        parallel = self.config.get("parallel") or {}
        if isinstance(parallel, dict):
            if parallel.get("stripe_hier"):
                return True
            if str(parallel.get("seq_p_attn_type", "")) == "stripe_hier":
                return True
        return os.environ.get("LIGHTX2V_STRIPE_HIER", "0") == "1"

    def _stripe_formc_hier_until_chunk(self) -> int | None:
        """Inclusive chunk index for Form C → hier early branch; None = disabled.

        Default when Form C is on and SP=4: use hier for seg_index <= 2.
        Override via parallel.stripe_formc_hier_until_chunk or
        LIGHTX2V_STRIPE_FORMC_HIER_UNTIL (-1 disables).
        """
        import os

        parallel = self.config.get("parallel") or {}
        if isinstance(parallel, dict) and "stripe_formc_hier_until_chunk" in parallel:
            until = int(parallel["stripe_formc_hier_until_chunk"])
            return None if until < 0 else until
        env = os.environ.get("LIGHTX2V_STRIPE_FORMC_HIER_UNTIL", "").strip()
        if env:
            until = int(env)
            return None if until < 0 else until
        # Default: enable early hier only when Form C itself is selected.
        return 2

    def _stripe_partial_out(self) -> bool:
        """Form C: alltoall partial (out,lse) rows instead of all_gather full partials."""
        import os

        if (
            not self._sp_stripe_kv()
            or self._stripe_full_q()
            or self._stripe_q_exchange()
            or self._stripe_hier()
        ):
            return False
        parallel = self.config.get("parallel") or {}
        if isinstance(parallel, dict):
            if parallel.get("stripe_partial_out"):
                return True
            if str(parallel.get("seq_p_attn_type", "")) == "stripe_pe":
                return True
        return os.environ.get("LIGHTX2V_STRIPE_PARTIAL_OUT", "0") == "1"

    def _stripe_q_exchange(self) -> bool:
        """Form B2: send Q_local to K owners, remote Flash, partials return."""
        import os

        if not self._sp_stripe_kv() or self._stripe_full_q() or self._stripe_hier():
            return False
        parallel = self.config.get("parallel") or {}
        if isinstance(parallel, dict):
            if parallel.get("stripe_q_exchange"):
                return True
            if str(parallel.get("seq_p_attn_type", "")) == "stripe_b2":
                return True
        return os.environ.get("LIGHTX2V_STRIPE_Q_EXCHANGE", "0") == "1"

    def _sp_cache_token_counts(
        self,
        seq_parallel: bool,
        sp_world_size: int,
        num_new: int,
    ) -> tuple[int, int, int]:
        local_per_frame = num_new // self.num_frame_per_chunk if self.num_frame_per_chunk > 0 else 0
        if seq_parallel and self._sp_stripe_kv() and self._stripe_full_q():
            # Q/activation is full on every rank; only KV is seq-striped.
            if num_new % sp_world_size != 0:
                raise ValueError(
                    f"stripe_full_q requires num_new ({num_new}) % sp_world_size ({sp_world_size}) == 0"
                )
            global_chunk_tokens = num_new
            local_chunk_tokens = num_new // sp_world_size
            cache_per_frame = local_per_frame // sp_world_size
            return global_chunk_tokens, local_chunk_tokens, cache_per_frame

        global_chunk_tokens = num_new * sp_world_size if seq_parallel else num_new
        if seq_parallel and self._sp_stripe_kv():
            local_chunk_tokens = num_new
            cache_per_frame = local_per_frame
        else:
            local_chunk_tokens = global_chunk_tokens
            cache_per_frame = local_per_frame * sp_world_size if seq_parallel else local_per_frame
        return global_chunk_tokens, local_chunk_tokens, cache_per_frame

    def _gather_self_attn_kv(
        self,
        kv_cache,
        k_cur: torch.Tensor,
        v_cur: torch.Tensor,
        attn_start: int,
        local_start_idx: int,
        local_end_idx: int,
        block_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mark = self._cuda_mark("kv_read")
        if local_start_idx > attn_start:
            hist_k = kv_cache.k_cache(block_idx, attn_start, local_start_idx)
            hist_v = kv_cache.v_cache(block_idx, attn_start, local_start_idx)
            attn_k = torch.cat([hist_k, k_cur], dim=0)
            attn_v = torch.cat([hist_v, v_cur], dim=0)
        else:
            attn_k = k_cur
            attn_v = v_cur
        self._cuda_end(*mark)
        return attn_k, attn_v

    def _apply_rope_sp(self, q, k, grid_sizes, freqs, start_frame):
        f, h, w = grid_sizes[0].tolist()
        full_seq_len = f * h * w
        c = q.size(-1) // 2

        freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        pos_freqs = torch.cat(
            [
                freqs_split[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(full_seq_len, 1, -1)

        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        padding_size = (world_size - (full_seq_len % world_size)) % world_size
        if padding_size > 0:
            pos_freqs = F.pad(pos_freqs, (0, 0, 0, 0, 0, padding_size))
        pos_freqs = torch.chunk(pos_freqs, world_size, dim=0)[cur_rank][: q.size(0)]

        n = q.size(1)
        q_c = torch.view_as_complex(q.float().reshape(q.size(0), n, -1, 2))
        k_c = torch.view_as_complex(k.float().reshape(k.size(0), n, -1, 2))
        pos_freqs = pos_freqs.to(torch.complex64)
        q = torch.view_as_real(q_c * pos_freqs).flatten(2).type_as(q)
        k = torch.view_as_real(k_c * pos_freqs).flatten(2).type_as(k)
        return q, k

    def infer_with_kvcache(self, blocks, x, pre_infer_out):
        """Run all transformer blocks with the rolling self-attention KV cache."""
        mgr = self.kv_cache_manager
        self.kv_cache_size = mgr.kv_size
        self.max_attention_size = mgr.max_attention_size
        self._kv_offload = self._ar_kv_offload
        kv_cache = mgr.self_attn_kv_cache
        num_blocks = len(blocks)

        for block_idx in range(num_blocks):
            self.block_idx = block_idx
            if self._kv_offload:
                self._next_prefetch = None
            x = self.infer_block_func(blocks[block_idx], x, pre_infer_out)

        if self._kv_offload:
            comp = getattr(kv_cache, "compute_stream", None)
            if comp is not None:
                comp.synchronize()
            kv_cache.sync_all()
        return x

    def infer_with_kvcache_blocks_offload(self, blocks, x, pre_infer_out):
        """Run transformer blocks with both weight offload and KV cache support."""
        mgr = self.kv_cache_manager
        self.kv_cache_size = mgr.kv_size
        self.max_attention_size = mgr.max_attention_size
        self._kv_offload = self._ar_kv_offload
        kv_cache = mgr.self_attn_kv_cache
        num_blocks = len(blocks)

        for block_idx in range(num_blocks):
            self.block_idx = block_idx
            if self._kv_offload:
                self._next_prefetch = None

            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            self.offload_manager.prefetch_weights((block_idx + 1) % num_blocks, blocks)
            gpu_block = self.offload_manager.cuda_buffers[0]
            if AI_DEVICE == "xpu":
                x = self.infer_block_func(gpu_block, x, pre_infer_out)
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    x = self.infer_block_func(gpu_block, x, pre_infer_out)

            self.offload_manager.swap_blocks()

        if self.clean_cuda_cache:
            del pre_infer_out.embed0, pre_infer_out.context
            torch_device_module.empty_cache()

        if self._kv_offload:
            if self._weight_offload_block_compute and AI_DEVICE == "cuda":
                self.offload_manager.compute_stream.synchronize()
            else:
                comp = getattr(kv_cache, "compute_stream", None)
                if comp is not None:
                    comp.synchronize()
            kv_cache.sync_all()
        return x

    def infer_block_with_kvoffload(self, block, x, pre_infer_out):
        """Run a transformer block with KV cache offload.

        ``RollingKVCachePool`` uses OffloadedStaticCache-style whole-layer
        prefetch inside ``infer_self_attn_with_kvcache`` (after ring roll).
        """
        kv_cache = self.kv_cache_manager.self_attn_kv_cache
        if self._weight_offload_block_compute:
            return self.infer_block_with_kvcache(block, x, pre_infer_out)
        comp = getattr(kv_cache, "compute_stream", None)
        if comp is not None:
            with torch_device_module.stream(comp):
                return self.infer_block_with_kvcache(block, x, pre_infer_out)
        return self.infer_block_with_kvcache(block, x, pre_infer_out)

    def infer_block_with_kvcache(self, block, x, pre_infer_out):
        """Run a transformer block with kv cache."""
        if hasattr(block.compute_phases[0], "before_proj"):
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )

        y_out = self.infer_self_attn_with_kvcache(
            block.compute_phases[0],
            pre_infer_out.grid_sizes.tensor,
            x,
            pre_infer_out.seq_lens,
            pre_infer_out.freqs,
            shift_msa,
            scale_msa,
        )

        x, attn_out = self.infer_cross_attn_with_kvcache(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
        )

        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)

        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        return x

    def _modulate_norm1(self, norm1_out, shift_msa, scale_msa, phase):
        """Apply per-frame AdaLN on norm1 output (tokens are frame-major)."""
        num_frames = shift_msa.shape[0]
        frame_seqlen = norm1_out.shape[0] // num_frames
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa * phase.smooth_norm1_bias.tensor
            norm1_out = norm1_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
            return norm1_out.flatten(0, 1)
        norm1_out = norm1_out.unsqueeze(0)
        scale_4d = scale_msa.unsqueeze(0)
        shift_4d = shift_msa.unsqueeze(0)
        return self.modulate_func(norm1_out, scale=scale_4d, shift=shift_4d).squeeze(0)

    def infer_self_attn_with_kvcache(self, phase, grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa):
        sa_start, sa_end, sa_kind = self._cuda_mark("self_attn")
        norm1_out = phase.norm1.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        norm1_out = self._modulate_norm1(norm1_out, shift_msa, scale_msa, phase)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        fc_mgr = getattr(self, "flowcache_manager", None)
        if (
            fc_mgr is not None
            and fc_mgr.uses_kv_compress
            and self.block_idx == 0
            and getattr(self.scheduler, "is_rerun", False)
        ):
            fc_mgr.store_query_state(self.block_idx, q)

        seg_index = int(self.scheduler.seg_index)
        current_start_frame = seg_index * self.num_frame_per_chunk

        if self.config.get("seq_parallel", False) and not self._stripe_full_q():
            q, k = self._apply_rope_sp(q, k, grid_sizes, freqs, current_start_frame)
        else:
            q = self.causal_rope_apply_func(q.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]
            k = self.causal_rope_apply_func(k.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]

        kv_cache = self.kv_cache_manager.self_attn_kv_cache
        seq_parallel = self.config.get("seq_parallel", False)
        sp_world_size = dist.get_world_size(self.seq_p_group) if seq_parallel else 1
        stripe_full_q = bool(seq_parallel and self._stripe_full_q())

        num_new = int(q.size(0))
        global_chunk_tokens, local_chunk_tokens, cache_per_frame = self._sp_cache_token_counts(
            seq_parallel, sp_world_size, num_new,
        )
        current_global_start = seg_index * global_chunk_tokens
        current_global_end = current_global_start + global_chunk_tokens
        global_end = kv_cache.get_global_end(self.block_idx)
        local_end = kv_cache.get_local_end(self.block_idx)
        sink_tokens = self.kv_cache_manager.sink_size * cache_per_frame
        global_delta = current_global_end - global_end
        local_delta = global_delta // sp_world_size if (seq_parallel and self._sp_stripe_kv()) else global_delta

        need_roll = (
            self.kv_cache_manager.local_attn_size != -1
            and current_global_end > global_end
            and local_chunk_tokens + local_end > self.kv_cache_size
        )
        if need_roll:
            num_evicted = local_chunk_tokens + local_end - self.kv_cache_size
            local_end_after_roll = local_end - num_evicted
        else:
            num_evicted = 0
            local_end_after_roll = local_end

        local_end_idx = local_end_after_roll + local_delta
        local_start_idx = local_end_idx - local_chunk_tokens
        attn_start = max(0, local_end_idx - self.max_attention_size)

        if hasattr(kv_cache, "_align") and not getattr(self, "_kivi_align_logged", False):
            self._kivi_align_logged = True
            A = kv_cache._align
            logger.info(
                "KIVI align: num_new={}, sink={}, num_evicted={}, local_start={}, mods: num_new={}, sink={}, evict={}, local_start={}, align={}",
                num_new,
                sink_tokens,
                num_evicted if need_roll else 0,
                local_start_idx,
                num_new % A,
                sink_tokens % A,
                (num_evicted if need_roll else 0) % A,
                local_start_idx % A,
                A,
            )

        # Ring rolling is metadata-only. Do it before materializing the
        # offload GPU window so logical [attn_start:local_end_idx) maps to
        # the post-roll physical layout.
        if need_roll:
            kv_cache.roll_window(self.block_idx, sink_tokens, num_evicted)

        if self._kv_offload:
            kv_cache.begin_layer(self.block_idx)

        persist_kv = self._should_persist_self_attn_kv()
        sp_stripe_kv = seq_parallel and self._sp_stripe_kv()
        stripe_hier = bool(sp_stripe_kv and self._stripe_hier())
        stripe_q_exchange = bool(sp_stripe_kv and self._stripe_q_exchange())
        stripe_partial_out = bool(sp_stripe_kv and self._stripe_partial_out())
        # Form C early-chunk branch: use hierarchical Stripe while KV is still small.
        if (
            stripe_partial_out
            and not stripe_hier
            and not stripe_q_exchange
            and sp_world_size in (4, 6)
        ):
            until = self._stripe_formc_hier_until_chunk()
            if until is not None and seg_index <= until:
                stripe_hier = True
                stripe_partial_out = False
        if seq_parallel and not sp_stripe_kv:
            a2a_start, a2a_end, a2a_kind = self._cuda_mark("sp_a2a")
            k_cur = all2all_seq2head(k, group=self.seq_p_group)
            v_cur = all2all_seq2head(v, group=self.seq_p_group)
            self._cuda_end(a2a_start, a2a_end, a2a_kind)
        else:
            k_cur, v_cur = k, v

        # Full-Q stripe: each rank computed full K/V for the chunk; keep only local stripe.
        if stripe_full_q:
            cur_rank = dist.get_rank(self.seq_p_group)
            k_cur = torch.chunk(k_cur, sp_world_size, dim=0)[cur_rank].contiguous()
            v_cur = torch.chunk(v_cur, sp_world_size, dim=0)[cur_rank].contiguous()

        if persist_kv:
            store_start, store_end, store_kind = self._cuda_mark("store_kv")
            kv_cache.store_kv(k_cur, v_cur, local_start_idx, local_end_idx, self.block_idx)
            self._cuda_end(store_start, store_end, store_kind)
            kv_cache.set_ends(self.block_idx, current_global_end, local_end_idx)
        elif self._kv_store_profiler is not None:
            self._kv_store_profiler.skipped_store_kv_calls += 1

        if self.clean_cuda_cache:
            del norm1_out
            torch_device_module.empty_cache()

        if seq_parallel:
            if attn_start >= local_start_idx:
                attn_k, attn_v = k_cur, v_cur
            elif persist_kv:
                read_start, read_end, read_kind = self._cuda_mark("kv_read")
                attn_k = kv_cache.k_cache(self.block_idx, attn_start, local_end_idx)
                attn_v = kv_cache.v_cache(self.block_idx, attn_start, local_end_idx)
                self._cuda_end(read_start, read_end, read_kind)
            else:
                attn_k, attn_v = self._gather_self_attn_kv(
                    kv_cache, k_cur, v_cur, attn_start, local_start_idx, local_end_idx, self.block_idx,
                )
            attn_out = kv_cache.sp_kvcache_attn_stripe(
                q=q,
                k_cache=attn_k,
                v_cache=attn_v,
                seq_p_group=self.seq_p_group,
                head_dim=self.head_dim,
                full_q=stripe_full_q,
                partial_out_exchange=stripe_partial_out,
                q_exchange=stripe_q_exchange,
                hier_exchange=stripe_hier,
            ) if sp_stripe_kv else kv_cache.sp_kvcache_attn_head_shard(
                q=q,
                k_cache=attn_k,
                v_cache=attn_v,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
        else:
            if attn_start >= local_start_idx:
                attn_k, attn_v = k_cur, v_cur
            elif persist_kv:
                read_start, read_end, read_kind = self._cuda_mark("kv_read")
                attn_k = kv_cache.k_cache(self.block_idx, attn_start, local_end_idx)
                attn_v = kv_cache.v_cache(self.block_idx, attn_start, local_end_idx)
                self._cuda_end(read_start, read_end, read_kind)
            else:
                attn_k, attn_v = self._gather_self_attn_kv(
                    kv_cache, k_cur, v_cur, attn_start, local_start_idx, local_end_idx, self.block_idx,
                )

            if self.config.get("ar_config", {}).get("kv_quant", {}).get("calibrate", False):
                kv_cache.capture_attn(self.block_idx, attn_start, local_end_idx)

            if isinstance(attn_k, tuple):
                k_lens = torch.empty_like(seq_lens).fill_(attn_k[0].size(0))
            else:
                k_lens = torch.empty_like(seq_lens).fill_(attn_k.size(0))
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=k_lens)
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=attn_k.size(0) if not isinstance(attn_k, tuple) else attn_k[0].size(0),
            )

        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch_device_module.empty_cache()
        if self._kv_offload:
            self.kv_cache_manager.self_attn_kv_cache.end_layer(
                self.block_idx,
                next_prefetch=None,
            )
        self._cuda_end(sa_start, sa_end, sa_kind)
        return y

    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa):
        ca_start, ca_end, ca_kind = self._cuda_mark("cross_attn")
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // num_frames
        seg_index = self.scheduler.seg_index

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))

        norm3_out = phase.norm3.apply(x)

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if context_img is not None:
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        cross_kv_cache = self.kv_cache_manager.cross_attn_kv_cache

        if seg_index == 0:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context).view(-1, n, d)
            ck_start, ck_end, ck_kind = self._cuda_mark("cross_kv_store")
            cross_kv_cache.store_kv(k, v, self.block_idx)
            self._cuda_end(ck_start, ck_end, ck_kind)
            self._cross_kv_len = k.size(0)
        else:
            L = self._cross_kv_len
            k = cross_kv_cache.k_cache(self.block_idx)[:L]
            v = cross_kv_cache.v_cache(self.block_idx)[:L]

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32),
        )
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32),
            )
            attn_out.add_(
                phase.cross_attn_2.apply(
                    q=q,
                    k=k_img,
                    v=v_img,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_k,
                    max_seqlen_q=q.size(0),
                    max_seqlen_kv=k_img.size(0),
                )
            )

            if self.clean_cuda_cache:
                del k_img, v_img
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        self._cuda_end(ca_start, ca_end, ca_kind)
        return x, attn_out

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if self.clean_cuda_cache:
            del attn_out
            torch.cuda.empty_cache()

        num_frames = c_shift_msa.shape[0]
        frame_seqlen = x.shape[0] // c_shift_msa.shape[0]

        if hasattr(phase, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze() * phase.smooth_norm2_bias.tensor
        else:
            norm2_weight = 1 + c_scale_msa
            norm2_bias = c_shift_msa

        norm2_out = phase.norm2.apply(x)
        norm2_out = norm2_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        norm2_out.mul_(norm2_weight).add_(norm2_bias)
        norm2_out = norm2_out.flatten(0, 1)

        y = phase.ffn_0.apply(norm2_out)
        if self.clean_cuda_cache:
            del norm2_out, x, norm2_weight, norm2_bias
            torch.cuda.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()
        y = phase.ffn_2.apply(y)

        return y

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        num_frames = c_gate_msa.shape[0]
        frame_seqlen = x.shape[0] // c_gate_msa.shape[0]
        y = y.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        x.add_(y * c_gate_msa)
        x = x.flatten(0, 1)

        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch.cuda.empty_cache()
        return x

    def infer_non_blocks(self, weights, x, e):
        num_frames = e.shape[0]
        frame_seqlen = x.shape[0] // e.shape[0]

        x = weights.norm.apply(x)
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))

        t = self.scheduler.timestep_input
        e = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        modulation = weights.head_modulation.tensor
        e = (modulation.unsqueeze(1) + e).chunk(2, dim=2)

        x.mul_(1 + e[1][0]).add_(e[0][0])
        x = x.flatten(0, 1)
        x = weights.head.apply(x)

        if self.clean_cuda_cache:
            del e
            torch.cuda.empty_cache()
        return x
