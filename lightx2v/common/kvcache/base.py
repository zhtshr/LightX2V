import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head

try:
    import flash_attn
except ImportError:
    flash_attn = None


class StripeAttnProfiler:
    """Accumulate CUDA-event timings for stripe attention phases."""

    PHASES = (
        "pre_barrier",
        "all_gather_q",
        "alltoall_q",
        "p2p_q",
        "p2p_kv",
        "ring_q",
        "gather_cat",
        "flash",
        "all_gather_out",
        "all_gather_lse",
        "alltoall_out",
        "alltoall_lse",
        "p2p_out",
        "p2p_lse",
        "ring_out",
        "merge",
        "slice",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ms = {k: 0.0 for k in self.PHASES}
        self.calls = 0
        self._pending: list[tuple[torch.cuda.Event, torch.cuda.Event, str]] = []

    def mark(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def end(self, start: torch.cuda.Event, end: torch.cuda.Event, kind: str) -> None:
        end.record()
        self._pending.append((start, end, kind))

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for start, end, kind in self._pending:
            self.ms[kind] = self.ms.get(kind, 0.0) + float(start.elapsed_time(end))
        self._pending.clear()

    def snapshot(self) -> dict:
        self.finalize()
        total = sum(self.ms.values())
        comm = (
            self.ms.get("all_gather_q", 0.0)
            + self.ms.get("alltoall_q", 0.0)
            + self.ms.get("p2p_q", 0.0)
            + self.ms.get("p2p_kv", 0.0)
            + self.ms.get("ring_q", 0.0)
            + self.ms.get("all_gather_out", 0.0)
            + self.ms.get("all_gather_lse", 0.0)
            + self.ms.get("alltoall_out", 0.0)
            + self.ms.get("alltoall_lse", 0.0)
            + self.ms.get("p2p_out", 0.0)
            + self.ms.get("p2p_lse", 0.0)
            + self.ms.get("ring_out", 0.0)
        )
        out = {
            "calls": self.calls,
            "total_ms": round(total, 3),
            "comm_ms": round(comm, 3),
            **{f"{k}_ms": round(v, 3) for k, v in self.ms.items()},
        }
        if total > 0:
            out["pct"] = {k: round(100.0 * self.ms[k] / total, 1) for k in self.PHASES if self.ms.get(k, 0.0)}
            out["comm_pct"] = round(100.0 * comm / total, 1)
        return out


_STRIPE_PROFILER: StripeAttnProfiler | None = None


class StripeGatherSkewRecorder:
    """Per-call Q-gather arrival skew without a pre-gather barrier.

    Each rank times its own all_gather (CUDA events). After the collective
    returns, ranks exchange those durations with a tiny all_gather.

    Same wall-clock end ⇒ longer duration means entered earlier (waited more).
    skew_ms = max(dur) - min(dur) ≈ how much earlier the fastest rank arrived
    vs the slowest for that call.
    """

    def __init__(self) -> None:
        self.samples: list[dict] = []

    def record(self, local_ms: float, seq_p_group) -> None:
        ws = dist.get_world_size(seq_p_group)
        device = torch.cuda.current_device()
        t = torch.tensor([local_ms], device=device, dtype=torch.float64)
        gathered = [torch.empty_like(t) for _ in range(ws)]
        dist.all_gather(gathered, t, group=seq_p_group)
        ms_by_rank = [float(x.item()) for x in gathered]
        lo = min(ms_by_rank)
        hi = max(ms_by_rank)
        self.samples.append(
            {
                "ms_by_rank": [round(x, 3) for x in ms_by_rank],
                "min_ms": round(lo, 3),
                "max_ms": round(hi, 3),
                "skew_ms": round(hi - lo, 3),
                "early_rank": int(ms_by_rank.index(hi)),
                "late_rank": int(ms_by_rank.index(lo)),
            }
        )

    def summary(self) -> dict:
        if not self.samples:
            return {"calls": 0}
        skews = [s["skew_ms"] for s in self.samples]
        mins = [s["min_ms"] for s in self.samples]
        maxs = [s["max_ms"] for s in self.samples]
        skews_sorted = sorted(skews)
        n = len(skews_sorted)

        def pct(p: float) -> float:
            if n == 1:
                return skews_sorted[0]
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return skews_sorted[idx]

        # How often "pure transfer" (late rank) stays near idle ~1.5ms
        late_near_idle = sum(1 for m in mins if m < 2.5) / n
        return {
            "calls": n,
            "skew_ms": {
                "mean": round(sum(skews) / n, 3),
                "p50": round(pct(0.50), 3),
                "p90": round(pct(0.90), 3),
                "p99": round(pct(0.99), 3),
                "max": round(max(skews), 3),
            },
            "late_rank_gather_ms_mean": round(sum(mins) / n, 3),
            "early_rank_gather_ms_mean": round(sum(maxs) / n, 3),
            "frac_calls_late_rank_lt_2.5ms": round(late_near_idle, 3),
            "note": (
                "skew_ms≈arrival gap (early vs late). late_rank_gather≈transfer; "
                "early_rank_gather≈wait+transfer. No barrier before Q gather."
            ),
            "head_samples": self.samples[:5],
            "tail_samples": self.samples[-5:],
        }


_STRIPE_GATHER_SKEW: StripeGatherSkewRecorder | None = None


def enable_stripe_gather_skew(enable: bool = True) -> StripeGatherSkewRecorder | None:
    global _STRIPE_GATHER_SKEW
    if enable:
        _STRIPE_GATHER_SKEW = StripeGatherSkewRecorder()
    else:
        _STRIPE_GATHER_SKEW = None
    return _STRIPE_GATHER_SKEW


def get_stripe_gather_skew() -> "StripeGatherSkewRecorder | None":
    global _STRIPE_GATHER_SKEW
    if _STRIPE_GATHER_SKEW is not None:
        return _STRIPE_GATHER_SKEW
    if os.environ.get("LIGHTX2V_STRIPE_GATHER_SKEW", "0") == "1":
        return enable_stripe_gather_skew(True)
    return None


def enable_stripe_attn_profiler(enable: bool = True) -> StripeAttnProfiler | None:
    global _STRIPE_PROFILER
    if enable:
        _STRIPE_PROFILER = StripeAttnProfiler()
    else:
        _STRIPE_PROFILER = None
    return _STRIPE_PROFILER


def get_stripe_attn_profiler() -> StripeAttnProfiler | None:
    if _STRIPE_PROFILER is not None:
        return _STRIPE_PROFILER
    if os.environ.get("LIGHTX2V_STRIPE_PROFILE", "0") == "1":
        return enable_stripe_attn_profiler(True)
    return None


class UlyssesAttnProfiler:
    """Accumulate CUDA-event timings for Ulysses head-shard attention phases."""

    PHASES = ("all2all_q", "flash", "all2all_out")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ms = {k: 0.0 for k in self.PHASES}
        self.calls = 0
        self._pending: list[tuple[torch.cuda.Event, torch.cuda.Event, str]] = []

    def mark(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def end(self, start: torch.cuda.Event, end: torch.cuda.Event, kind: str) -> None:
        end.record()
        self._pending.append((start, end, kind))

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for start, end, kind in self._pending:
            self.ms[kind] = self.ms.get(kind, 0.0) + float(start.elapsed_time(end))
        self._pending.clear()

    def snapshot(self) -> dict:
        self.finalize()
        total = sum(self.ms.values())
        comm = self.ms.get("all2all_q", 0.0) + self.ms.get("all2all_out", 0.0)
        out = {
            "calls": self.calls,
            "total_ms": round(total, 3),
            "comm_ms": round(comm, 3),
            **{f"{k}_ms": round(v, 3) for k, v in self.ms.items()},
        }
        if total > 0:
            out["pct"] = {k: round(100.0 * self.ms[k] / total, 1) for k in self.PHASES}
            out["comm_pct"] = round(100.0 * comm / total, 1)
        return out


_ULYSSES_PROFILER: UlyssesAttnProfiler | None = None


def enable_ulysses_attn_profiler(enable: bool = True) -> UlyssesAttnProfiler | None:
    global _ULYSSES_PROFILER
    if enable:
        _ULYSSES_PROFILER = UlyssesAttnProfiler()
    else:
        _ULYSSES_PROFILER = None
    return _ULYSSES_PROFILER


def get_ulysses_attn_profiler() -> UlyssesAttnProfiler | None:
    if _ULYSSES_PROFILER is not None:
        return _ULYSSES_PROFILER
    if os.environ.get("LIGHTX2V_ULYSSES_PROFILE", "0") == "1":
        return enable_ulysses_attn_profiler(True)
    return None


class BaseKVCachePool:
    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._cache_size = cache_size
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._device = device
        self._dtype = dtype

    def _init_kv_buffer(self):
        self._k_buffer = torch.zeros(
            (self._num_layers, self._cache_size, self._num_heads, self._head_dim),
            dtype=self._dtype,
            device=self._device,
        )
        self._v_buffer = torch.zeros(
            (self._num_layers, self._cache_size, self._num_heads, self._head_dim),
            dtype=self._dtype,
            device=self._device,
        )

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        if attn_start is None and local_end is None:
            return self._k_buffer[layer_id]
        return self._k_buffer[layer_id][attn_start:local_end]

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        if attn_start is None and local_end is None:
            return self._v_buffer[layer_id]
        return self._v_buffer[layer_id][attn_start:local_end]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        self._k_buffer[layer_id, : k.shape[0]] = k
        self._v_buffer[layer_id, : v.shape[0]] = v

    def reset(self) -> None:
        self._k_buffer.zero_()
        self._v_buffer.zero_()

    def sp_kvcache_attn_head_shard(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attention_module,
        seq_p_group,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """SP attention for KV cache stored as [global_seq, local_heads, head_dim].

        The caller keeps Q in the normal sequence-sharded layout
        [local_seq, global_heads, head_dim]. We convert Q to the head-sharded
        Ulysses layout, attend against the already head-sharded KV cache, then
        convert the output back to sequence-sharded layout.
        """
        if isinstance(k_cache, tuple) or isinstance(v_cache, tuple):
            raise TypeError(f"{self.__class__.__name__} does not support tuple K/V in head-shard SP path.")

        world_size = dist.get_world_size(seq_p_group)
        shard_heads = num_heads // world_size
        prof = get_ulysses_attn_profiler()

        if prof is not None:
            s, e = prof.mark()
            q_heads = all2all_seq2head(q, group=seq_p_group)
            prof.end(s, e, "all2all_q")
        else:
            q_heads = all2all_seq2head(q, group=seq_p_group)
        kv_len = int(k_cache.size(0))

        q_lens = torch.tensor([q_heads.size(0)], dtype=torch.int32)
        k_lens = torch.tensor([kv_len], dtype=torch.int32)
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)

        if prof is not None:
            s, e = prof.mark()
            attn_out = attention_module.apply(
                q=q_heads,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=q_heads.size(0),
                max_seqlen_kv=kv_len,
            )
            prof.end(s, e, "flash")
        else:
            attn_out = attention_module.apply(
                q=q_heads,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=q_heads.size(0),
                max_seqlen_kv=kv_len,
            )
        attn_out = attn_out.view(q_heads.size(0), shard_heads, head_dim)
        if prof is not None:
            s, e = prof.mark()
            attn_out = all2all_head2seq(attn_out, group=seq_p_group)
            prof.end(s, e, "all2all_out")
            prof.calls += 1
        else:
            attn_out = all2all_head2seq(attn_out, group=seq_p_group)
        return attn_out.reshape(q.size(0), num_heads * head_dim)

    @staticmethod
    def _stripe_local_flash(q, k, v, *, softmax_scale: float):
        if flash_attn is None:
            raise ImportError("flash_attn is required for stripe KV partial attention")
        return flash_attn.flash_attn_interface._flash_attn_forward(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

    @staticmethod
    def _stripe_merge_block(out, lse, block_out, block_lse):
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        out = out - F.sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
        return out, lse

    @staticmethod
    def _pack_out_lse_chunk(out_c: torch.Tensor, lse_c: torch.Tensor) -> torch.Tensor:
        """Pack bf16 out + fp32 lse into one contiguous uint8 buffer for a single collective."""
        o = out_c.contiguous().reshape(-1)
        l = lse_c.contiguous().reshape(-1)
        packed = torch.empty(
            o.numel() * o.element_size() + l.numel() * l.element_size(),
            dtype=torch.uint8,
            device=o.device,
        )
        packed[: o.nbytes].view(o.dtype).copy_(o)
        packed[o.nbytes :].view(l.dtype).copy_(l)
        return packed

    @staticmethod
    def _unpack_out_lse_chunk(
        packed: torch.Tensor,
        out_shape: torch.Size,
        lse_shape: torch.Size,
        out_dtype: torch.dtype,
        lse_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_out = 1
        for d in out_shape:
            n_out *= int(d)
        out_nbytes = n_out * torch.empty((), dtype=out_dtype).element_size()
        out = packed[:out_nbytes].view(out_dtype).view(out_shape)
        lse = packed[out_nbytes:].view(lse_dtype).view(lse_shape)
        return out, lse

    @staticmethod
    def _stripe_partial_out_alltoall(
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
        q_chunk_lens: list[int],
        seq_p_group,
        prof: "StripeAttnProfiler | None",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Exchange only each query-owner's partial (out, lse) rows (form C).

        LIGHTX2V_STRIPE_PACK_OUT_LSE=1 — single alltoall of packed (out||lse) instead of two.
        """
        world_size = len(q_chunk_lens)
        if not all(length == q_chunk_lens[0] for length in q_chunk_lens):
            raise ValueError(f"partial out exchange requires equal Q chunks, got {q_chunk_lens}")

        out_chunks = torch.split(block_out, q_chunk_lens, dim=1)
        lse_chunks = torch.split(block_lse, q_chunk_lens, dim=-1)
        cur_rank = dist.get_rank(seq_p_group)
        pack = os.environ.get("LIGHTX2V_STRIPE_PACK_OUT_LSE", "0") == "1"

        if pack:
            send_pack = [
                BaseKVCachePool._pack_out_lse_chunk(o, l)
                for o, l in zip(out_chunks, lse_chunks, strict=True)
            ]
            recv_pack = [torch.empty_like(send_pack[cur_rank]) for _ in range(world_size)]
            if prof is not None:
                s, e = prof.mark()
            dist.all_to_all(recv_pack, send_pack, group=seq_p_group)
            if prof is not None:
                prof.end(s, e, "alltoall_out")
            out_shape = out_chunks[cur_rank].shape
            lse_shape = lse_chunks[cur_rank].shape
            out_dtype = out_chunks[cur_rank].dtype
            lse_dtype = lse_chunks[cur_rank].dtype
            recv_out: list[torch.Tensor] = []
            recv_lse: list[torch.Tensor] = []
            for packed in recv_pack:
                o, l = BaseKVCachePool._unpack_out_lse_chunk(
                    packed, out_shape, lse_shape, out_dtype, lse_dtype
                )
                recv_out.append(o)
                recv_lse.append(l)
            return recv_out, recv_lse

        if os.environ.get("LIGHTX2V_STRIPE_SINGLE_A2A", "0") == "1":
            # Batched transport replaces per-peer temporary tensors. Peer order
            # and merge order stay unchanged; returned tensors are buffer views.
            batch, _, heads, dim = block_out.shape
            local_q = q_chunk_lens[0]
            send_out_buffer = block_out.reshape(batch, world_size, local_q, heads, dim).transpose(0, 1).contiguous()
            send_lse_buffer = block_lse.reshape(batch, heads, world_size, local_q).permute(2, 0, 1, 3).contiguous()
            recv_out_buffer = torch.empty_like(send_out_buffer)
            recv_lse_buffer = torch.empty_like(send_lse_buffer)
            if prof is not None:
                s, e = prof.mark()
            dist.all_to_all_single(recv_out_buffer, send_out_buffer, group=seq_p_group)
            if prof is not None:
                prof.end(s, e, "alltoall_out")
                s, e = prof.mark()
            dist.all_to_all_single(recv_lse_buffer, send_lse_buffer, group=seq_p_group)
            if prof is not None:
                prof.end(s, e, "alltoall_lse")
            return list(recv_out_buffer.unbind(0)), list(recv_lse_buffer.unbind(0))

        send_out = [chunk.contiguous() for chunk in out_chunks]
        send_lse = [chunk.contiguous() for chunk in lse_chunks]
        recv_out = [torch.empty_like(send_out[cur_rank]) for _ in range(world_size)]
        recv_lse = [torch.empty_like(send_lse[cur_rank]) for _ in range(world_size)]

        if prof is not None:
            s, e = prof.mark()
        dist.all_to_all(recv_out, send_out, group=seq_p_group)
        if prof is not None:
            prof.end(s, e, "alltoall_out")
            s, e = prof.mark()
        dist.all_to_all(recv_lse, send_lse, group=seq_p_group)
        if prof is not None:
            prof.end(s, e, "alltoall_lse")
        return recv_out, recv_lse

    def _stripe_gather_q_full(
        self,
        q: torch.Tensor,
        seq_p_group,
        prof: "StripeAttnProfiler | None",
        *,
        phase: str = "all_gather_q",
    ) -> tuple[torch.Tensor, list[int]]:
        """Gather seq-sharded Q into one contiguous buffer (no torch.cat).

        Optional: LIGHTX2V_STRIPE_GATHER_WITH_CAT=1 — list all_gather + cat (A/B only).
        """
        world_size = dist.get_world_size(seq_p_group)
        q_local = q.contiguous()
        q_len = int(q_local.size(0))
        use_cat = os.environ.get("LIGHTX2V_STRIPE_GATHER_WITH_CAT", "0") == "1"

        if use_cat:
            gathered = [torch.empty_like(q_local) for _ in range(world_size)]
            skew = get_stripe_gather_skew()
            if prof is not None:
                s, e = prof.mark()
            if skew is not None:
                s_sk = torch.cuda.Event(enable_timing=True)
                e_sk = torch.cuda.Event(enable_timing=True)
                s_sk.record()
            dist.all_gather(gathered, q_local, group=seq_p_group)
            if skew is not None:
                e_sk.record()
                torch.cuda.synchronize()
                skew.record(float(s_sk.elapsed_time(e_sk)), seq_p_group)
            if prof is not None:
                prof.end(s, e, phase)
                s, e = prof.mark()
            q_full = torch.cat(gathered, dim=0)
            if prof is not None:
                prof.end(s, e, "gather_cat")
            return q_full, [q_len] * world_size

        needed_shape = (q_len * world_size, *q_local.shape[1:])
        buf = getattr(self, "_stripe_q_gather_buf", None)
        if (
            buf is None
            or buf.shape != needed_shape
            or buf.dtype != q_local.dtype
            or buf.device != q_local.device
        ):
            self._stripe_q_gather_buf = torch.empty(
                needed_shape,
                dtype=q_local.dtype,
                device=q_local.device,
            )
            buf = self._stripe_q_gather_buf
        skew = get_stripe_gather_skew()
        if prof is not None:
            s, e = prof.mark()
        if skew is not None:
            s_sk = torch.cuda.Event(enable_timing=True)
            e_sk = torch.cuda.Event(enable_timing=True)
            s_sk.record()
        dist.all_gather_into_tensor(buf, q_local, group=seq_p_group)
        if skew is not None:
            e_sk.record()
            torch.cuda.synchronize()
            skew.record(float(s_sk.elapsed_time(e_sk)), seq_p_group)
        if prof is not None:
            prof.end(s, e, phase)
        return buf, [q_len] * world_size

    @staticmethod
    def _stripe_p2p_exchange_equal(
        send_bufs: list[torch.Tensor],
        recv_bufs: list[torch.Tensor],
        *,
        cur_rank: int,
        seq_p_group,
    ) -> None:
        """Full-mesh P2P all-to-all for equal-shaped chunks (no collective entry barrier).

        Rank r keeps ``send_bufs[r]`` locally as ``recv_bufs[r]`` and exchanges with peers
        via ``batch_isend_irecv``. Peers that have already posted can make progress without
        waiting for a common ``all_gather``/``all_to_all`` entry.
        """
        world_size = len(send_bufs)
        ops: list[dist.P2POp] = []
        for peer in range(world_size):
            if peer == cur_rank:
                recv_bufs[peer].copy_(send_bufs[peer])
                continue
            ops.append(dist.P2POp(dist.isend, send_bufs[peer], peer, group=seq_p_group))
            ops.append(dist.P2POp(dist.irecv, recv_bufs[peer], peer, group=seq_p_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()

    def _stripe_gather_q_p2p(
        self,
        q_local: torch.Tensor,
        seq_p_group,
        prof: "StripeAttnProfiler | None",
    ) -> tuple[torch.Tensor, list[int]]:
        """Assemble Q_full via P2P mesh: each rank pushes Q_local when ready."""
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        q_local = q_local.contiguous()
        q_len = int(q_local.size(0))
        needed_shape = (q_len * world_size, *q_local.shape[1:])
        buf = getattr(self, "_stripe_q_gather_buf", None)
        if (
            buf is None
            or buf.shape != needed_shape
            or buf.dtype != q_local.dtype
            or buf.device != q_local.device
        ):
            self._stripe_q_gather_buf = torch.empty(
                needed_shape,
                dtype=q_local.dtype,
                device=q_local.device,
            )
            buf = self._stripe_q_gather_buf

        # Layout matches all_gather_into_tensor: rank r occupies [r*q_len:(r+1)*q_len].
        send_slices = [q_local] * world_size
        recv_slices = [buf[i * q_len : (i + 1) * q_len] for i in range(world_size)]
        if prof is not None:
            s, e = prof.mark()
        self._stripe_p2p_exchange_equal(
            send_slices, recv_slices, cur_rank=cur_rank, seq_p_group=seq_p_group,
        )
        if prof is not None:
            prof.end(s, e, "p2p_q")
        return buf, [q_len] * world_size

    def _stripe_partial_out_p2p(
        self,
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
        q_chunk_lens: list[int],
        seq_p_group,
        prof: "StripeAttnProfiler | None",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return each query-owner's partial rows via P2P mesh (replaces alltoall)."""
        world_size = len(q_chunk_lens)
        if not all(length == q_chunk_lens[0] for length in q_chunk_lens):
            raise ValueError(f"P2P partial exchange requires equal Q chunks, got {q_chunk_lens}")

        cur_rank = dist.get_rank(seq_p_group)
        send_out = [chunk.contiguous() for chunk in torch.split(block_out, q_chunk_lens, dim=1)]
        send_lse = [chunk.contiguous() for chunk in torch.split(block_lse, q_chunk_lens, dim=-1)]
        recv_out = [torch.empty_like(send_out[cur_rank]) for _ in range(world_size)]
        recv_lse = [torch.empty_like(send_lse[cur_rank]) for _ in range(world_size)]

        if prof is not None:
            s, e = prof.mark()
        self._stripe_p2p_exchange_equal(
            send_out, recv_out, cur_rank=cur_rank, seq_p_group=seq_p_group,
        )
        if prof is not None:
            prof.end(s, e, "p2p_out")
            s, e = prof.mark()
        self._stripe_p2p_exchange_equal(
            send_lse, recv_lse, cur_rank=cur_rank, seq_p_group=seq_p_group,
        )
        if prof is not None:
            prof.end(s, e, "p2p_lse")
        return recv_out, recv_lse

    def _stripe_attn_form_b2_ring(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
    ) -> torch.Tensor:
        """Form B2 via Q-ring: rotate Q_local (+ traveling out/lse) P times.

        Each step on rank r:
          1. post send Q→next / recv Q←prev (overlaps Flash)
          2. Flash(Q_cur, K_local) and online-merge into traveling (out, lse)
          3. exchange (out, lse) so state stays with its Q
        After P steps, (Q, out, lse) return home with all K stripes merged.
        """
        from lightx2v.common.ops.attn.utils.ring_comm import RingComm

        world_size = dist.get_world_size(seq_p_group)
        softmax_scale = head_dim ** -0.5
        prof = get_stripe_attn_profiler()
        q_cur = q.contiguous()
        local_q_len = int(q_cur.size(0))

        if world_size == 1:
            q_b = q_cur.unsqueeze(0).contiguous()
            k_b = k_cache.unsqueeze(0).contiguous()
            v_b = v_cache.unsqueeze(0).contiguous()
            block_out, _, _, _ = self._stripe_local_flash(
                q_b, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.calls += 1
            return block_out.squeeze(0).reshape(local_q_len, -1)

        comm = RingComm(seq_p_group)
        k_b = k_cache.unsqueeze(0).contiguous()
        v_b = v_cache.unsqueeze(0).contiguous()
        out = None
        lse = None

        for _step in range(world_size):
            # Overlap Q hop with Flash (Q is read-only during Flash).
            next_q = comm.send_recv(q_cur)
            comm.commit()

            q_b = q_cur.unsqueeze(0).contiguous()
            if prof is not None:
                s_f, e_f = prof.mark()
            block_out, block_lse, _, _ = self._stripe_local_flash(
                q_b, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.end(s_f, e_f, "flash")

            if out is None:
                out = block_out.to(torch.float32)
                lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
            else:
                out, lse = self._stripe_merge_block(out, lse, block_out, block_lse)
            out = out.contiguous()
            lse = lse.contiguous()

            if prof is not None:
                s_q, e_q = prof.mark()
            comm.wait()
            if prof is not None:
                prof.end(s_q, e_q, "ring_q")

            if prof is not None:
                s_o, e_o = prof.mark()
            next_out = comm.send_recv(out)
            next_lse = comm.send_recv(lse)
            comm.commit()
            comm.wait()
            if prof is not None:
                prof.end(s_o, e_o, "ring_out")

            q_cur = next_q
            out = next_out
            lse = next_lse

        if prof is not None:
            prof.calls += 1
        return out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)

    def _stripe_attn_form_b2_p2p_overlap(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
    ) -> torch.Tensor:
        """Form B2 P2P with Flash/comm overlap and fused remote Flash.

        Per rank (K_local fixed):
          1. batch Q mesh // Flash(Q_local)
          2. wait Q mesh
          3. **one** Flash(cat(remote Q), K_local)  — same FLOPs as P-1 small
             flashes but Form-C-like kernel efficiency
          4. mesh-return each owner's remote partial; merge on query owner

        Env ``LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH=1`` restores legacy per-peer
        XOR Flash rounds (usually slower Flash).
        """
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        softmax_scale = head_dim ** -0.5
        prof = get_stripe_attn_profiler()
        q_local = q.contiguous()
        local_q_len = int(q_local.size(0))

        if world_size == 1:
            q_b = q_local.unsqueeze(0).contiguous()
            k_b = k_cache.unsqueeze(0).contiguous()
            v_b = v_cache.unsqueeze(0).contiguous()
            block_out, _, _, _ = self._stripe_local_flash(
                q_b, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.calls += 1
            return block_out.squeeze(0).reshape(local_q_len, -1)

        k_b = k_cache.unsqueeze(0).contiguous()
        v_b = v_cache.unsqueeze(0).contiguous()
        peers = [r for r in range(world_size) if r != cur_rank]

        q_send = q_local.clone()
        q_bufs: dict[int, torch.Tensor] = {}
        q_ops: list[dist.P2POp] = []
        for peer in peers:
            buf = torch.empty_like(q_local)
            q_bufs[peer] = buf
            q_ops.append(dist.P2POp(dist.isend, q_send, peer, group=seq_p_group))
            q_ops.append(dist.P2POp(dist.irecv, buf, peer, group=seq_p_group))
        q_reqs = dist.batch_isend_irecv(q_ops) if q_ops else []

        if prof is not None:
            s_f, e_f = prof.mark()
        block_out, block_lse, _, _ = self._stripe_local_flash(
            q_local.unsqueeze(0).contiguous(), k_b, v_b, softmax_scale=softmax_scale,
        )
        if prof is not None:
            prof.end(s_f, e_f, "flash")

        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        if prof is not None:
            s_q, e_q = prof.mark()
        for req in q_reqs:
            req.wait()
        if prof is not None:
            prof.end(s_q, e_q, "p2p_q")

        if not peers:
            if prof is not None:
                prof.calls += 1
            return out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)

        multi_flash = os.environ.get("LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH", "0") == "1"
        power_of_two = world_size > 0 and (world_size & (world_size - 1)) == 0
        if multi_flash and power_of_two:
            # Legacy: per-XOR-partner Flash + overlapped partial exchange.
            pending: tuple | None = None  # (ro, rl, reqs, send_hold)
            for k in range(1, world_size):
                partner = cur_rank ^ k
                if prof is not None:
                    s_f2, e_f2 = prof.mark()
                po, pl, _, _ = self._stripe_local_flash(
                    q_bufs[partner].unsqueeze(0).contiguous(),
                    k_b,
                    v_b,
                    softmax_scale=softmax_scale,
                )
                if prof is not None:
                    prof.end(s_f2, e_f2, "flash")
                po = po.contiguous()
                pl = pl.contiguous()

                if pending is not None:
                    ro_p, rl_p, reqs_p, _hold = pending
                    if prof is not None:
                        s_o, e_o = prof.mark()
                    for req in reqs_p:
                        req.wait()
                    if prof is not None:
                        prof.end(s_o, e_o, "p2p_out")
                    out, lse = self._stripe_merge_block(out, lse, ro_p, rl_p)

                ro = torch.empty_like(po)
                rl = torch.empty_like(pl)
                ops = [
                    dist.P2POp(dist.isend, po, partner, group=seq_p_group),
                    dist.P2POp(dist.irecv, ro, partner, group=seq_p_group),
                    dist.P2POp(dist.isend, pl, partner, group=seq_p_group),
                    dist.P2POp(dist.irecv, rl, partner, group=seq_p_group),
                ]
                reqs = dist.batch_isend_irecv(ops)
                pending = (ro, rl, reqs, (po, pl))

            if pending is not None:
                ro_p, rl_p, reqs_p, _hold = pending
                if prof is not None:
                    s_o, e_o = prof.mark()
                for req in reqs_p:
                    req.wait()
                if prof is not None:
                    prof.end(s_o, e_o, "p2p_out")
                out, lse = self._stripe_merge_block(out, lse, ro_p, rl_p)
        else:
            # Default: fuse all remote Q into one Flash, then mesh-return partials.
            # Peer order must match split order for send_out[i] → peers[i].
            q_remote = torch.cat([q_bufs[p] for p in peers], dim=0).unsqueeze(0).contiguous()
            if prof is not None:
                s_f2, e_f2 = prof.mark()
            rem_out, rem_lse, _, _ = self._stripe_local_flash(
                q_remote, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.end(s_f2, e_f2, "flash")
            send_out = [c.contiguous() for c in torch.split(rem_out, local_q_len, dim=1)]
            send_lse = [c.contiguous() for c in torch.split(rem_lse, local_q_len, dim=-1)]
            recv_out = [torch.empty_like(block_out) for _ in peers]
            recv_lse = [torch.empty_like(block_lse) for _ in peers]
            ret_ops: list[dist.P2POp] = []
            for i, peer in enumerate(peers):
                ret_ops.append(dist.P2POp(dist.isend, send_out[i], peer, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.irecv, recv_out[i], peer, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.isend, send_lse[i], peer, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.irecv, recv_lse[i], peer, group=seq_p_group))
            if prof is not None:
                s_o, e_o = prof.mark()
            for req in dist.batch_isend_irecv(ret_ops):
                req.wait()
            if prof is not None:
                prof.end(s_o, e_o, "p2p_out")
            for po, pl in zip(recv_out, recv_lse, strict=True):
                out, lse = self._stripe_merge_block(out, lse, po, pl)

        if prof is not None:
            prof.calls += 1
        return out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)

    def _stripe_attn_form_b2_mesh(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
        use_collective: bool,
    ) -> torch.Tensor:
        """Form B2 via full-mesh Q + Flash + partial return.

        Default P2P: Q mesh // local Flash, then **one fused** remote Flash, then
        mesh-return partials. Opt-in:
          - ``LIGHTX2V_STRIPE_B2_P2P_SERIAL=1``: assemble Q then single Flash(Q_full)
            (Form-C Flash shape; no local/remote overlap — often worse E2E).
          - ``LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH=1``: legacy P per-peer Flashes.
        """
        if not use_collective:
            if os.environ.get("LIGHTX2V_STRIPE_B2_P2P_SERIAL", "0") == "1":
                pass  # fall through to serial gather + fused Q_full Flash below
            else:
                return self._stripe_attn_form_b2_p2p_overlap(
                    q, k_cache, v_cache, seq_p_group, head_dim=head_dim,
                )

        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        softmax_scale = head_dim ** -0.5
        prof = get_stripe_attn_profiler()
        q_local = q.contiguous()

        if world_size == 1:
            q_b = q_local.unsqueeze(0).contiguous()
            k_b = k_cache.unsqueeze(0).contiguous()
            v_b = v_cache.unsqueeze(0).contiguous()
            block_out, _, _, _ = self._stripe_local_flash(
                q_b, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.calls += 1
            return block_out.squeeze(0).reshape(q_local.size(0), -1)

        if use_collective:
            q_full, q_chunk_lens = self._stripe_gather_q_full(
                q_local, seq_p_group, prof, phase="alltoall_q",
            )
        else:
            q_full, q_chunk_lens = self._stripe_gather_q_p2p(q_local, seq_p_group, prof)

        q_b = q_full.unsqueeze(0).contiguous()
        k_b = k_cache.unsqueeze(0).contiguous()
        v_b = v_cache.unsqueeze(0).contiguous()

        if prof is not None:
            s, e = prof.mark()
        block_out, block_lse, _, _ = self._stripe_local_flash(
            q_b, k_b, v_b, softmax_scale=softmax_scale,
        )
        if prof is not None:
            prof.end(s, e, "flash")

        if prof is not None:
            s, e = prof.mark()
        if use_collective:
            recv_out, recv_lse = self._stripe_partial_out_alltoall(
                block_out, block_lse, q_chunk_lens, seq_p_group, prof,
            )
        else:
            recv_out, recv_lse = self._stripe_partial_out_p2p(
                block_out, block_lse, q_chunk_lens, seq_p_group, prof,
            )
        out = recv_out[0].to(torch.float32)
        lse = recv_lse[0].transpose(-2, -1).unsqueeze(dim=-1)
        for partial_out, partial_lse in zip(recv_out[1:], recv_lse[1:], strict=True):
            out, lse = self._stripe_merge_block(out, lse, partial_out, partial_lse)
        if prof is not None:
            prof.end(s, e, "merge")
            prof.calls += 1

        local_q_len = q_chunk_lens[cur_rank]
        return out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)

    def _stripe_attn_form_b2_xor_rd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
    ) -> torch.Tensor:
        """Form B2 via XOR recursive-doubling Q exchange + Flash overlap.

        P must be a power of 2. Stages (example P=4):
          stage0: (0↔1)(2↔3) exchange 1×Q_local  // Flash(Q_local)
          stage1: (0↔2)(1↔3) exchange 2×Q_local  // Flash(Q newly received)
        Then XOR return of partials for each query owner (same as p2p XOR path).

        Total Q bytes still (P-1)×|Q_local|, but only log2(P) hops; Flash hides hop latency.
        """
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        softmax_scale = head_dim ** -0.5
        prof = get_stripe_attn_profiler()
        q_local = q.contiguous()
        local_q_len = int(q_local.size(0))

        if world_size == 1:
            q_b = q_local.unsqueeze(0).contiguous()
            k_b = k_cache.unsqueeze(0).contiguous()
            v_b = v_cache.unsqueeze(0).contiguous()
            block_out, _, _, _ = self._stripe_local_flash(
                q_b, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.calls += 1
            return block_out.squeeze(0).reshape(local_q_len, -1)

        if world_size & (world_size - 1) != 0:
            # Non-power-of-2: fall back to mesh overlap path.
            return self._stripe_attn_form_b2_p2p_overlap(
                q, k_cache, v_cache, seq_p_group, head_dim=head_dim,
            )

        k_b = k_cache.unsqueeze(0).contiguous()
        v_b = v_cache.unsqueeze(0).contiguous()
        held: dict[int, torch.Tensor] = {cur_rank: q_local}
        # Flash(Q_src, K_me) → partial for owner src
        partial_out: dict[int, torch.Tensor] = {}
        partial_lse: dict[int, torch.Tensor] = {}
        out = None
        lse = None

        def _flash_src(src: int, q_src: torch.Tensor) -> None:
            nonlocal out, lse
            if src in partial_out:
                return
            if prof is not None:
                s_f, e_f = prof.mark()
            po, pl, _, _ = self._stripe_local_flash(
                q_src.unsqueeze(0).contiguous(), k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.end(s_f, e_f, "flash")
            po = po.contiguous()
            pl = pl.contiguous()
            partial_out[src] = po
            partial_lse[src] = pl
            if src == cur_rank:
                out = po.to(torch.float32)
                lse = pl.transpose(-2, -1).unsqueeze(dim=-1)

        stages = world_size.bit_length() - 1
        pending_new: list[int] = []

        for stage in range(stages):
            dist_bit = 1 << stage
            partner = cur_rank ^ dist_bit
            send_ranks = sorted(held.keys())
            send_cat = torch.cat([held[r] for r in send_ranks], dim=0).contiguous()
            recv_cat = torch.empty_like(send_cat)
            if prof is not None:
                s_q, e_q = prof.mark()
            ops = [
                dist.P2POp(dist.isend, send_cat, partner, group=seq_p_group),
                dist.P2POp(dist.irecv, recv_cat, partner, group=seq_p_group),
            ]
            reqs = dist.batch_isend_irecv(ops)

            # Overlap: Flash(Q we already hold) while this hop runs.
            if stage == 0:
                _flash_src(cur_rank, held[cur_rank])
            else:
                for src in pending_new:
                    _flash_src(src, held[src])

            for req in reqs:
                req.wait()
            if prof is not None:
                prof.end(s_q, e_q, "p2p_q")

            recv_keys = sorted(r ^ dist_bit for r in send_ranks)
            for key, chunk in zip(recv_keys, torch.split(recv_cat, local_q_len, dim=0), strict=True):
                held[key] = chunk.contiguous()
            pending_new = recv_keys

        for src in pending_new:
            _flash_src(src, held[src])

        if out is None or lse is None or len(partial_out) != world_size:
            raise RuntimeError(
                f"xor_rd incomplete: flashed={sorted(partial_out)} ws={world_size}"
            )

        # XOR return of partials: send Flash(Q_partner,K_me) → partner; recv Flash(Q_me,K_partner).
        pending: tuple | None = None
        for k in range(1, world_size):
            partner = cur_rank ^ k
            po = partial_out[partner]
            pl = partial_lse[partner]
            ro = torch.empty_like(po)
            rl = torch.empty_like(pl)
            if pending is not None:
                ro_p, rl_p, reqs_p, _hold = pending
                if prof is not None:
                    s_o, e_o = prof.mark()
                for req in reqs_p:
                    req.wait()
                if prof is not None:
                    prof.end(s_o, e_o, "p2p_out")
                out, lse = self._stripe_merge_block(out, lse, ro_p, rl_p)

            ops = [
                dist.P2POp(dist.isend, po, partner, group=seq_p_group),
                dist.P2POp(dist.irecv, ro, partner, group=seq_p_group),
                dist.P2POp(dist.isend, pl, partner, group=seq_p_group),
                dist.P2POp(dist.irecv, rl, partner, group=seq_p_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            pending = (ro, rl, reqs, (po, pl))

        if pending is not None:
            ro_p, rl_p, reqs_p, _hold = pending
            if prof is not None:
                s_o, e_o = prof.mark()
            for req in reqs_p:
                req.wait()
            if prof is not None:
                prof.end(s_o, e_o, "p2p_out")
            out, lse = self._stripe_merge_block(out, lse, ro_p, rl_p)

        if prof is not None:
            prof.calls += 1
        return out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)

    @staticmethod
    def _hier_pair_and_crosses(rank: int, world_size: int) -> tuple[int, list[int]]:
        """Pair partner for KV exchange + cross peers for Q exchange.

        P=4: pairs (0,1)(2,3); each rank sends Q to one opposite-pair peer.
        P=6: pairs (0,1)(2,3)(4,5); each rank sends Q to one peer in each
        of the other two pairs (same within-pair offset).
        """
        if world_size == 4:
            return rank ^ 1, [rank ^ 2]
        if world_size == 6:
            pair = rank ^ 1
            my_g = rank // 2
            off = rank % 2
            crosses = [g * 2 + off for g in range(3) if g != my_g]
            return pair, crosses
        raise ValueError(
            f"stripe_hier supports seq_p_size=4 or 6, got world_size={world_size}"
        )

    def _stripe_attn_form_hier(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
    ) -> torch.Tensor:
        """Hierarchical Stripe for P=4 or P=6 (pair-KV then cross-Q).

        P=4: 2 pairs, Q to 1 cross rank.
        P=6: 3 pairs, Q to 2 cross ranks (one in each other pair).

        Overlap (side CUDA streams):
          1) Post pair-KV; optional Flash(Q, K_local) while KV flies.
          2) Wait peer KV → build K_group (2 stripes).
          3) Post cross-Q to all cross peers; while Q flies, finish local Q
             (peer half merge, or one Flash on K_group).
          4) Wait Q → fused Flash(cat remote Qs, K_group) → return partials → merge.

        LIGHTX2V_STRIPE_HIER_PACK=1 (default): pack K||V and out||lse.
        LIGHTX2V_STRIPE_HIER_OVERLAP_LOCAL=1 (default): Flash(Q,K_local) during KV.
        """
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        softmax_scale = head_dim ** -0.5
        prof = get_stripe_attn_profiler()
        q_local = q.contiguous()
        local_q_len = int(q_local.size(0))
        use_pack = os.environ.get("LIGHTX2V_STRIPE_HIER_PACK", "1") == "1"
        overlap_kv = os.environ.get("LIGHTX2V_STRIPE_HIER_OVERLAP_LOCAL", "1") == "1"

        if world_size == 1:
            q_b = q_local.unsqueeze(0).contiguous()
            k_b = k_cache.unsqueeze(0).contiguous()
            v_b = v_cache.unsqueeze(0).contiguous()
            block_out, _, _, _ = self._stripe_local_flash(
                q_b, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.calls += 1
            return block_out.squeeze(0).reshape(local_q_len, -1)

        pair, crosses = self._hier_pair_and_crosses(cur_rank, world_size)

        k_send = k_cache.contiguous()
        v_send = v_cache.contiguous()
        q_b_local = q_local.unsqueeze(0).contiguous()
        q_bufs = {c: torch.empty_like(q_local) for c in crosses}

        if use_pack:
            kv_send = torch.stack((k_send, v_send), dim=0).contiguous()
            kv_peer = torch.empty_like(kv_send)
            kv_ops = [
                dist.P2POp(dist.isend, kv_send, pair, group=seq_p_group),
                dist.P2POp(dist.irecv, kv_peer, pair, group=seq_p_group),
            ]
        else:
            k_peer = torch.empty_like(k_send)
            v_peer = torch.empty_like(v_send)
            kv_ops = [
                dist.P2POp(dist.isend, k_send, pair, group=seq_p_group),
                dist.P2POp(dist.irecv, k_peer, pair, group=seq_p_group),
                dist.P2POp(dist.isend, v_send, pair, group=seq_p_group),
                dist.P2POp(dist.irecv, v_peer, pair, group=seq_p_group),
            ]

        compute_stream = torch.cuda.current_stream()
        kv_stream = torch.cuda.Stream(device=compute_stream.device)
        with torch.cuda.stream(kv_stream):
            kv_stream.wait_stream(compute_stream)
            kv_reqs = dist.batch_isend_irecv(kv_ops)

        local_out = local_lse = None
        if overlap_kv:
            if prof is not None:
                s_f, e_f = prof.mark()
            local_out, local_lse, _, _ = self._stripe_local_flash(
                q_b_local,
                k_send.unsqueeze(0).contiguous(),
                v_send.unsqueeze(0).contiguous(),
                softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.end(s_f, e_f, "flash")

        if prof is not None:
            s_kv, e_kv = prof.mark()
        compute_stream.wait_stream(kv_stream)
        for req in kv_reqs:
            req.wait()
        if prof is not None:
            prof.end(s_kv, e_kv, "p2p_kv")

        if use_pack:
            k_peer = kv_peer[0]
            v_peer = kv_peer[1]

        if cur_rank < pair:
            k_group = torch.cat([k_send, k_peer], dim=0)
            v_group = torch.cat([v_send, v_peer], dim=0)
        else:
            k_group = torch.cat([k_peer, k_send], dim=0)
            v_group = torch.cat([v_peer, v_send], dim=0)
        k_b = k_group.unsqueeze(0).contiguous()
        v_b = v_group.unsqueeze(0).contiguous()

        # Cross-Q to all other pair-groups (1 peer for P=4, 2 for P=6).
        q_stream = torch.cuda.Stream(device=compute_stream.device)
        with torch.cuda.stream(q_stream):
            q_stream.wait_stream(compute_stream)
            q_ops: list[dist.P2POp] = []
            for c in crosses:
                q_ops.append(dist.P2POp(dist.isend, q_local, c, group=seq_p_group))
                q_ops.append(dist.P2POp(dist.irecv, q_bufs[c], c, group=seq_p_group))
            q_reqs = dist.batch_isend_irecv(q_ops)

        if overlap_kv and local_out is not None:
            if prof is not None:
                s_f, e_f = prof.mark()
            peer_out, peer_lse, _, _ = self._stripe_local_flash(
                q_b_local,
                k_peer.unsqueeze(0).contiguous(),
                v_peer.unsqueeze(0).contiguous(),
                softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.end(s_f, e_f, "flash")
            out = local_out.to(torch.float32)
            lse = local_lse.transpose(-2, -1).unsqueeze(dim=-1)
            if prof is not None:
                s_m, e_m = prof.mark()
            out, lse = self._stripe_merge_block(out, lse, peer_out, peer_lse)
            if prof is not None:
                prof.end(s_m, e_m, "merge")
        else:
            if prof is not None:
                s_f, e_f = prof.mark()
            block_out, block_lse, _, _ = self._stripe_local_flash(
                q_b_local, k_b, v_b, softmax_scale=softmax_scale,
            )
            if prof is not None:
                prof.end(s_f, e_f, "flash")
            out = block_out.to(torch.float32)
            lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        if prof is not None:
            s_q, e_q = prof.mark()
        compute_stream.wait_stream(q_stream)
        for req in q_reqs:
            req.wait()
        if prof is not None:
            prof.end(s_q, e_q, "p2p_q")

        # Fused Flash on all remote Qs against local K_group.
        q_remote = torch.cat([q_bufs[c] for c in crosses], dim=0).unsqueeze(0).contiguous()
        if prof is not None:
            s_f2, e_f2 = prof.mark()
        rem_out, rem_lse, _, _ = self._stripe_local_flash(
            q_remote, k_b, v_b, softmax_scale=softmax_scale,
        )
        if prof is not None:
            prof.end(s_f2, e_f2, "flash")

        send_out = [c.contiguous() for c in torch.split(rem_out, local_q_len, dim=1)]
        send_lse = [c.contiguous() for c in torch.split(rem_lse, local_q_len, dim=-1)]
        # Template shapes from first remote partial (same as local Q rows).
        out_tmpl = send_out[0]
        lse_tmpl = send_lse[0]
        recv_out = [torch.empty_like(out_tmpl) for _ in crosses]
        recv_lse = [torch.empty_like(lse_tmpl) for _ in crosses]

        if use_pack:
            send_packs = [
                self._pack_out_lse_chunk(o, l)
                for o, l in zip(send_out, send_lse, strict=True)
            ]
            recv_packs = [torch.empty_like(send_packs[0]) for _ in crosses]
            ret_ops = []
            for i, c in enumerate(crosses):
                ret_ops.append(dist.P2POp(dist.isend, send_packs[i], c, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.irecv, recv_packs[i], c, group=seq_p_group))
            if prof is not None:
                s_o, e_o = prof.mark()
            for req in dist.batch_isend_irecv(ret_ops):
                req.wait()
            if prof is not None:
                prof.end(s_o, e_o, "p2p_out")
            for i, packed in enumerate(recv_packs):
                o, l = self._unpack_out_lse_chunk(
                    packed, out_tmpl.shape, lse_tmpl.shape, out_tmpl.dtype, lse_tmpl.dtype,
                )
                recv_out[i] = o
                recv_lse[i] = l
        else:
            ret_ops = []
            for i, c in enumerate(crosses):
                ret_ops.append(dist.P2POp(dist.isend, send_out[i], c, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.irecv, recv_out[i], c, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.isend, send_lse[i], c, group=seq_p_group))
                ret_ops.append(dist.P2POp(dist.irecv, recv_lse[i], c, group=seq_p_group))
            if prof is not None:
                s_o, e_o = prof.mark()
            for req in dist.batch_isend_irecv(ret_ops):
                req.wait()
            if prof is not None:
                prof.end(s_o, e_o, "p2p_out")

        if prof is not None:
            s_m, e_m = prof.mark()
        for po, pl in zip(recv_out, recv_lse, strict=True):
            out, lse = self._stripe_merge_block(out, lse, po, pl)
        if prof is not None:
            prof.end(s_m, e_m, "merge")
            prof.calls += 1
        return out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)

    def _stripe_attn_form_b2(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
    ) -> torch.Tensor:
        """Form B2: Q to each K owner, remote Flash, merge on query owner.

        Mode via LIGHTX2V_STRIPE_B2_MODE:
          - p2p (default): Q mesh // local Flash, fused remote Flash, mesh partials
          - xor_rd: recursive-doubling Q exchange // Flash
          - ring: rotate Q (+ traveling out/lse) P steps
          - collective: all_gather Q + alltoall partials

        Legacy: LIGHTX2V_STRIPE_B2_COLLECTIVE=1 forces collective.
        LIGHTX2V_STRIPE_B2_P2P_SERIAL=1: blocking Q assemble + one Flash(Q_full).
        LIGHTX2V_STRIPE_B2_P2P_MULTI_FLASH=1: legacy per-peer Flash XOR rounds.
        """
        mode = os.environ.get("LIGHTX2V_STRIPE_B2_MODE", "p2p").strip().lower()
        if os.environ.get("LIGHTX2V_STRIPE_B2_COLLECTIVE", "0") == "1":
            mode = "collective"
        if mode in ("xor_rd", "xor", "rd"):
            return self._stripe_attn_form_b2_xor_rd(
                q, k_cache, v_cache, seq_p_group, head_dim=head_dim,
            )
        if mode == "ring":
            return self._stripe_attn_form_b2_ring(
                q, k_cache, v_cache, seq_p_group, head_dim=head_dim,
            )
        return self._stripe_attn_form_b2_mesh(
            q, k_cache, v_cache, seq_p_group,
            head_dim=head_dim,
            use_collective=(mode == "collective"),
        )

    def sp_kvcache_attn_stripe(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_p_group,
        *,
        head_dim: int,
        full_q: bool = False,
        partial_out_exchange: bool = False,
        q_exchange: bool = False,
        hier_exchange: bool = False,
    ) -> torch.Tensor:
        """Stripe-sharded KV attention with partial softmax merge.

        Default (full_q=False): gather seq-sharded Q into full Q, attend against the
        local K/V stripe, all_gather partial (out, lse), merge, return local seq slice.

        partial_out_exchange=True (form C): keep gather_q + Flash(Q_full, K_local), but
        replace all_gather(out,lse) with all_to_all of each query-owner's partial rows.

        q_exchange=True (form B2): Q visits each K owner.
        LIGHTX2V_STRIPE_B2_MODE=p2p|xor_rd|ring|collective (default p2p).

        hier_exchange=True (P=4 hierarchical): pair-exchange KV then cross-exchange Q.
        Mutually exclusive with full_q; takes precedence over q_exchange / partial_out.

        full_q=True: caller already holds full Q on every rank (activation not
        seq-sharded). Skip all_gather_q and return the full merged output.
        """
        if isinstance(k_cache, tuple) or isinstance(v_cache, tuple):
            raise TypeError(f"{self.__class__.__name__} does not support tuple K/V in stripe SP path.")

        if hier_exchange:
            if full_q:
                raise ValueError("hier_exchange (stripe_hier) is incompatible with full_q")
            return self._stripe_attn_form_hier(
                q, k_cache, v_cache, seq_p_group, head_dim=head_dim,
            )

        if q_exchange:
            if full_q:
                raise ValueError("q_exchange (form B2) is incompatible with full_q")
            return self._stripe_attn_form_b2(
                q, k_cache, v_cache, seq_p_group, head_dim=head_dim,
            )

        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        softmax_scale = head_dim ** -0.5
        prof = get_stripe_attn_profiler()
        q_chunk_lens: list[int] = []

        if world_size > 1 and not full_q:
            q_full, q_chunk_lens = self._stripe_gather_q_full(q, seq_p_group, prof)
            local_q_start = sum(q_chunk_lens[:cur_rank])
            local_q_len = q_chunk_lens[cur_rank]
        else:
            q_full = q
            local_q_start = 0
            local_q_len = int(q.size(0))
            if world_size > 1 and partial_out_exchange:
                if local_q_len % world_size != 0:
                    raise ValueError(
                        f"partial out exchange requires Q divisible by world_size, got Q={local_q_len}, ws={world_size}"
                    )
                q_chunk_lens = [local_q_len // world_size] * world_size

        q_b = q_full.unsqueeze(0).contiguous()
        k_b = k_cache.unsqueeze(0).contiguous()
        v_b = v_cache.unsqueeze(0).contiguous()

        if prof is not None:
            s, e = prof.mark()
        block_out, block_lse, _, _ = self._stripe_local_flash(
            q_b,
            k_b,
            v_b,
            softmax_scale=softmax_scale,
        )
        if prof is not None:
            prof.end(s, e, "flash")

        if world_size == 1:
            if prof is not None:
                prof.calls += 1
            return block_out.squeeze(0).reshape(q_full.size(0), -1)

        if prof is not None:
            s, e = prof.mark()

        if partial_out_exchange:
            recv_out, recv_lse = self._stripe_partial_out_alltoall(
                block_out,
                block_lse,
                q_chunk_lens,
                seq_p_group,
                prof,
            )
            out = recv_out[0].to(torch.float32)
            lse = recv_lse[0].transpose(-2, -1).unsqueeze(dim=-1)
            for partial_out, partial_lse in zip(recv_out[1:], recv_lse[1:], strict=True):
                out, lse = self._stripe_merge_block(out, lse, partial_out, partial_lse)
            if prof is not None:
                prof.end(s, e, "merge")
            out_local = out.to(q.dtype).squeeze(0).reshape(local_q_len, -1)
            if prof is not None:
                prof.calls += 1
            return out_local if not full_q else out.to(q.dtype).squeeze(0).reshape(q_full.size(0), -1)
        else:
            out_list = [torch.empty_like(block_out) for _ in range(world_size)]
            lse_list = [torch.empty_like(block_lse) for _ in range(world_size)]
            dist.all_gather(out_list, block_out.contiguous(), group=seq_p_group)
            if prof is not None:
                prof.end(s, e, "all_gather_out")
                s, e = prof.mark()
            dist.all_gather(lse_list, block_lse.contiguous(), group=seq_p_group)
            if prof is not None:
                prof.end(s, e, "all_gather_lse")

            # Opt-in: rows are independent in the softmax merge. Slice before
            # converting/merging to avoid computing rows owned by other ranks.
            local_merge = not full_q and os.environ.get("LIGHTX2V_STRIPE_LOCAL_MERGE", "0") == "1"
            if local_merge:
                end = local_q_start + local_q_len
                out_list = [part[:, local_q_start:end] for part in out_list]
                lse_list = [part[:, :, local_q_start:end] for part in lse_list]
            out = out_list[0].to(torch.float32)
            lse = lse_list[0].transpose(-2, -1).unsqueeze(dim=-1)
            for partial_out, partial_lse in zip(out_list[1:], lse_list[1:], strict=True):
                out, lse = self._stripe_merge_block(out, lse, partial_out, partial_lse)

            if prof is not None:
                prof.end(s, e, "merge")
                s, e = prof.mark()
            out_full = out.to(q.dtype).squeeze(0).reshape(local_q_len if local_merge else q_full.size(0), -1)
            result = out_full if full_q or local_merge else out_full[local_q_start : local_q_start + local_q_len]
            if prof is not None:
                prof.end(s, e, "slice")
                prof.calls += 1
            return result

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def cache_size(self) -> int:
        return self._cache_size
