#!/usr/bin/env python3
"""Probe SLA + SP rank imbalance on distill models.

Collects per-rank:
  - active K-block counts (union of selected blocks)
  - local tokens contributed to sparse KV all2all
  - sparse-map / attn wall times (CUDA events)
  - optional barrier wait skew

Usage (SP=2 on physical GPUs 6,7):
  CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --standalone --nproc_per_node=2 \\
    scripts/disagg/probe_sla_sp_imbalance.py \\
    --config_json save_results/optimization_study/baseline_moe_i2v_480_sla_triton_seqp2.json \\
    --attn_modes ulysses,ulysses_sparse,ulysses_sparse_l2
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
from lightx2v.common.ops.attn.utils.sparse_block_comm import _local_active_seq_mask
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1_bench", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ImbalanceCollector:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.enabled = False
        self.mode = ""

    def clear(self) -> None:
        self.records.clear()

    def add(self, row: dict[str, Any]) -> None:
        if self.enabled:
            self.records.append(row)


COLLECTOR = ImbalanceCollector()
_ORIG_SPARSE = UlyssesAttnWeight._sparse_img_kv_all2all


def _patched_sparse_img_kv_all2all(self, img_k_local, img_v_local, img_q_perm, **kwargs):
    attention_module = kwargs["attention_module"]
    world_size = kwargs["world_size"]
    shard_seqlen = kwargs["shard_seqlen"]
    global_img_seqlen = kwargs["global_img_seqlen"]
    q_shard_heads = kwargs["q_shard_heads"]
    kv_shard_heads = kwargs["kv_shard_heads"]
    hidden_dims = kwargs["hidden_dims"]
    seq_p_group = kwargs["seq_p_group"]

    # Reuse original path but instrument around block-map / all2all.
    from lightx2v.common.ops.attn.utils.sla_util import mean_pool
    from lightx2v.common.ops.attn.utils.sparse_block_comm import (
        block_map_from_pooled,
        expand_sparse_kv_to_dense,
        fill_inactive_from_kmeans,
        gather_global_kmeans,
        global_active_k_blocks,
        sparse_kv_ulysses_all2all,
    )

    output_q = torch.empty_like(img_q_perm)
    dist.all_to_all_single(output_q, img_q_perm, group=seq_p_group)
    shard_img_q = output_q.reshape(global_img_seqlen, q_shard_heads, hidden_dims)

    topk_ratio = float(getattr(attention_module, "topk", 0.2))
    blkq = int(getattr(attention_module, "BLKQ", 64))
    blkk = int(getattr(attention_module, "BLKK", 64))

    k_means_global = gather_global_kmeans(img_k_local, world_size=world_size, blkk=blkk, seq_p_group=seq_p_group)
    cur_rank = dist.get_rank(seq_p_group)
    h0 = cur_rank * kv_shard_heads
    h1 = h0 + kv_shard_heads
    k_means_shard = k_means_global[:, h0:h1, :, :]
    q_means = mean_pool(shard_img_q.transpose(0, 1).unsqueeze(0).contiguous(), blkq)

    map_start = torch.cuda.Event(enable_timing=True)
    map_end = torch.cuda.Event(enable_timing=True)
    map_start.record()
    sparse_map, lut, real_topk = block_map_from_pooled(q_means, k_means_shard, topk_ratio)
    active_blocks = global_active_k_blocks(sparse_map)
    map_end.record()

    seq_mask = _local_active_seq_mask(active_blocks, cur_rank, shard_seqlen, blkk)
    local_active_tokens = int(seq_mask.sum().item())
    local_active_blocks = 0
    kb_shard = (shard_seqlen + blkk - 1) // blkk
    block_start = cur_rank * kb_shard
    block_end = block_start + kb_shard
    for bid in active_blocks.tolist():
        if block_start <= int(bid) < block_end:
            local_active_blocks += 1

    a2a_start = torch.cuda.Event(enable_timing=True)
    a2a_end = torch.cuda.Event(enable_timing=True)
    a2a_start.record()
    sparse_k, sparse_v = sparse_kv_ulysses_all2all(
        img_k_local,
        img_v_local,
        active_blocks,
        world_size=world_size,
        q_shard_heads=q_shard_heads,
        kv_shard_heads=kv_shard_heads,
        hidden_dims=hidden_dims,
        seq_p_group=seq_p_group,
    )
    a2a_end.record()
    torch.cuda.synchronize()

    mode = self.sparse_kv_mode
    self._sla_sparse_meta = {
        "sla_precomputed_img_sparse_map": sparse_map,
        "sla_precomputed_img_lut": lut,
        "sla_precomputed_img_topk": real_topk,
        "sla_img_seqlen": global_img_seqlen,
        "sla_k_means_shard": k_means_shard,
        "sla_skip_get_block_map": mode >= 1,
    }
    if mode >= 2:
        self._sla_sparse_meta["sla_kv_compact"] = True
        self._sla_sparse_meta["sla_active_blocks"] = active_blocks
        self._sla_sparse_meta["sla_blkk"] = blkk
        shard_img_k, shard_img_v = sparse_k, sparse_v
    else:
        shard_img_k, shard_img_v = expand_sparse_kv_to_dense(
            sparse_k,
            sparse_v,
            active_blocks,
            global_seqlen=global_img_seqlen,
            world_size=world_size,
            shard_seqlen=shard_seqlen,
            blkk=blkk,
        )
        if mode < 1:
            fill_inactive_from_kmeans(
                shard_img_k,
                shard_img_v,
                k_means_global,
                active_blocks,
                global_seqlen=global_img_seqlen,
                cur_rank=cur_rank,
                kv_shard_heads=kv_shard_heads,
                blkk=blkk,
            )

    COLLECTOR.add(
        {
            "rank": cur_rank,
            "mode": COLLECTOR.mode,
            "active_blocks": int(active_blocks.numel()),
            "local_active_blocks": local_active_blocks,
            "local_active_tokens": local_active_tokens,
            "sparse_k_len": int(sparse_k.shape[0]),
            "real_topk": int(real_topk),
            "map_ms": float(map_start.elapsed_time(map_end)),
            "sparse_a2a_ms": float(a2a_start.elapsed_time(a2a_end)),
            "shard_seqlen": int(shard_seqlen),
            "global_img_seqlen": int(global_img_seqlen),
        }
    )
    return shard_img_q, shard_img_k, shard_img_v


def _summarize_records(records: list[dict[str, Any]], world: int) -> dict[str, Any]:
    if not records:
        return {"n_calls": 0}
    by_rank: dict[int, list[dict[str, Any]]] = {r: [] for r in range(world)}
    for row in records:
        by_rank[int(row["rank"])].append(row)

    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / max(len(vals), 1))

    per_rank = {}
    for r, rows in by_rank.items():
        if not rows:
            continue
        per_rank[str(r)] = {
            "n_calls": len(rows),
            "active_blocks_mean": _mean([x["active_blocks"] for x in rows]),
            "local_active_blocks_mean": _mean([x["local_active_blocks"] for x in rows]),
            "local_active_tokens_mean": _mean([x["local_active_tokens"] for x in rows]),
            "sparse_k_len_mean": _mean([x["sparse_k_len"] for x in rows]),
            "map_ms_mean": _mean([x["map_ms"] for x in rows]),
            "sparse_a2a_ms_mean": _mean([x["sparse_a2a_ms"] for x in rows]),
            "local_active_tokens_min": min(x["local_active_tokens"] for x in rows),
            "local_active_tokens_max": max(x["local_active_tokens"] for x in rows),
            "active_blocks_min": min(x["active_blocks"] for x in rows),
            "active_blocks_max": max(x["active_blocks"] for x in rows),
        }

    # Pairwise imbalance between ranks on same call index.
    skew_tokens: list[float] = []
    skew_blocks: list[float] = []
    n = min(len(by_rank[r]) for r in range(world)) if world > 0 else 0
    for i in range(n):
        toks = [by_rank[r][i]["local_active_tokens"] for r in range(world)]
        blks = [by_rank[r][i]["active_blocks"] for r in range(world)]
        tmean = sum(toks) / world
        bmean = sum(blks) / world
        if tmean > 0:
            skew_tokens.append((max(toks) - min(toks)) / tmean)
        if bmean > 0:
            skew_blocks.append((max(blks) - min(blks)) / bmean)

    return {
        "n_calls_total": len(records),
        "n_calls_aligned": n,
        "per_rank": per_rank,
        "token_skew_rel_mean": _mean(skew_tokens) if skew_tokens else 0.0,
        "token_skew_rel_max": max(skew_tokens) if skew_tokens else 0.0,
        "active_blocks_skew_rel_mean": _mean(skew_blocks) if skew_blocks else 0.0,
        "active_blocks_skew_rel_max": max(skew_blocks) if skew_blocks else 0.0,
    }


def _run_mode(
    phase1: Any,
    base_config: dict[str, Any],
    payload: dict[str, Any],
    attn_type: str,
    *,
    warmup: int,
    measure: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["parallel"] = dict(config.get("parallel") or {})
    config["parallel"]["seq_p_size"] = int(config["parallel"].get("seq_p_size", 2))
    config["parallel"]["seq_p_attn_type"] = attn_type
    config["self_attn_1_type"] = "sla_attn"
    config.setdefault("sla_attn_setting", {"sparsity_ratio": 0.8, "operator": "triton"})
    # Keep original offload setting (A14B MoE needs cpu_offload on A10).
    # Imbalance metrics (active blocks / local tokens) are independent of weight offload.

    from lightx2v.disagg.utils import load_wan_transformer

    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)

    sparse = attn_type.startswith("ulysses_sparse") or attn_type == "ring_sla"
    COLLECTOR.mode = attn_type
    COLLECTOR.enabled = sparse
    COLLECTOR.clear()

    # Warmup without collecting.
    COLLECTOR.enabled = False
    for _ in range(warmup):
        phase1._run_transformer_compute(scheduler, model, payload)
    if dist.is_initialized():
        dist.barrier()

    COLLECTOR.enabled = sparse
    COLLECTOR.clear()
    samples_s: list[float] = []
    barrier_skew_ms: list[float] = []
    for _ in range(measure):
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        phase1._run_transformer_compute(scheduler, model, payload)
        torch.cuda.synchronize()
        samples_s.append(time.perf_counter() - t0)

        # Measure arrive-to-barrier skew: early ranks wait longer.
        if dist.is_initialized():
            torch.cuda.synchronize()
            local_t = torch.tensor([time.perf_counter()], device="cuda")
            gathered = [torch.empty_like(local_t) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_t)
            times = [float(x.item()) for x in gathered]
            barrier_skew_ms.append((max(times) - min(times)) * 1000.0)
            dist.barrier()

    local_summary = _summarize_records(COLLECTOR.records, dist.get_world_size() if dist.is_initialized() else 1)

    # Gather records to rank0 for global summary.
    if dist.is_initialized():
        obj = [COLLECTOR.records]
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj[0])
        all_records = []
        for part in gathered:
            all_records.extend(part or [])
        global_summary = _summarize_records(all_records, dist.get_world_size())
    else:
        global_summary = local_summary

    # Per-rank transformer times via all_gather.
    local_avg = sum(samples_s) / len(samples_s)
    if dist.is_initialized():
        t = torch.tensor([local_avg], device="cuda")
        outs = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(outs, t)
        per_rank_transformer_s = [float(x.item()) for x in outs]
    else:
        per_rank_transformer_s = [local_avg]

    del model, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()

    return {
        "attn_type": attn_type,
        "sparse_kv_path": sparse,
        "transformer_s_mean": local_avg,
        "transformer_s_samples": [round(x, 4) for x in samples_s],
        "per_rank_transformer_s": [round(x, 4) for x in per_rank_transformer_s],
        "transformer_rank_skew_rel": (
            (max(per_rank_transformer_s) - min(per_rank_transformer_s)) / (sum(per_rank_transformer_s) / len(per_rank_transformer_s))
            if per_rank_transformer_s
            else 0.0
        ),
        "barrier_skew_ms_mean": (sum(barrier_skew_ms) / len(barrier_skew_ms)) if barrier_skew_ms else 0.0,
        "imbalance": global_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_json",
        default="/root/zht/LightX2V/save_results/optimization_study/baseline_moe_i2v_480_sla_triton_seqp2.json",
    )
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v")
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--seq_p_size", type=int, default=2)
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--attn_modes", default="ulysses,ulysses_sparse,ulysses_sparse_l2")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="/root/zht/LightX2V/save_results/optimization_study/sla_sp2_imbalance_probe.json",
    )
    parser.add_argument(
        "--prompt",
        default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    )
    parser.add_argument("--image_path", default="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg")
    args = parser.parse_args()

    phase1 = _load_phase1()
    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        seq_p_size=args.seq_p_size,
        tensor_p_size=0,
    )
    config = phase1._load_config(ns)
    seed_all(args.seed)
    phase1._init_distributed(config)

    # Install sparse instrumentation for all modes that use it.
    UlyssesAttnWeight._sparse_img_kv_all2all = _patched_sparse_img_kv_all2all

    payload = phase1._prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=args.prompt,
        image_path=args.image_path,
        seed=args.seed,
        force=False,
        task=args.task,
    )
    payload = phase1._prepare_payload_on_device(payload)

    modes = [m.strip() for m in args.attn_modes.split(",") if m.strip()]
    results: dict[str, Any] = {
        "gpus_visible": torch.cuda.device_count(),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "config": args.config_json,
        "model_path": args.model_path,
        "modes": {},
    }

    for mode in modes:
        if is_main_process():
            print(f"=== probing attn_type={mode} ===")
        try:
            results["modes"][mode] = _run_mode(
                phase1,
                config,
                payload,
                mode,
                warmup=args.warmup,
                measure=args.measure_iters,
            )
            if is_main_process():
                m = results["modes"][mode]
                imb = m["imbalance"]
                print(
                    f"  transformer={m['transformer_s_mean']:.3f}s "
                    f"rank_skew={m['transformer_rank_skew_rel']:.3f} "
                    f"token_skew_mean={imb.get('token_skew_rel_mean', 0):.3f} "
                    f"token_skew_max={imb.get('token_skew_rel_max', 0):.3f} "
                    f"active_blk_skew_mean={imb.get('active_blocks_skew_rel_mean', 0):.3f}"
                )
                for rk, st in (imb.get("per_rank") or {}).items():
                    print(
                        f"    rank{rk}: local_tok={st['local_active_tokens_mean']:.1f} "
                        f"local_blk={st['local_active_blocks_mean']:.1f} "
                        f"active_blk={st['active_blocks_mean']:.1f} "
                        f"a2a_ms={st['sparse_a2a_ms_mean']:.2f}"
                    )
        except Exception as exc:
            if is_main_process():
                print(f"  FAILED {mode}: {exc}")
            results["modes"][mode] = {"error": str(exc)}
            if dist.is_initialized():
                dist.barrier()

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
