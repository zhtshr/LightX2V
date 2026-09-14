#!/usr/bin/env python3
"""Isolate A2B1 overlap: real COMP pump vs GEMM-only on compute_stream."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import seed_all


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _wall(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    pp = _load("pp", here / "tp_phase_pipeline.py")

    ns = Namespace(
        config_json="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json",
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v", model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="t", seed=42, seq_p_size=1, tensor_p_size=2,
        inputs_cache="save_results/optimization_study/phase1_encoder_inputs_256x256.pt",
        refresh_inputs_cache=False, negative_prompt="",
    )
    config = p1._load_config(ns)
    config["tp_norm_p2p"] = True
    seed_all(42)
    p1._init_distributed(config)
    pp.configure_phase_pipeline(use_comm_p2p=True)

    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(config=config, cache_path=Path(ns.inputs_cache),
                                 prompt=ns.prompt, image_path=ns.image_path, seed=42,
                                 force=False, task="i2v", negative_prompt=""))
    payload_b = p1._prepare_payload_on_device({
        "seed": 43, "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"], "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    sched_a, sched_b = WanScheduler(config), WanScheduler(config)
    model.set_scheduler(sched_a)
    ta = ov.TenantCtx("A", sched_a, payload_a["inputs"])
    tb = ov.TenantCtx("B", sched_b, payload_b["inputs"])
    for t, p, s in ((ta, payload_a, sched_a), (tb, payload_b, sched_b)):
        s.prepare(seed=int(p["seed"]), latent_shape=p["latent_shape"],
                  image_encoder_output=p["image_encoder_output"])
        s.step_pre(step_index=0)
        ov._pre_infer_tenant(model, t)
    wan, ti = ov._bind_tenant(model, ta)
    ov._preload_blocks(ta, wan, ti, len(wan.transformer_weights.blocks))
    wan_b, ti_b = ov._bind_tenant(model, tb)
    ov._preload_blocks(tb, wan_b, ti_b, len(wan_b.transformer_weights.blocks))
    ta.pipe_states, tb.pipe_states = {}, {}
    layer = 10
    orch = pp._get_orch(torch.device(f"cuda:{dist.get_rank()}"))

    # prep through COMM
    pp.run_tenant_phase(model, ta, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
    pp.run_tenant_phase(model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)

    from lightx2v.common.ops.tp_p2p_allreduce import get_tp_p2p_allreduce

    block = wan.transformer_weights.blocks[layer]
    ph = block.compute_phases[0]
    o_t = ta.pipe_states[layer].self_scratch["o_partial"]
    ex = get_tp_p2p_allreduce(ph.self_attn_o.tp_group)
    gemm = torch.randn(4096, 4096, device=o_t.device, dtype=torch.float16)

    def overlap_real() -> None:
        orch.enabled = True
        orch.set_peer(pp._make_peer_runner(
            model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap))
        pp.run_tenant_phase(model, ta, layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        orch.drain_peer()
        orch.enabled = False
        orch.set_peer(None)

    def overlap_gemm_only() -> None:
        peer = pp._make_peer_runner(model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        with torch.cuda.stream(orch.comm_stream):
            work = ex.all_reduce(o_t, orch.comm_stream, async_op=True)
        with torch.cuda.stream(orch.compute_stream):
            while not peer.finished():
                for _ in range(2):
                    torch.matmul(gemm, gemm)
                peer.advance(orch)
        if work is not None:
            work.wait()

    def overlap_gemm_nccl() -> None:
        peer = pp._make_peer_runner(model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        with torch.cuda.stream(orch.comm_stream):
            work = dist.all_reduce(o_t, op=dist.ReduceOp.SUM, group=ph.self_attn_o.tp_group, async_op=True)
        with torch.cuda.stream(orch.compute_stream):
            while not peer.finished():
                for _ in range(2):
                    torch.matmul(gemm, gemm)
                peer.advance(orch)
        if work is not None:
            work.wait()

    if dist.get_rank() == 0:
        print(f"overlap_real_ms: {_wall(overlap_real):.2f}")
        ta.pipe_states, tb.pipe_states = {}, {}
        pp.run_tenant_phase(model, ta, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        pp.run_tenant_phase(model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        o_t = ta.pipe_states[layer].self_scratch["o_partial"]
        print(f"overlap_gemm+p2p_ms: {_wall(overlap_gemm_only):.2f}")
        print(f"overlap_gemm+nccl_ms: {_wall(overlap_gemm_nccl):.2f}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
