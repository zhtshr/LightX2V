#!/usr/bin/env python3
"""Measure whether A2B1 overlap comm_stream and compute_stream timelines overlap on GPU."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.profiler import no_sync_profiling
from lightx2v.utils.utils import is_main_process, seed_all


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("ph", here / "tp_phase_pipeline.py")

    ns = __import__("argparse").Namespace(
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
    phase.configure_phase_pipeline(use_comm_p2p=False)

    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(
            config=config, cache_path=Path(ns.inputs_cache),
            prompt=ns.prompt, image_path=ns.image_path, seed=42,
            force=False, task="i2v", negative_prompt="",
        )
    )
    payload_b = p1._prepare_payload_on_device({
        "seed": 43, "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })
    model = load_wan_transformer(config)
    sa, sb = WanScheduler(config), WanScheduler(config)
    ta = ov.TenantCtx("A", sa, payload_a["inputs"])
    tb = ov.TenantCtx("B", sb, payload_b["inputs"])
    for t, p, s in ((ta, payload_a, sa), (tb, payload_b, sb)):
        s.prepare(seed=int(p["seed"]), latent_shape=p["latent_shape"],
                  image_encoder_output=p["image_encoder_output"])
        t.inputs = p["inputs"]
        s.step_pre(step_index=0)
        ov._pre_infer_tenant(model, t)

    layer = 10
    orch = phase._get_orch(torch.device(f"cuda:{dist.get_rank()}"))

    def prep(tenant, ph):
        tenant.pipe_states = {}
        for p in range(1, ph):
            phase.run_tenant_phase(model, tenant, layer, p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)

    prep(ta, 2)
    prep(tb, 1)
    prep(ta, 2)

    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    comp_start = torch.cuda.Event(enable_timing=True)
    comp_end = torch.cuda.Event(enable_timing=True)
    wall_start = torch.cuda.Event(enable_timing=True)
    wall_end = torch.cuda.Event(enable_timing=True)

    orig_handle = orch._handle_ar

    def instrumented_handle(launch):
        comm_start.record(orch.comm_stream)
        orig_handle(launch)
        comm_end.record(orch.comm_stream)

    orig_advance = orch._advance_peer

    def instrumented_advance():
        nonlocal comp_started
        if not comp_started:
            with torch.cuda.stream(orch.compute_stream):
                comp_start.record(orch.compute_stream)
            comp_started = True
        return orig_advance()

    comp_started = False
    orch._handle_ar = instrumented_handle
    orch._advance_peer = instrumented_advance

    wall_start.record()
    with no_sync_profiling(enabled=True):
        orch.enabled = True
        orch.set_peer(phase._make_peer_runner(
            model, tb, layer, 1, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        phase.run_tenant_phase(model, ta, layer, 2, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap)
        orch.drain_peer()
        orch.enabled = False
        orch.set_peer(None)
    with torch.cuda.stream(orch.compute_stream):
        comp_end.record(orch.compute_stream)
    wall_end.record()
    torch.cuda.synchronize()

    comm_ms = comm_start.elapsed_time(comm_end)
    comp_ms = comp_start.elapsed_time(comp_end) if comp_started else 0.0
    wall_ms = wall_start.elapsed_time(wall_end)
    # overlap of intervals on GPU timeline (approx): if wall << comm+comp then concurrent
    serial_est = comm_ms + comp_ms

    if is_main_process():
        print({
            "comm_span_ms": comm_ms,
            "comp_span_ms": comp_ms,
            "wall_ms": wall_ms,
            "comm_plus_comp_ms": serial_est,
            "wall_vs_serial_sum_ratio": wall_ms / max(serial_est, 0.001),
            "concurrent_if_ratio_near_1_of_max": wall_ms / max(max(comm_ms, comp_ms), 0.001),
        })

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
