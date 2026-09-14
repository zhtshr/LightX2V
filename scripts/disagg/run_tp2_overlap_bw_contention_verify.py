#!/usr/bin/env python3
"""Verify memory-bandwidth contention: per COMP micro-step latency alone vs during O AR.

If overlap slows compute steps without changing kernels, HBM/PCIe contention is confirmed.

Run:
  torchrun --standalone --nproc_per_node=2 \\
    scripts/disagg/run_tp2_overlap_bw_contention_verify.py \\
    --layer 10 --pair A2B1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.utils.utils import is_main_process, seed_all


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cuda_step_ms(fn, stream: torch.cuda.Stream) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        fn()
        end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _prep_comp_state(
    phase_mod: Any,
    model: Any,
    tenant: Any,
    layer: int,
    comp_p: int,
    steps: tuple[str, ...],
    upto: int,
    ov: Any,
    orch: Any,
) -> None:
    tenant.pipe_states = {}
    orch.enabled = False
    orch.set_peer(None)
    for name in steps[:upto]:
        phase_mod._run_comp_substep(
            comp_p, name, model, tenant, layer,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap, orch,
        )


def _prep_o_ar_tensor(
    phase_mod: Any,
    model: Any,
    comm_t: Any,
    layer: int,
    comm_p: int,
    ov: Any,
) -> tuple[torch.Tensor, dist.ProcessGroup]:
    comm_t.pipe_states = {}
    for ph in range(1, comm_p):
        phase_mod.run_tenant_phase(
            model, comm_t, layer, ph, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
    wan, ti = ov._bind_tenant(model, comm_t)
    block = ov._ensure_block(comm_t, wan, ti, layer)
    ti.block_idx = layer
    st = phase_mod._state(comm_t, layer)
    phase_mod._self_o_linear(wan, ti, comm_t, block, st)
    y_out = st.self_scratch["o_partial"]
    pf = block.compute_phases[0]
    return y_out, pf.self_attn_o.tp_group


def _cuda_ar_ms(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    stream: torch.cuda.Stream,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
        work.wait()
        end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    here = Path(__file__).parent
    prof_mod = _load("prof", here / "run_tp2_phase_overlap_profile.py")
    p1 = _load("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load("overlap", here / "run_phase3_dual_overlap_bench.py")
    phase_mod = _load("phase_pipe", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--pair", default="A2B1", choices=["A2B1"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output_json", default="save_results/optimization_study/tp2_bw_contention_verify.json")
    args = parser.parse_args()

    pair_map = {
        "A2B1": (2, 1),
    }
    comm_p, comp_p = pair_map[args.pair]

    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="test",
        seed=42,
        seq_p_size=1,
        tensor_p_size=2,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="",
    )
    config = p1._load_config(ns)
    config["tp_norm_p2p"] = True
    seed_all(42)
    p1._init_distributed(config)

    device = torch.device(f"cuda:{dist.get_rank()}")
    orch = phase_mod._get_orch(device)

    payload_a = p1._prepare_inputs_cache(
        config=config, cache_path=Path(args.inputs_cache), prompt=ns.prompt,
        image_path=ns.image_path, seed=42, force=False, task="i2v", negative_prompt="",
    )
    payload_a = p1._prepare_payload_on_device(payload_a)
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    tenant_a, tenant_b, _ = prof_mod._prepare_tenants(p1, ov, model, config, payload_a, payload_b)
    wan, ti = ov._bind_tenant(model, tenant_a)
    ov._preload_blocks(tenant_a, wan, ti, len(wan.transformer_weights.blocks))
    prof_mod._warmup_all_phases(phase_mod, model, tenant_a, args.layer, ov)

    comm_t, comp_t = (tenant_a, tenant_b) if args.pair == "A2B1" else (tenant_b, tenant_a)

    runner = phase_mod._make_peer_runner(
        model, comp_t, args.layer, comp_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
    )
    steps = runner._steps()

    ar_tensor, ar_group = _prep_o_ar_tensor(phase_mod, model, comm_t, args.layer, comm_p, ov)
    ar_alone_samples: list[float] = []
    for _ in range(args.repeats):
        ar_alone_samples.append(_cuda_ar_ms(ar_tensor.clone(), ar_group, orch.comm_stream))

    per_step: list[dict[str, Any]] = []
    for step_i, name in enumerate(steps):
        alone_r: list[float] = []
        during_r: list[float] = []
        for _ in range(args.repeats):
            _prep_comp_state(phase_mod, model, comp_t, args.layer, comp_p, steps, step_i, ov, orch)
            alone_r.append(_cuda_step_ms(
                lambda n=name: phase_mod._run_comp_substep(
                    comp_p, n, model, comp_t, args.layer,
                    ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap, orch,
                ),
                orch.compute_stream,
            ))

            _prep_comp_state(phase_mod, model, comp_t, args.layer, comp_p, steps, step_i, ov, orch)
            ar_buf = ar_tensor.clone()
            with torch.cuda.stream(orch.comm_stream):
                work = dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=ar_group, async_op=True)
            during_r.append(_cuda_step_ms(
                lambda n=name: phase_mod._run_comp_substep(
                    comp_p, n, model, comp_t, args.layer,
                    ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap, orch,
                ),
                orch.compute_stream,
            ))
            if work is not None:
                work.wait()
            torch.cuda.synchronize()

        a = statistics.mean(alone_r)
        d = statistics.mean(during_r)
        per_step.append({
            "step": name,
            "alone_cuda_ms": a,
            "during_ar_cuda_ms": d,
            "slowdown_ratio": d / max(a, 0.001),
            "delta_ms": d - a,
        })

    sum_alone = sum(s["alone_cuda_ms"] for s in per_step)
    sum_during = sum(s["during_ar_cuda_ms"] for s in per_step)
    ar_alone = statistics.mean(ar_alone_samples)
    ideal_no_contention = max(sum_alone, ar_alone)
    naive_serial = sum_alone + ar_alone

    # Full pump path (same as overlap profile): wall vs cuda-event sum
    prof2 = prof_mod.OrchProfile()
    prof2.attach(orch)
    overlap_walls: list[float] = []
    pump_cuda: list[float] = []
    ar_cuda: list[float] = []
    for _ in range(args.repeats):
        prof_mod._prep_through_phase(phase_mod, model, comm_t, args.layer, comm_p, ov)
        prof_mod._prep_through_phase(phase_mod, model, comp_t, args.layer, comp_p, ov)
        prof2.ar_window_ms.clear()
        prof2.pump_during_ar_ms.clear()
        orch.enabled = True
        orch.set_peer(phase_mod._make_peer_runner(
            model, comp_t, args.layer, comp_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        phase_mod.run_tenant_phase(
            model, comm_t, args.layer, comm_p, ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        )
        orch.drain_peer()
        orch.enabled = False
        orch.set_peer(None)
        torch.cuda.synchronize()
        overlap_walls.append((time.perf_counter() - t0) * 1000.0)
        if prof2.ar_window_ms:
            ar_cuda.append(prof2.ar_window_ms[-1])
        if prof2.pump_during_ar_ms:
            pump_cuda.append(prof2.pump_during_ar_ms[-1])
    prof2.detach(orch)

    result = {
        "pair": args.pair,
        "layer": args.layer,
        "comm_p": comm_p,
        "comp_p": comp_p,
        "repeats": args.repeats,
        "ar_alone_cuda_ms": ar_alone,
        "sum_step_alone_cuda_ms": sum_alone,
        "sum_step_during_ar_cuda_ms": sum_during,
        "step_slowdown_ratio_mean": statistics.mean(s["slowdown_ratio"] for s in per_step),
        "step_slowdown_ratio_weighted": sum_during / max(sum_alone, 0.001),
        "ideal_if_perfect_overlap_ms": ideal_no_contention,
        "naive_serial_sum_ms": naive_serial,
        "predicted_overlap_wall_if_contention_ms": sum_during,
        "predicted_overlap_wall_if_no_contention_ms": ideal_no_contention,
        "full_pump_overlap_wall_ms": statistics.mean(overlap_walls),
        "full_pump_ar_cuda_ms": statistics.mean(ar_cuda) if ar_cuda else None,
        "full_pump_compute_cuda_ms": statistics.mean(pump_cuda) if pump_cuda else None,
        "dual_stream_naive_wall_ms": None,
        "per_step": per_step,
        "verdict": None,
    }
    if result["step_slowdown_ratio_weighted"] > 1.15:
        result["verdict"] = "bandwidth_contention_supported"
    elif result["full_pump_overlap_wall_ms"] > ideal_no_contention * 1.5:
        result["verdict"] = "lack_of_stream_parallelism_not_step_slowdown"
    else:
        result["verdict"] = "overlap_works_as_expected"

    result["interpretation"] = (
        "Per-step during AR: slowdown≈1.0 → not bandwidth via slower kernels. "
        "If full_pump_overlap_wall >> ideal while step sum≈ideal, streams fail to overlap in wall time."
    )

    out_path = Path(args.output_json)
    if is_main_process():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {out_path}")
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
