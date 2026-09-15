#!/usr/bin/env python3
"""Compare dual-tenant overlap strategies: serial vs legacy vs profile-segment.

Run (MoE 480p SLA P=4):
  torchrun --standalone --nproc_per_node=4 \\
    scripts/disagg/run_phase3_segment_overlap_compare.py \\
    --config_json save_results/optimization_study/baseline_moe_i2v_480_sla_triton_seqp4.json \\
    --measure_steps 4 \\
    --output_json save_results/optimization_study/p3_segment_overlap_compare_moe_480_sla_seqp4.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load_phase1():
    path = Path(__file__).with_name("run_phase1_transformer_bench.py")
    spec = importlib.util.spec_from_file_location("phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_phase3():
    path = Path(__file__).with_name("run_phase3_dual_overlap_bench.py")
    spec = importlib.util.spec_from_file_location("phase3", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_p_size", type=int, default=4)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--task", default="i2v", choices=("i2v", "t2v"))
    parser.add_argument("--model_cls", default="wan2.2_moe")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--measure_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    p1 = _load_phase1()
    p3 = _load_phase3()
    ns = argparse.Namespace(
        seq_p_size=args.seq_p_size,
        config_json=args.config_json,
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
    )
    config = p1._load_config(ns)
    p1._init_distributed(config)
    seed_all(42)

    payload_a = p1._prepare_inputs_cache(
        config, Path(args.inputs_cache), prompt="bench", image_path="", seed=42, task=args.task,
    )
    payload_a = p1._prepare_payload_on_device(payload_a)
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    scheduler_a = WanScheduler(config)
    scheduler_b = WanScheduler(config)
    tenant_a = p3.TenantCtx("A", scheduler_a, payload_a["inputs"])
    tenant_b = p3.TenantCtx("B", scheduler_b, payload_b["inputs"])

    results: dict[str, object] = {
        "config_json": args.config_json,
        "seq_p_size": config.get("parallel", {}).get("seq_p_size", 1),
        "self_attn_type": config.get("self_attn_1_type"),
        "resolution": f"{config.get('target_height')}x{config.get('target_width')}",
        "measure_steps": args.measure_steps,
        "runs": {},
    }

    modes = (
        ("serial", False, "legacy"),
        ("legacy_overlap", True, "legacy"),
        ("segment_overlap", True, "segment"),
    )

    for name, overlap, strategy in modes:
        for _ in range(args.warmup_steps):
            p3._run_dual_a2a_pipeline(
                model, tenant_a, tenant_b, payload_a, payload_b,
                overlap=overlap, overlap_strategy=strategy, steps=1,
            )
            if dist.is_initialized():
                dist.barrier()

        wall_s, stats = p3._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=overlap, overlap_strategy=strategy, steps=args.measure_steps,
        )
        one_step_s, one_stats = p3._run_dual_a2a_pipeline(
            model, tenant_a, tenant_b, payload_a, payload_b,
            overlap=overlap, overlap_strategy=strategy, steps=1,
        )
        results["runs"][name] = {
            "overlap": overlap,
            "overlap_strategy": strategy,
            "wall_s": wall_s,
            "one_step_s": one_step_s,
            "throughput_rps": 2.0 / wall_s if wall_s > 0 else None,
            "stats": stats,
            "one_step_stats": one_stats,
        }
        if is_main_process():
            print(f"{name}: wall={wall_s:.3f}s one_step={one_step_s:.3f}s", flush=True)

    serial_s = results["runs"]["serial"]["wall_s"]  # type: ignore[index]
    legacy_s = results["runs"]["legacy_overlap"]["wall_s"]  # type: ignore[index]
    segment_s = results["runs"]["segment_overlap"]["wall_s"]  # type: ignore[index]
    results["speedup"] = {
        "legacy_vs_serial": serial_s / legacy_s if legacy_s else None,
        "segment_vs_serial": serial_s / segment_s if segment_s else None,
        "segment_vs_legacy": legacy_s / segment_s if segment_s else None,
    }
    results["pairing"] = {
        "input_comm_ms": 25.2,
        "middle_compute_ms": 27.4,
        "output_comm_ms": 8.4,
        "cross_ffn_ms": 32.0,
        "input_pair": "input_a2a (~25ms) || cross_ffn (~32ms)",
        "middle_pair": "middle_compute (~27ms) || next_layer_input_comm (~25ms)",
        "output_pair": "output_a2a (~8ms) || cross_ffn_remainder (noop if done)",
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if is_main_process():
        print(json.dumps({"speedup": results["speedup"], "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
