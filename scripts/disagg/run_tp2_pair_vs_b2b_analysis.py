#!/usr/bin/env python3
"""Measure all 6 overlap pairs vs b2b — explain E2E gap at 256×256.

  CUDA_VISIBLE_DEVICES=0,1 PROFILING_DEBUG_LEVEL=0 PYTHONPATH=$PWD \\
    torchrun --standalone --nproc_per_node=2 \\
      scripts/disagg/run_tp2_pair_vs_b2b_analysis.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _cuda_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _aggregate_pair_timings(pair_timings: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Group A2B1_L10 -> A2B1, average wall_s across layers."""
    by_pair: dict[str, list[float]] = defaultdict(list)
    for rec in pair_timings:
        name = rec["pair"]
        base = name.rsplit("_L", 1)[0]
        by_pair[base].append(float(rec["wall_s"]) * 1000.0)
    out: dict[str, dict[str, float]] = {}
    for base, walls_ms in sorted(by_pair.items()):
        out[base] = {
            "mean_wall_ms": statistics.mean(walls_ms),
            "sum_wall_ms": sum(walls_ms),
            "layers": len(walls_ms),
        }
    return out


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("phase1", here / "run_phase1_transformer_bench.py")
    ov = _load("overlap", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("phase_pipe", here / "tp_phase_pipeline.py")
    prof_mod = _load("prof", here / "run_tp2_phase_overlap_profile.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--measure_steps", type=int, default=4)
    parser.add_argument(
        "--output_json",
        default="save_results/optimization_study/wan22_256_tp2_pair_vs_b2b.json",
    )
    args = parser.parse_args()

    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
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
    phase.configure_phase_pipeline(use_stream_ar_wait=True, use_row_ready_event=True)

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

    device = torch.device(f"cuda:{dist.get_rank()}")
    orch = phase._get_orch(device)
    prof = prof_mod.OrchProfile()
    prof.attach(orch)

    tenant_a, tenant_b, scheduler_a = prof_mod._prepare_tenants(
        p1, ov, model, config, payload_a, payload_b,
    )
    scheduler_b = tenant_b.scheduler
    prof_mod._warmup_all_phases(phase, model, tenant_a, args.layer, ov)

    pair_specs = [
        ("A2B1", tenant_a, 2, tenant_b, 1),
        ("B2A3", tenant_b, 2, tenant_a, 3),
        ("A4B3", tenant_a, 4, tenant_b, 3),
        ("B4A5", tenant_b, 4, tenant_a, 5),
        ("A6B5", tenant_a, 6, tenant_b, 5),
        ("B6A1", tenant_b, 6, tenant_a, 1),
    ]

    layer = args.layer
    pairs_micro: dict[str, Any] = {}
    for label, comm_t, comm_p, comp_t, comp_p in pair_specs:
        serial = prof_mod._profile_pair(
            phase, model, ov, orch, prof, layer=layer,
            comm_t=comm_t, comm_p=comm_p, comp_t=comp_t, comp_p=comp_p, overlap=False,
        )
        overlap = prof_mod._profile_pair(
            phase, model, ov, orch, prof, layer=layer,
            comm_t=comm_t, comm_p=comm_p, comp_t=comp_t, comp_p=comp_p, overlap=True,
        )
        pairs_micro[label] = {"serial": serial, "overlap": overlap}

    prof.detach(orch)

    phase_kwargs = dict(
        bind_tenant=ov._bind_tenant,
        ensure_block=ov._ensure_block,
        preload_blocks=ov._preload_blocks,
        pre_infer_tenant=ov._pre_infer_tenant,
        finish_step_tenant=ov._finish_step_tenant,
        capture_ti_snap=ov._capture_ti_snap,
        time_fn=ov._time_fn,
    )

    for _ in range(1):
        ov._run_denoise_serial(model, scheduler_a, payload_a)
    _cuda_gc()

    single_s = ov._time_fn(lambda: ov._run_denoise_serial(model, scheduler_a, payload_a))
    _cuda_gc()
    dual_b2b_s = ov._time_fn(lambda: (
        ov._run_denoise_serial(model, scheduler_a, payload_a),
        ov._run_denoise_serial(model, scheduler_b, payload_b),
    ))
    _cuda_gc()

    one_step_serial_s, stats_serial = phase.run_dual_phase_pipeline(
        model, tenant_a, tenant_b, payload_a, payload_b,
        overlap=False, steps=1, **phase_kwargs,
    )
    _cuda_gc()
    one_step_overlap_s, stats_overlap = phase.run_dual_phase_pipeline(
        model, tenant_a, tenant_b, payload_a, payload_b,
        overlap=True, steps=1, **phase_kwargs,
    )
    _cuda_gc()
    dual_phase_overlap_s, _ = phase.run_dual_phase_pipeline(
        model, tenant_a, tenant_b, payload_a, payload_b,
        overlap=True, steps=args.measure_steps, **phase_kwargs,
    )

    e2e_pair_serial = _aggregate_pair_timings(stats_serial.get("pair_timings") or [])
    e2e_pair_overlap = _aggregate_pair_timings(stats_overlap.get("pair_timings") or [])

    micro_rows = []
    sum_serial_ms = 0.0
    sum_overlap_ms = 0.0
    sum_saved_ms = 0.0
    sum_regress_ms = 0.0
    for label in ("A2B1", "B2A3", "A4B3", "B4A5", "A6B5", "B6A1"):
        s = pairs_micro[label]["serial"]
        o = pairs_micro[label]["overlap"]
        saved = s["pair_wall_ms"] - o["pair_wall_ms"]
        sum_serial_ms += s["pair_wall_ms"]
        sum_overlap_ms += o["pair_wall_ms"]
        if saved >= 0:
            sum_saved_ms += saved
        else:
            sum_regress_ms += -saved
        micro_rows.append({
            "pair": label,
            "comm_p": s.get("comm_phase_wall_ms"),
            "comp_p": s.get("comp_phase_wall_ms"),
            "serial_wall_ms": s["pair_wall_ms"],
            "overlap_wall_ms": o["pair_wall_ms"],
            "ideal_max_ms": s["ideal_max_wall_ms"],
            "serial_sum_ms": s["serial_sum_wall_ms"],
            "saved_vs_serial_ms": saved,
            "gap_vs_ideal_ms": o["pair_wall_ms"] - s["ideal_max_wall_ms"],
            "stream_contention_ms": o.get("stream_contention_ms"),
            "schedule_overhead_ms": o.get("schedule_overhead_ms"),
            "dual_stream_naive_ms": o.get("dual_stream_naive_wall_ms"),
        })

    num_layers = len(model.transformer_weights.blocks) if hasattr(model, "transformer_weights") else 40
    wan_a, _ = ov._bind_tenant(model, tenant_a)
    num_layers = len(wan_a.transformer_weights.blocks)

    # One-step E2E pair sums (all 40 layers, measured in pipeline)
    e2e_sum_serial_ms = sum(v["sum_wall_ms"] for v in e2e_pair_serial.values())
    e2e_sum_overlap_ms = sum(v["sum_wall_ms"] for v in e2e_pair_overlap.values())

    # Micro projection: layer-10 pair sum × 40 layers (ignores layer-0 cold start)
    proj_serial_ms = sum_serial_ms * num_layers
    proj_overlap_ms = sum_overlap_ms * num_layers

    per_tenant_step_b2b_ms = dual_b2b_s / max(args.measure_steps * 2, 1) * 1000.0
    per_tenant_step_overlap_ms = dual_phase_overlap_s / max(args.measure_steps * 2, 1) * 1000.0
    per_tenant_step_single_ms = single_s / max(args.measure_steps, 1) * 1000.0

    one_step_measured_ms = one_step_overlap_s * 1000.0
    one_step_serial_measured_ms = one_step_serial_s * 1000.0
    pair_sum_e2e_ms = e2e_sum_overlap_ms
    non_pair_overhead_ms = one_step_measured_ms - pair_sum_e2e_ms

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": "256x256",
        "layer_microbench": layer,
        "num_layers": num_layers,
        "measure_steps": args.measure_steps,
        "phase_flags": {"use_stream_ar_wait": True, "use_row_ready_event": True},
        "e2e_wall_s": {
            "single_transformer": single_s,
            "dual_b2b": dual_b2b_s,
            "one_step_phase_serial": one_step_serial_s,
            "one_step_phase_overlap": one_step_overlap_s,
            "dual_phase_overlap_4step": dual_phase_overlap_s,
        },
        "e2e_per_tenant_step_ms": {
            "single_infer": per_tenant_step_single_ms,
            "dual_b2b": per_tenant_step_b2b_ms,
            "phase_overlap_4step": per_tenant_step_overlap_ms,
        },
        "pairs_micro_layer10": micro_rows,
        "pairs_micro_totals": {
            "sum_serial_wall_ms": sum_serial_ms,
            "sum_overlap_wall_ms": sum_overlap_ms,
            "net_saved_ms": sum_serial_ms - sum_overlap_ms,
            "pairs_faster_ms": sum_saved_ms,
            "pairs_slower_ms": sum_regress_ms,
        },
        "one_step_e2e_pair_sums_ms": {
            "phase_serial": e2e_pair_serial,
            "phase_overlap": e2e_pair_overlap,
            "sum_serial_all_layers": e2e_sum_serial_ms,
            "sum_overlap_all_layers": e2e_sum_overlap_ms,
            "one_step_overlap_measured_ms": one_step_measured_ms,
            "non_pair_overhead_ms": non_pair_overhead_ms,
        },
        "projection_40_layers_from_L10": {
            "serial_ms": proj_serial_ms,
            "overlap_ms": proj_overlap_ms,
            "saved_ms": proj_serial_ms - proj_overlap_ms,
        },
        "gap_analysis": {
            "b2b_vs_overlap_4step_pct": 100.0 * (dual_phase_overlap_s - dual_b2b_s) / dual_b2b_s,
            "per_tenant_step_gap_ms": per_tenant_step_overlap_ms - per_tenant_step_b2b_ms,
            "decomposed_vs_monolithic_per_step_ms": per_tenant_step_overlap_ms - per_tenant_step_single_ms,
            "pair_overlap_net_saved_one_layer_ms": sum_serial_ms - sum_overlap_ms,
            "pair_regression_total_ms_per_layer": sum_regress_ms,
            "non_pair_overhead_one_step_ms": non_pair_overhead_ms,
        },
    }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

        ga = result["gap_analysis"]
        tot = result["pairs_micro_totals"]
        lines = [
            "# 256×256 — 6 Pair 计时 vs b2b 归因",
            "",
            f"稳态 layer **{layer}** micro-bench；E2E one-step 全 **{num_layers}** layer pair 求和。",
            f"phase flags: stream_ar_wait + row_ready_event | measure_steps: **{args.measure_steps}**",
            "",
            "## E2E 墙钟",
            "",
            "| 模式 | wall (s) | 每 tenant·step (ms) |",
            "|---|---:|---:|",
            f"| single infer (TP=2) | {single_s:.3f} | {per_tenant_step_single_ms:.1f} |",
            f"| **dual b2b** | **{dual_b2b_s:.3f}** | **{per_tenant_step_b2b_ms:.1f}** |",
            f"| one-step phase-serial | {one_step_serial_s:.3f} | {one_step_serial_measured_ms/2:.1f} |",
            f"| one-step phase-overlap | {one_step_overlap_s:.3f} | {one_step_measured_ms/2:.1f} |",
            f"| 4-step phase-overlap | {dual_phase_overlap_s:.3f} | {per_tenant_step_overlap_ms:.1f} |",
            "",
            f"- b2b vs 4-step overlap: **{ga['b2b_vs_overlap_4step_pct']:+.1f}%**",
            f"- 每 tenant·step gap (overlap − b2b): **{ga['per_tenant_step_gap_ms']:+.1f} ms**",
            "",
            "## 6 Pair @ layer 10（serial vs overlap wall）",
            "",
            "| pair | COMM ms | COMP ms | serial | overlap | ideal | saved | gap vs ideal | contention |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in micro_rows:
            lines.append(
                f"| {r['pair']} | {r['comm_p']:.2f} | {r['comp_p']:.2f} | "
                f"{r['serial_wall_ms']:.2f} | {r['overlap_wall_ms']:.2f} | {r['ideal_max_ms']:.2f} | "
                f"{r['saved_vs_serial_ms']:+.2f} | {r['gap_vs_ideal_ms']:.2f} | "
                f"{r.get('stream_contention_ms', 0):+.2f} |"
            )
        lines.extend([
            "",
            f"- **单层 6 pair 合计**: serial **{sum_serial_ms:.2f} ms** → overlap **{sum_overlap_ms:.2f} ms** "
            f"(净省 **{tot['net_saved_ms']:+.2f} ms**)",
            f"- 其中变快的 pair 合计省 **{tot['pairs_faster_ms']:.2f} ms**；"
            f"变慢的 pair 合计多 **{tot['pairs_slower_ms']:.2f} ms**",
            "",
            "## One-step E2E：40 layer 各 pair 求和 vs 实测",
            "",
            f"| | pair 求和 (ms) |",
            f"|---|---:|",
            f"| phase-serial 全 layer pair sum | {e2e_sum_serial_ms:.1f} |",
            f"| phase-overlap 全 layer pair sum | {e2e_sum_overlap_ms:.1f} |",
            f"| one-step overlap 实测 | {one_step_measured_ms:.1f} |",
            f"| **非 pair 开销** (实测 − pair sum) | **{non_pair_overhead_ms:.1f}** |",
            "",
            "## 为什么仍慢于 b2b",
            "",
            f"1. **分解路径比 model.infer 慢**：每 tenant·step overlap **{per_tenant_step_overlap_ms:.0f} ms** "
            f"vs single **{per_tenant_step_single_ms:.0f} ms** (+{ga['decomposed_vs_monolithic_per_step_ms']:.0f} ms)。",
            f"2. **6 pair 单层净省仅 {tot['net_saved_ms']:+.1f} ms**（×40 layer ≈ "
            f"{(tot['net_saved_ms'] * num_layers):+.0f} ms），不够抵消分解开销。",
            f"3. **部分 pair overlap 反而更慢**（B2A3/A6B5 等 attention 段），单层回归合计 **{sum_regress_ms:.1f} ms**。",
            f"4. **非 pair 开销** ~**{non_pair_overhead_ms:.0f} ms**/step（COMP-1 冷启动、scheduler、pre/finish_step、pipe state）。",
            f"5. **b2b 无调度税**：两次原生 infer，无 pump/barrier/双 tenant 切换。",
            "",
        ])
        md = out.with_suffix(".md")
        md.write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(result["pairs_micro_totals"], indent=2))
        print(json.dumps(result["gap_analysis"], indent=2))
        print(f"wrote {out}")
        print(f"wrote {md}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
