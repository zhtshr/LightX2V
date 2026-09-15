#!/usr/bin/env python3
"""One-step wall: model.infer b2b vs 6-phase serial vs 6-phase overlap."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import is_main_process, seed_all


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("ph", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--output_json", default="save_results/optimization_study/wan22_256_tp2_phase_overhead_compare.json")
    args = parser.parse_args()

    ns = argparse.Namespace(
        config_json=args.config_json,
        model_path="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models",
        task="i2v",
        model_cls="wan2.2_moe",
        image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        prompt="t",
        seed=42,
        seq_p_size=1,
        tensor_p_size=2,
        inputs_cache=args.inputs_cache,
        refresh_inputs_cache=False,
        negative_prompt="",
    )
    config = p1._load_config(ns)
    seed_all(42)
    p1._init_distributed(config)

    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(
            config=config,
            cache_path=Path(args.inputs_cache),
            prompt=ns.prompt,
            image_path=ns.image_path,
            seed=42,
            force=False,
            task="i2v",
            negative_prompt="",
        )
    )
    payload_b = p1._prepare_payload_on_device({
        "seed": 43,
        "latent_shape": payload_a["latent_shape"],
        "image_encoder_output": payload_a["image_encoder_output"],
        "inputs": payload_a["inputs"],
    })

    model = load_wan_transformer(config)
    sa, sb = WanScheduler(config), WanScheduler(config)
    ta = ov.TenantCtx("A", sa, payload_a["inputs"])
    tb = ov.TenantCtx("B", sb, payload_b["inputs"])
    for t, p, s in ((ta, payload_a, sa), (tb, payload_b, sb)):
        t.scheduler.prepare(
            seed=int(p["seed"]),
            latent_shape=p["latent_shape"],
            image_encoder_output=p["image_encoder_output"],
        )
        t.inputs = p["inputs"]
        s.step_pre(step_index=0)
        ov._pre_infer_tenant(model, t)

    ov._run_denoise_serial(model, sa, payload_a)

    def _prep_step() -> None:
        sa.step_pre(step_index=0)
        sb.step_pre(step_index=0)
        ov._pre_infer_tenant(model, ta)
        ov._pre_infer_tenant(model, tb)

    _prep_step()
    infer_a_ms = _wall_ms(lambda: (
        model.set_scheduler(sa),
        model.infer(payload_a["inputs"]),
        sa.step_post(),
    ))

    _prep_step()
    infer_b_ms = _wall_ms(lambda: (
        model.set_scheduler(sb),
        model.infer(payload_b["inputs"]),
        sb.step_post(),
    ))

    _prep_step()
    b2b_ms = _wall_ms(lambda: (
        model.set_scheduler(sa),
        model.infer(payload_a["inputs"]),
        sa.step_post(),
        sb.step_pre(step_index=0),
        ov._pre_infer_tenant(model, tb),
        model.set_scheduler(sb),
        model.infer(payload_b["inputs"]),
        sb.step_post(),
    ))

    _prep_step()
    phase_serial_ms = _wall_ms(lambda: phase.phase_pipeline_main_blocks(
        model, ta, tb,
        ov._bind_tenant, ov._ensure_block, ov._preload_blocks, ov._capture_ti_snap,
        overlap=False,
    ))

    _prep_step()
    phase_overlap_ms = _wall_ms(lambda: phase.phase_pipeline_main_blocks(
        model, ta, tb,
        ov._bind_tenant, ov._ensure_block, ov._preload_blocks, ov._capture_ti_snap,
        overlap=True,
    ))

    wan, ti = ov._bind_tenant(model, ta)
    ov._preload_blocks(ta, wan, ti, len(wan.transformer_weights.blocks))
    layer = 10
    six_phase_ms = sum(
        _wall_ms(lambda ph=ph: phase.run_tenant_phase(
            model, ta, layer, ph,
            ov._bind_tenant, ov._ensure_block, ov._capture_ti_snap,
        ))
        for ph in range(1, 7)
    )

    per_req_serial = phase_serial_ms / 2.0
    out = {
        "one_step_infer_A_ms": infer_a_ms,
        "one_step_infer_B_ms": infer_b_ms,
        "one_step_dual_b2b_ms": b2b_ms,
        "one_step_phase_serial_ms": phase_serial_ms,
        "one_step_phase_overlap_ms": phase_overlap_ms,
        "six_phase_one_layer_ms": six_phase_ms,
        "per_request_phase_serial_ms": per_req_serial,
        "decomposed_vs_infer_pct": 100.0 * (per_req_serial - infer_a_ms) / infer_a_ms,
        "overlap_vs_serial_pct": 100.0 * (phase_overlap_ms - phase_serial_ms) / phase_serial_ms,
        "overlap_vs_b2b_pct": 100.0 * (phase_overlap_ms - b2b_ms) / b2b_ms,
        "serial_vs_b2b_pct": 100.0 * (phase_serial_ms - b2b_ms) / b2b_ms,
        "projected_4step_dual_b2b_s": b2b_ms * 4 / 1000.0,
        "projected_4step_phase_overlap_s": phase_overlap_ms * 4 / 1000.0,
    }

    if is_main_process():
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        print(f"wrote {path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
