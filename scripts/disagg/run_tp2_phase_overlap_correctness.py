#!/usr/bin/env python3
"""Numerical check: phase-overlap vs phase-serial with optional pipeline configure."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from argparse import Namespace
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


def _run_once(phase, model, ta, tb, ov, *, overlap: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ta.pipe_states = {}
    tb.pipe_states = {}
    ta.scheduler.step_pre(step_index=0)
    tb.scheduler.step_pre(step_index=0)
    ov._pre_infer_tenant(model, ta)
    ov._pre_infer_tenant(model, tb)
    phase.phase_pipeline_main_blocks(
        model, ta, tb,
        ov._bind_tenant, ov._ensure_block, ov._preload_blocks, ov._capture_ti_snap,
        overlap=overlap,
    )
    return ta.x.detach().clone(), tb.x.detach().clone()


def main() -> int:
    here = Path(__file__).parent
    p1 = _load("p1", here / "run_phase1_transformer_bench.py")
    ov = _load("ov", here / "run_phase3_dual_overlap_bench.py")
    phase = _load("ph", here / "tp_phase_pipeline.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="configs/disagg/baseline/wan22_moe_i2v_256_tp2_fair.json")
    parser.add_argument("--inputs_cache", default="save_results/optimization_study/phase1_encoder_inputs_256x256.pt")
    parser.add_argument("--use_comm_p2p", action="store_true", default=False)
    parser.add_argument("--use_chunked_ar", action="store_true", default=False)
    args = parser.parse_args()

    ns = Namespace(
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
    config["tp_norm_p2p"] = True
    seed_all(42)
    p1._init_distributed(config)
    phase.configure_phase_pipeline(
        use_comm_p2p=args.use_comm_p2p,
        use_chunked_ar=args.use_chunked_ar,
    )

    payload_a = p1._prepare_payload_on_device(
        p1._prepare_inputs_cache(
            config=config, cache_path=Path(args.inputs_cache),
            prompt=ns.prompt, image_path=ns.image_path, seed=42,
            force=False, task="i2v", negative_prompt="",
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
        s.prepare(
            seed=int(p["seed"]),
            latent_shape=p["latent_shape"],
            image_encoder_output=p["image_encoder_output"],
        )
        t.inputs = p["inputs"]

    xa_s, xb_s = _run_once(phase, model, ta, tb, ov, overlap=False)
    seed_all(42)
    for t, p, s in ((ta, payload_a, sa), (tb, payload_b, sb)):
        s.prepare(
            seed=int(p["seed"]),
            latent_shape=p["latent_shape"],
            image_encoder_output=p["image_encoder_output"],
        )
        t.inputs = p["inputs"]
    xa_o, xb_o = _run_once(phase, model, ta, tb, ov, overlap=True)

    da = (xa_s - xa_o).abs().max().item()
    db = (xb_s - xb_o).abs().max().item()
    if is_main_process():
        print(f"max_abs_diff A: {da:.6e}  B: {db:.6e}  ok={da < 1e-2 and db < 1e-2}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if da < 1e-2 and db < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
