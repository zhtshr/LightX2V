#!/usr/bin/env python3
"""Profile PP=2 per-step compute vs blocking P2P latency (480x832 MoE)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.networks.wan.infer.pipeline_parallel import (
    recv_activation,
    recv_noise_pred,
    recv_pre_metadata,
    send_activation,
    send_noise_pred,
    send_pre_metadata,
)
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

from scripts.disagg.run_phase1_transformer_bench import (
    _init_distributed,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)


@dataclass
class StepProfile:
    segments_ms: dict[str, float] = field(default_factory=dict)
    tensor_bytes: dict[str, int] = field(default_factory=dict)


class CudaTimer:
    def __init__(self) -> None:
        self._s: torch.cuda.Event | None = None

    def start(self) -> None:
        self._s = torch.cuda.Event(enable_timing=True)
        self._s.record()

    def stop_ms(self) -> float:
        assert self._s is not None
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        return float(self._s.elapsed_time(end))


def _infer_pp_profiled(wan_model: Any, inputs: dict, pp_rank: int, pp_size: int, pp_group) -> StepProfile:
    last = pp_size - 1
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    prof = StepProfile()
    timer = CudaTimer()

    wan_model.scheduler.infer_condition = True

    if pp_rank == 0:
        timer.start()
        pre = wan_model.pre_infer.infer(wan_model.pre_weight, inputs)
        prof.segments_ms["pre_infer"] = timer.stop_ms()

        timer.start()
        send_pre_metadata(pre, last, pp_group)
        prof.segments_ms["p2p_send_meta"] = timer.stop_ms()

        wan_model.transformer_infer.cos_sin = pre.cos_sin
        wan_model.transformer_infer.reset_infer_states()
        timer.start()
        x = wan_model.transformer_infer.infer_main_blocks(wan_model.transformer_weights.blocks, pre)
        prof.segments_ms["blocks_stage0"] = timer.stop_ms()
        prof.tensor_bytes["activation"] = int(x.numel() * x.element_size())

        timer.start()
        send_activation(x, last, pp_group)
        prof.segments_ms["p2p_send_activation"] = timer.stop_ms()

        timer.start()
        noise_pred = recv_noise_pred(last, pp_group, device)
        prof.segments_ms["p2p_recv_noise_pred"] = timer.stop_ms()
        prof.tensor_bytes["noise_pred"] = int(noise_pred.numel() * noise_pred.element_size())

        wan_model.scheduler.noise_pred = noise_pred
        return prof

    if pp_rank == last:
        timer.start()
        pre = recv_pre_metadata(0, pp_group, device)
        prof.segments_ms["p2p_recv_meta"] = timer.stop_ms()

        timer.start()
        x = recv_activation(0, pp_group, device)
        prof.segments_ms["p2p_recv_activation"] = timer.stop_ms()
        prof.tensor_bytes["activation"] = int(x.numel() * x.element_size())

        pre.x = x
        wan_model.transformer_infer.cos_sin = pre.cos_sin
        wan_model.transformer_infer.reset_infer_states()

        timer.start()
        x = wan_model.transformer_infer.infer_main_blocks(wan_model.transformer_weights.blocks, pre)
        prof.segments_ms["blocks_stage1"] = timer.stop_ms()

        timer.start()
        x = wan_model.transformer_infer.infer_non_blocks(wan_model.transformer_weights, x, pre.embed)
        noise_pred = wan_model.post_infer.infer(x, pre)[0]
        prof.segments_ms["head_post"] = timer.stop_ms()
        prof.tensor_bytes["noise_pred"] = int(noise_pred.numel() * noise_pred.element_size())

        timer.start()
        send_noise_pred(noise_pred, 0, pp_group)
        prof.segments_ms["p2p_send_noise_pred"] = timer.stop_ms()

        wan_model.scheduler.noise_pred = noise_pred
        return prof

    raise RuntimeError(f"bad pp_rank={pp_rank}")


def _sum_comm_ms(segments: dict[str, float]) -> float:
    keys = (
        "p2p_send_meta",
        "p2p_recv_meta",
        "p2p_send_activation",
        "p2p_recv_activation",
        "p2p_send_noise_pred",
        "p2p_recv_noise_pred",
    )
    return sum(segments.get(k, 0.0) for k in keys)


def _sum_compute_ms(segments: dict[str, float]) -> float:
    keys = ("pre_infer", "blocks_stage0", "blocks_stage1", "head_post")
    return sum(segments.get(k, 0.0) for k in keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_pp2_bench.json")
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/lightx2v/Wan2.2-Distill-Models")
    parser.add_argument("--inputs_cache", default="/root/zht/LightX2V/save_results/optimization_study/phase1_encoder_inputs.pt")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=4)
    parser.add_argument("--output_json", default="/root/zht/LightX2V/save_results/optimization_study/wan22_480_pp2_p2p_profile.json")
    args = parser.parse_args()

    config = set_config(
        model_path=args.model_path,
        task="i2v",
        model_cls="wan2.2_moe",
        config_path=args.config_json,
    )
    config["parallel"] = {"pipe_p_size": 2}
    seed_all(42)
    _init_distributed(config)

    rank = dist.get_rank()
    pp_group = config["device_mesh"].get_group(mesh_dim="pipe_p")
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)

    payload = _prepare_payload_on_device(
        _prepare_inputs_cache(
            config=config,
            cache_path=Path(args.inputs_cache),
            prompt="profile",
            image_path="/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
            seed=42,
        )
    )
    inputs = payload["inputs"]

    runner = load_wan_transformer(config)
    assert isinstance(runner, MultiModelStruct)
    scheduler = WanScheduler(config)
    runner.set_scheduler(scheduler)
    scheduler.prepare(
        seed=payload["seed"],
        latent_shape=payload["latent_shape"],
        image_encoder_output=payload["image_encoder_output"],
    )

    runner.get_current_model_index()
    wan = runner.model[runner.cur_model_index]
    assert wan.use_pp

    step_profiles: list[dict[str, Any]] = []
    infer_steps = scheduler.infer_steps
    for step_index in range(infer_steps):
        scheduler.step_pre(step_index=step_index)
        _sync_device()
        wall_s = torch.cuda.Event(enable_timing=True)
        wall_e = torch.cuda.Event(enable_timing=True)
        wall_s.record()
        prof = _infer_pp_profiled(wan, inputs, pp_rank, pp_size, pp_group)
        wall_e.record()
        torch.cuda.synchronize()
        wall_ms = float(wall_s.elapsed_time(wall_e))
        if step_index >= args.warmup_steps:
            rec = {
                "step_index": step_index,
                "pp_rank": pp_rank,
                "wall_cuda_ms": wall_ms,
                "segments_ms": prof.segments_ms,
                "tensor_bytes": prof.tensor_bytes,
                "comm_cuda_ms": _sum_comm_ms(prof.segments_ms),
                "compute_cuda_ms": _sum_compute_ms(prof.segments_ms),
            }
            step_profiles.append(rec)
        scheduler.step_post()
        _sync_device()

    local = {
        "pp_rank": pp_rank,
        "steps": step_profiles,
    }
    if pp_rank == 0:
        gathered = [None, None]
    else:
        gathered = None
    if pp_rank == 0:
        dist.gather_object(local, gathered, dst=0)
    else:
        dist.gather_object(local, None, dst=0)

    if is_main_process():
        r0, r1 = gathered
        steps = len(r0["steps"])
        avg = lambda rank_data, key: sum(s[key] for s in rank_data["steps"]) / steps

        def avg_seg(rank_data, seg: str) -> float:
            return sum(s["segments_ms"].get(seg, 0.0) for s in rank_data["steps"]) / steps

        comm_r0 = avg(r0, "comm_cuda_ms")
        comm_r1 = avg(r1, "comm_cuda_ms")
        comp_r0 = avg(r0, "compute_cuda_ms")
        comp_r1 = avg(r1, "compute_cuda_ms")
        wall_r0 = avg(r0, "wall_cuda_ms")
        wall_r1 = avg(r1, "wall_cuda_ms")

        # Serial PP critical path estimate (rank0 then rank1 overlap only at P2P handoff)
        serial_est_ms = (
            avg_seg(r0, "pre_infer")
            + avg_seg(r0, "blocks_stage0")
            + avg_seg(r0, "p2p_send_meta")
            + avg_seg(r0, "p2p_send_activation")
            + avg_seg(r1, "p2p_recv_meta")
            + avg_seg(r1, "p2p_recv_activation")
            + avg_seg(r1, "blocks_stage1")
            + avg_seg(r1, "head_post")
            + avg_seg(r1, "p2p_send_noise_pred")
            + avg_seg(r0, "p2p_recv_noise_pred")
        )

        act_mb = r0["steps"][0]["tensor_bytes"].get("activation", 0) / 1e6
        out = {
            "description": "PP=2 per-step cuda-event profile; comm segments are blocking dist.send/recv",
            "resolution": f"{config['target_height']}x{config['target_width']}",
            "infer_steps_measured": steps,
            "per_rank_avg_ms": {
                "rank0_wall": wall_r0,
                "rank1_wall": wall_r1,
                "rank0_compute": comp_r0,
                "rank1_compute": comp_r1,
                "rank0_comm": comm_r0,
                "rank1_comm": comm_r1,
            },
            "per_segment_avg_ms": {
                "rank0": {k: avg_seg(r0, k) for k in r0["steps"][0]["segments_ms"]},
                "rank1": {k: avg_seg(r1, k) for k in r1["steps"][0]["segments_ms"]},
            },
            "activation_mb": act_mb,
            "comm_total_both_ranks_ms": comm_r0 + comm_r1,
            "compute_total_both_ranks_ms": comp_r0 + comp_r1,
            "serial_critical_path_est_ms": serial_est_ms,
            "full_denoise_comm_est_s": (comm_r0 + comm_r1) * steps / 1000.0,
            "full_denoise_serial_est_s": serial_est_ms * steps / 1000.0,
            "reference": {
                "pp2_single_transformer_s": 70.0,
                "tp2_no_offload_transformer_s": 58.0,
                "tp1_offload_transformer_s": 73.0,
            },
            "raw": {"rank0": r0, "rank1": r1},
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=2))
        print(f"wrote {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
