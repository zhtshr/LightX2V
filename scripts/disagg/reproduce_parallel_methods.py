#!/usr/bin/env python3
"""Reproduce the original Wan parallel implementations with warmed wall timings.

The configuration is explicit: no attention/quantization backend is substituted here.
PP/quad outputs are checked against serial execution using the same partition.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_transformer, set_config
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from scripts.disagg.run_phase1_transformer_bench import _init_distributed, _prepare_payload_on_device, _run_transformer_compute
from scripts.disagg.pp_interleaved_pipeline import PpTenantCtx, run_gpipe_dual_pipeline
from scripts.disagg.pp_sp_quad_overlap import QuadPpTenantCtx, run_gpipe_quad_sp_pipeline


def main():
    import faulthandler

    faulthandler.dump_traceback_later(900, repeat=True)
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", default="i2v")
    p.add_argument("--model-cls", default="wan2.2_moe")
    p.add_argument("--sp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--lps", type=int, default=2)
    p.add_argument("--quad", action="store_true")
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--tp-phase", action="store_true")
    p.add_argument("--safe-overlap", action="store_true")
    p.add_argument("--modes", default="", help="Comma-separated subset; validation references still run serially")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--profile", action="store_true", help="Export a warmed single-request rank0 CPU/CUDA trace after timing")
    p.add_argument("--sp-layout", choices=["base", "return", "notext", "both", "compare"], default="base")
    p.add_argument("--reference-latents", default="", help="Baseline final latents; create once, compare subsequent runs exactly")
    a = p.parse_args()
    if a.sp_layout not in ("base", "compare"):
        from scripts.disagg.sp_layout_experiment import install

        install(a.sp_layout)
    if a.safe_overlap:
        from scripts.disagg.reproduction_overlap_safety import install

        install()
    config = set_config(model_path=a.model, task=a.task, model_cls=a.model_cls, config_path=a.config)
    if a.tp > 1:
        config["parallel"] = {"tensor_p_size": a.tp}
    elif a.sp > 1:
        config["parallel"] = {"seq_p_size": a.sp, "seq_p_attn_type": "ulysses"}
    config["pp_layers_per_stage"] = a.lps
    if a.tp_phase:
        config["tp_norm_p2p"] = True
    _init_distributed(config)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    control = dist.new_group(backend="gloo") if dist.is_initialized() else None
    pp = bool(config.get("pipeline_parallel"))
    sp = bool(config.get("seq_parallel"))
    if pp:
        group = config["device_mesh"].get_group(mesh_dim="pipe_p")
        pp_rank, pp_size = dist.get_rank(group), dist.get_world_size(group)
    else:
        pp_rank, pp_size = 0, 1
    payload = _prepare_payload_on_device(torch.load(a.inputs, map_location="cpu", weights_only=False))
    expected = [16, (config["target_video_length"] - 1) // 4 + 1, config["target_height"] // 8, config["target_width"] // 8]
    assert list(payload["latent_shape"]) == expected, (payload["latent_shape"], expected)
    payloads = [dict(payload, seed=42 + i) for i in range(4 if a.quad else 2)]
    runner = load_wan_transformer(config)

    def serial(n):
        scheds = []
        for payload_i in payloads[:n]:
            sched = WanScheduler(config)
            runner.set_scheduler(sched)
            if pp:
                run_gpipe_dual_pipeline(runner, [PpTenantCtx(scheduler=sched, inputs=payload_i["inputs"])], [payload_i], group, pp_rank, pp_size, config["num_layers"], a.lps, seq_parallel=sp)
            else:
                _run_transformer_compute(sched, runner, payload_i)
            scheds.append(sched)
        return scheds

    def pipeline(n, overlap=False):
        scheds = [WanScheduler(config) for _ in range(n)]
        cls = QuadPpTenantCtx if n == 4 else PpTenantCtx
        tenants = [cls(scheduler=s, inputs=q["inputs"]) for s, q in zip(scheds, payloads)]
        args = (runner, tenants, payloads[:n], group, pp_rank, pp_size, config["num_layers"], a.lps)
        if n == 4:
            run_gpipe_quad_sp_pipeline(*args, seq_p_size=config["parallel"]["seq_p_size"], layout="dual", seq_parallel=sp, sp_overlap=overlap)
        else:
            run_gpipe_dual_pipeline(*args, seq_parallel=sp)
        return scheds

    modes = {"single": (1, lambda: serial(1)), "b2b": (len(payloads), lambda: serial(len(payloads)))}
    if pp:
        modes["pipeline"] = (2, lambda: pipeline(2))
    if a.quad:
        modes["quad_serial"] = (4, lambda: pipeline(4, False))
        modes["quad_overlap"] = (4, lambda: pipeline(4, True))

    if a.overlap or a.tp_phase:
        from scripts.disagg import run_phase3_dual_overlap_bench as ov

        def dual(overlap, phase=False):
            scheds = [WanScheduler(config), WanScheduler(config)]
            tenants = [ov.TenantCtx(str(i), sc, q["inputs"]) for i, (sc, q) in enumerate(zip(scheds, payloads))]
            if phase:
                from scripts.disagg import tp_phase_pipeline as phase_impl

                phase_impl.run_dual_phase_pipeline(
                    runner,
                    *tenants,
                    *payloads[:2],
                    overlap=overlap,
                    steps=config["infer_steps"],
                    bind_tenant=ov._bind_tenant,
                    ensure_block=ov._ensure_block,
                    preload_blocks=ov._preload_blocks,
                    pre_infer_tenant=ov._pre_infer_tenant,
                    finish_step_tenant=ov._finish_step_tenant,
                    capture_ti_snap=ov._capture_ti_snap,
                    time_fn=ov._time_fn,
                )
            else:
                ov._run_dual_a2a_pipeline(runner, *tenants, *payloads[:2], overlap=overlap, steps=config["infer_steps"])
            return scheds

        if a.overlap:
            modes["dual_serial"] = (2, lambda: dual(False))
            modes["dual_overlap"] = (2, lambda: dual(True))
        if a.tp_phase:
            modes["phase_serial"] = (2, lambda: dual(False, True))
            modes["phase_overlap"] = (2, lambda: dual(True, True))

    if a.sp_layout == "compare":
        from scripts.disagg.sp_layout_experiment import install

        def layout_run(variant):
            install(variant)
            return serial(1)

        modes = {"base": (1, lambda: layout_run("base")), "both": (1, lambda: layout_run("both"))}

    if a.modes:
        selected = a.modes.split(",")
        unknown = set(selected) - modes.keys()
        if unknown:
            raise ValueError(f"Unknown modes: {unknown}")
        modes = {name: modes[name] for name in selected}

    # New schedulers reset each request's generator, making repeated seeds comparable.
    refs = serial(len(payloads))
    reference = [s.latents.detach().float().clone() for s in refs] if pp_rank == 0 else []
    del refs
    if a.reference_latents and rank == 0:
        refpath = Path(a.reference_latents)
        if refpath.exists():
            baseline = torch.load(refpath, map_location="cpu", weights_only=True)
            exact = all(torch.equal(x.cpu(), y) for x, y in zip(reference, baseline))
            print("CROSS_BASELINE_EXACT", exact, flush=True)
            if not exact:
                raise RuntimeError("Layout experiment differs from original baseline")
        else:
            torch.save([x.cpu() for x in reference], refpath)
    validation = {}
    for name, (n, fn) in modes.items():
        scheds = fn()  # also warms every execution path
        torch.cuda.synchronize()
        errors = []
        if pp_rank == 0:
            for i, s in enumerate(scheds):
                x, y = s.latents.float(), reference[i]
                errors.append({"relative_l2": float((x - y).norm() / y.norm().clamp_min(1e-12)), "max_abs": float((x - y).abs().max()), "finite": bool(torch.isfinite(x).all())})
        valid = all(e["finite"] and (e["max_abs"] == 0 if a.sp_layout == "compare" else e["relative_l2"] < 0.02) for e in errors)
        flag = torch.tensor(int(valid))
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=control)
        validation[name] = {"passed": bool(flag.item()), "rank0_errors": errors if rank == 0 else []}
        if rank == 0:
            print("VALIDATION", name, validation[name], flush=True)
        del scheds
    del reference
    if rank == 0:
        Path(a.output).with_suffix(".validation.json").write_text(json.dumps(validation, indent=2))
    samples = {name: [] for name in modes}
    peak = {name: [] for name in modes}
    for iteration in range(a.samples):
        names = list(modes)
        if iteration % 2:
            names.reverse()
        for name in names:
            if not validation[name]["passed"]:
                continue
            if dist.is_initialized():
                dist.barrier(group=control)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            scheds = modes[name][1]()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            stats = torch.tensor([elapsed, torch.cuda.max_memory_allocated() / 1024**3], dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=control)
            samples[name].append(stats[0].item())
            peak[name].append(stats[1].item())
            if rank == 0:
                print("SAMPLE", name, iteration, samples[name][-1], flush=True)
            del scheds
    if a.profile:
        if dist.is_initialized():
            dist.barrier(group=control)
        torch.cuda.synchronize()
        if rank == 0:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                profiled = serial(1)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(Path(a.output).with_suffix(".trace.json")))
            del profiled
        else:
            profiled = serial(1)
            torch.cuda.synchronize()
            del profiled
        if dist.is_initialized():
            dist.barrier(group=control)
    if rank == 0:
        result = {
            "args": vars(a),
            "config_requested": json.loads(Path(a.config).read_text()),
            "world_size": world,
            "latent_shape": list(payload["latent_shape"]),
            "attention": config["self_attn_1_type"],
            "quantization": config.get("dit_quant_scheme", "Default"),
            "cpu_offload": config.get("cpu_offload", False),
            "validation": validation,
            "results": {},
        }
        for name, values in samples.items():
            mean = sum(values) / len(values) if values else None
            result["results"][name] = {
                "requests": modes[name][0],
                "samples_s": values,
                "mean_s": mean,
                "requests_per_s": modes[name][0] / mean if mean else None,
                "peak_allocated_gib": max(peak[name]) if peak[name] else None,
            }
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if all(v["passed"] for v in validation.values()) else 2


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
