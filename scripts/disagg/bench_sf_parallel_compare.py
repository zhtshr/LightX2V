"""SF parallel correctness, denoise, CPU/CUDA trace and warm text-to-video E2E."""

import argparse
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from lightx2v.common.kvcache.base import BaseKVCachePool
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_transformer, load_wan_text_encoder, load_wan_vae_decoder
from lightx2v.utils.utils import seed_all
from scripts.disagg.run_sf_transformer_sp_bench import _load_config, _init_distributed, _move_tensor_tree, _run_device, _run_sf_transformer_compute, DEFAULT_PROMPT


def torch_partial(q, k, v, *, softmax_scale):
    # Same Torch FlashAttention backend as the Ulysses SDPA path; expose LSE.
    result = torch.ops.aten._scaled_dot_product_flash_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 0.0, False, False, scale=softmax_scale)
    return result[0].transpose(1, 2).contiguous(), result[1], None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="models/Wan-AI/Wan2.1-T2V-14B")
    p.add_argument("--config_json", default="save_results/sf_14b_480p_scaling/config.json")
    p.add_argument("--seq_p_size", type=int, required=True)
    p.add_argument("--seq_p_attn_type", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--diagnostic-allow-drift", action="store_true")
    p.add_argument("--stripe-optimization", choices=["base", "local", "formc"], default="base")
    p.add_argument("--validation-only", action="store_true")
    a = p.parse_args()
    os.environ["LIGHTX2V_STRIPE_LOCAL_MERGE"] = "1" if a.stripe_optimization == "local" else "0"
    os.environ["LIGHTX2V_STRIPE_PARTIAL_OUT"] = "1" if a.stripe_optimization == "formc" else "0"
    os.environ["LIGHTX2V_STRIPE_FORMC_HIER_UNTIL"] = "-1"
    a.pipe_p_size = 1
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = _load_config(a)
    _init_distributed(config)
    rank = dist.get_rank()
    control = dist.new_group(backend="gloo")
    BaseKVCachePool._stripe_local_flash = staticmethod(torch_partial)
    payload = _move_tensor_tree(torch.load("save_results/sf_14b_480p_scaling/encoder_inputs.pt", weights_only=False), _run_device())
    shape = [16, (config["target_video_length"] - 1) // config["vae_stride"][0] + 1, config["target_height"] // config["vae_stride"][1], config["target_width"] // config["vae_stride"][2]]
    payload["latent_shape"] = shape
    payload["inputs"]["latent_shape"] = shape
    model = load_wan_transformer(config)

    def run():
        seed_all(42)
        sched = load_wan_scheduler(config)
        model.set_scheduler(sched)
        _run_sf_transformer_compute(model, sched, config, payload, include_rerun=True)
        return sched.latents

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(group=control)

    def maximum(v):
        x = torch.tensor(v, dtype=torch.float64)
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=control)
        return x.tolist()

    with torch.inference_mode():
        lat = run()
        barrier()
        reference_path = out.parent / f"p{a.seq_p_size}_ulysses_latents.pt"
        validation = {}
        if rank == 0:
            if a.seq_p_attn_type == "ulysses":
                torch.save(lat.cpu(), reference_path)
            ref = torch.load(reference_path, map_location=lat.device, weights_only=True).float()
            rel = float((lat.float() - ref).norm() / ref.norm())
            validation = {
                "finite": bool(torch.isfinite(lat).all()),
                "relative_l2": rel,
                "max_abs": float((lat.float() - ref).abs().max()),
                "threshold": 0.02,
                "passed": bool(torch.isfinite(lat).all()) and rel < 0.02,
            }
            if a.seq_p_attn_type == "stripe":
                stripe_ref = out.parent / f"p{a.seq_p_size}_stripe_baseline_latents.pt"
                if a.stripe_optimization == "base":
                    torch.save(lat.cpu(), stripe_ref)
                if stripe_ref.exists():
                    original = torch.load(stripe_ref, map_location=lat.device, weights_only=True)
                    validation["stripe_baseline_exact"] = torch.equal(lat, original)
                    validation["stripe_baseline_relative_l2"] = float((lat.float() - original.float()).norm() / original.float().norm())
                    if a.stripe_optimization != "base" and not validation["stripe_baseline_exact"]:
                        raise RuntimeError("Optimization is not exactly equal to original stripe")
            print("VALIDATION", validation, flush=True)
        ok = [validation.get("passed")]
        dist.broadcast_object_list(ok, src=0, group=control)
        if not ok[0] and not a.diagnostic_allow_drift:
            if rank == 0:
                out.write_text(json.dumps({"args": vars(a), "validation": validation}, indent=2))
            dist.destroy_process_group()
            return
        actual_frames = (lat.shape[1] - 1) * config["vae_stride"][0] + 1
        if a.validation_only:
            if rank == 0:
                out.write_text(json.dumps({"args": vars(a), "validation": validation, "actual_frames": actual_frames}, indent=2))
            dist.destroy_process_group()
            return
        samples = []
        for i in range(a.samples):
            barrier()
            t = time.perf_counter()
            lat = run()
            dt = maximum([time.perf_counter() - t])[0]
            samples.append(dt)
            if rank == 0:
                print("DENOISE", i, dt, flush=True)
        barrier()
        if rank == 0:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                lat = run()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out.with_suffix(".trace.json")))
        else:
            lat = run()
        barrier()
        # Encode/decode on rank0 with resident weights, identical across SP methods.
        if rank == 0:
            ec = dict(config)
            ec.update(parallel=False, load_from_rank0=False, t5_cpu_offload=False, vae_cpu_offload=False)
            encoder = load_wan_text_encoder(ec)[0]
            decoder = load_wan_vae_decoder(ec)
        barrier()
        e2e = []
        for i in range(a.samples + 1):
            barrier()
            start = time.perf_counter()
            enc = 0.0
            dec = 0.0
            if rank == 0:
                t = time.perf_counter()
                contexts = encoder.infer([DEFAULT_PROMPT])
                context = torch.stack([torch.cat([u, u.new_zeros(config["text_len"] - u.size(0), u.size(1))]) for u in contexts])
                torch.cuda.synchronize()
                enc = time.perf_counter() - t
                payload["inputs"]["text_encoder_output"]["context"].copy_(context)
            dist.broadcast(payload["inputs"]["text_encoder_output"]["context"], src=0)
            lat = run()
            if rank == 0:
                t = time.perf_counter()
                video = decoder.decode(lat)
                torch.cuda.synchronize()
                dec = time.perf_counter() - t
                assert list(video.shape) == [1, 3, actual_frames, config["target_height"], config["target_width"]], video.shape
                assert torch.isfinite(video).all()
                del video
            barrier()
            elapsed = time.perf_counter() - start
            if rank == 0:
                print("E2E", i, elapsed, enc, dec, flush=True)
                if i:
                    e2e.append({"total_s": elapsed, "t5_s": enc, "vae_s": dec})
        if rank == 0:
            out.write_text(
                json.dumps(
                    {
                        "args": vars(a),
                        "validation": validation,
                        "actual_frames": actual_frames,
                        "num_chunks": lat.shape[1] // config["ar_config"]["num_frame_per_chunk"],
                        "denoise_samples_s": samples,
                        "denoise_mean_s": sum(samples) / len(samples),
                        "e2e_samples": e2e,
                        "e2e_mean_s": sum(r["total_s"] for r in e2e) / len(e2e),
                        "attention_backend": "torch flash for all methods; stripe LSE exposed via aten",
                        "scope": "warm resident weights; T5 rank0 + broadcast + full AR/rerun + rank0 full VAE decode; no file encoding/load/queue",
                    },
                    indent=2,
                )
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
