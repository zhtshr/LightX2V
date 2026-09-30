#!/usr/bin/env python3
"""Repeat and validate the SF dual-request A2A benchmark on one SP group."""
import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('sf_dual', ROOT / 'scripts/disagg/run_sf_transformer_dual_overlap_bench.py')
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def deterministic_scheduler(factory, config):
    # Validation only: scheduler.prepare currently ignores its seed argument.
    # Per-step seeds make the stochastic trajectory independent of request order.
    scheduler = factory(config)
    prepare, post = scheduler.prepare, scheduler.step_post
    request_seed = 0
    def seeded_prepare(*, seed, **kwargs):
        nonlocal request_seed
        request_seed = int(seed)
        torch.cuda.manual_seed(request_seed)
        return prepare(seed=seed, **kwargs)
    def seeded_post():
        torch.cuda.manual_seed(request_seed * 10000 + scheduler.seg_index * 100 + scheduler.step_index)
        return post()
    scheduler.prepare, scheduler.step_post = seeded_prepare, seeded_post
    return scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_p_size', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--measure_iters', type=int, default=3)
    parser.add_argument('--output_dir', default=str(ROOT / 'save_results/sf_14b_480p_dual_overlap'))
    args = parser.parse_args()
    args.model_path = str(ROOT / 'models/Wan-AI/Wan2.1-T2V-14B')
    args.config_json = str(ROOT / 'save_results/sf_14b_480p_scaling/config.json')
    args.seq_p_attn_type = 'ulysses'
    config = b._load_config(args)
    b._validate_seq_p(config, args.seq_p_size)
    b._init_distributed(config)
    p3 = b._load_phase3_module()
    payload = torch.load(ROOT / 'save_results/sf_14b_480p_scaling/encoder_inputs.pt', map_location='cpu', weights_only=False)
    a = b._move_tensor_tree(dict(payload, seed=42), b._run_device())
    c = b._move_tensor_tree(dict(payload, seed=43), b._run_device())
    model = b.load_wan_transformer(config)
    factory = b.load_wan_scheduler
    scheduler = factory(config)
    model.set_scheduler(scheduler)
    validation = {}
    meta = {}

    if args.seq_p_size > 1:
        b.load_wan_scheduler = lambda cfg: deterministic_scheduler(factory, cfg)
        references = []
        for item in (a, c):
            sc = b.load_wan_scheduler(config)
            model.set_scheduler(sc)
            b._run_sf_serial(model, sc, config, item, include_rerun=True)
            references.append(sc.latents.detach().clone())
        ta, tb, meta = b._prepare_tenants(model, config, a, c)
        for overlap in (False, True):
            _, stats = b._run_sf_dual_a2a(p3, model, ta, tb, a, c, meta, overlap=overlap, include_rerun=True)
            checks = []
            for tenant, ref in zip((ta, tb), references):
                actual = tenant.scheduler.latents.float()
                ref = ref.float()
                err = actual - ref
                rel = float(err.norm() / ref.norm().clamp_min(1e-12))
                cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0))
                finite = bool(actual.isfinite().all())
                checks.append(dict(request=tenant.name, finite=finite, max_abs=float(err.abs().max()), relative_l2=rel, cosine=cosine))
            valid = all(x['finite'] and x['relative_l2'] < 0.02 and x['cosine'] > 0.999 for x in checks)
            flag = torch.tensor(int(valid), device=b._run_device())
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            validation['overlap' if overlap else 'decomposed_serial'] = dict(passed=bool(flag), requests=checks, stats=stats)
            if b.is_main_process():
                print('VALIDATION', 'overlap' if overlap else 'serial', json.dumps(validation), flush=True)
            if not bool(flag):
                out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
                if b.is_main_process(): (out / f'p{args.seq_p_size}_validation_failed.json').write_text(json.dumps(validation, indent=2))
                raise RuntimeError('Output validation failed; refusing to report throughput')
        del references, ta, tb, tenant, ref, actual, err
        b.DisaggSFKVCacheManager.teardown(model)
        b.load_wan_scheduler = factory
        model.set_scheduler(scheduler)
        torch.cuda.empty_cache()

    # Include scheduler/cache setup and final synchronization in every wall time.
    def single():
        model.set_scheduler(scheduler)
        return b._run_sf_serial(model, scheduler, config, a, include_rerun=True)

    def back_to_back():
        model.set_scheduler(scheduler)
        b._run_sf_serial(model, scheduler, config, a, include_rerun=True)
        return b._run_sf_serial(model, scheduler, config, c, include_rerun=True)

    def dual(overlap):
        ta, tb, meta = b._prepare_tenants(model, config, a, c)
        _, stats = b._run_sf_dual_a2a(p3, model, ta, tb, a, c, meta, overlap=overlap, include_rerun=True)
        b.DisaggSFKVCacheManager.teardown(model)
        return dict(meta, stats=stats)

    modes = {'single': single, 'back_to_back': back_to_back}
    if args.seq_p_size > 1:
        modes.update(decomposed_serial=lambda: dual(False), overlap=lambda: dual(True))
    samples = {name: [] for name in modes}
    peaks = {name: [] for name in modes}
    stats = {}
    for name, fn in modes.items():
        for _ in range(args.warmup): fn()
        if b.is_main_process(): print('WARMED', name, flush=True)
    for iteration in range(args.measure_iters):
        # Alternate mode order to reduce order/thermal bias.
        order = list(modes) if iteration % 2 == 0 else list(reversed(modes))
        for name in order:
            b._sync(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            meta = modes[name]()
            b._sync(); elapsed = time.perf_counter() - start
            measure = torch.tensor([elapsed, torch.cuda.max_memory_allocated() / 2**30], device=b._run_device(), dtype=torch.float64)
            if dist.is_initialized(): dist.all_reduce(measure, op=dist.ReduceOp.MAX)
            samples[name].append(float(measure[0])); peaks[name].append(float(measure[1]))
            stats[name] = meta.get('stats')
            if b.is_main_process(): print('SAMPLE', iteration, name, samples[name][-1], flush=True)
    frames = (meta['num_output_frames'] - 1) * config['vae_stride'][0] + 1
    result = dict(sp=args.seq_p_size, model='Krea SF 14B', resolution='832x480', frames_per_request=frames,
                  dtype='BF16', attention=config['self_attn_1_type'], warmup_per_mode=args.warmup,
                  measure_iters=args.measure_iters, requests_seeds=[42,43], same_prompt=True,
                  timing='AR denoise + KV allocation/reset + scheduler setup + final GPU/rank sync; includes rerun; excludes model/T5/VAE',
                  validation=validation, validation_relative_l2_tolerance=0.02, stats=stats, modes={})
    for name, values in samples.items():
        mean = statistics.mean(values)
        requests = 1 if name == 'single' else 2
        result['modes'][name] = dict(samples_s=values, mean_s=mean, stdev_s=statistics.stdev(values) if len(values)>1 else 0,
                                   requests=requests, rps=requests/mean, fps=requests*frames/mean,
                                   peak_allocated_gib=max(peaks[name]))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if b.is_main_process():
        (out/f'p{args.seq_p_size}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
