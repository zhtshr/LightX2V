"""Isolate dense attention head-count scaling from SP transport and layout costs."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--sequence", type=int, default=32760)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--kv-sequence", type=int, default=None, help="KV length; defaults to query length")
    a = p.parse_args()
    kv_sequence = a.kv_sequence if a.kv_sequence is not None else a.sequence
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    props = torch.cuda.get_device_properties(0)
    heads = [40, 20, 10, 5]
    # Wan TorchSDPA takes sequence-major BF16 q/k/v, then transposes as views.
    full = [torch.randn(length, 40, 128, device="cuda", dtype=torch.bfloat16) for length in (a.sequence, kv_sequence, kv_sequence)]
    inputs = {h: [x[:, :h].contiguous().unsqueeze(0).transpose(1, 2) for x in full] for h in heads}

    def run(h):
        return F.scaled_dot_product_attention(*inputs[h], dropout_p=0.0, is_causal=False)

    samples = {h: [] for h in heads}
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref = run(40)
        validation = {}
        for h in heads:
            result = run(h)
            validation[h] = {"exact": torch.equal(ref[:, :h], result), "max_abs": float((ref[:, :h] - result).abs().max())}
            for _ in range(10):
                run(h)
        torch.cuda.synchronize()
        for rnd in range(a.rounds):
            for h in heads if rnd % 2 == 0 else heads[::-1]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(a.repeats):
                    result = run(h)
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end) / a.repeats
                samples[h].append(ms)
                print("SAMPLE", rnd, h, ms, flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for h in heads:
                with torch.profiler.record_function(f"heads_{h}"):
                    run(h)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out.with_suffix(".trace.json")))
    means = {h: sum(v) / len(v) for h, v in samples.items()}
    results = []
    for h in heads:
        results.append(
            {
                "heads": h,
                "sp_equivalent": 40 // h,
                "mean_ms": means[h],
                "samples_ms": samples[h],
                "relative_compute_efficiency": means[40] * h / (40 * means[h]),
                "qk_av_tflops": 4 * a.sequence * kv_sequence * h * 128 / (means[h] / 1000) / 1e12,
            }
        )
    out.write_text(
        json.dumps(
            {
                "args": vars(a),
                "gpu": props.name,
                "sm_count": props.multi_processor_count,
                "torch": torch.__version__,
                "dtype": "bfloat16",
                "head_dim": 128,
                "backend": "Torch SDPA FLASH_ATTENTION",
                "validation": validation,
                "results": results,
            },
            indent=2,
        )
    )
    assert all(v["exact"] for v in validation.values()), validation
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
