#!/usr/bin/env python3
"""Summarize measured SF overlap throughput; SP=1 has no communication overlap."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
data = {p: json.loads((out/f'p{p}.json').read_text()) for p in (1,2,4,8)}
t1 = data[1]['modes']['single']['mean_s']
rows = []
for p,d in data.items():
    m=d['modes'];selected=m.get('overlap',m['back_to_back'])
    rows.append(dict(sp=p,single_s=m['single']['mean_s'],b2b_s=m['back_to_back']['mean_s'],
                     decomposed_serial_s=m.get('decomposed_serial',{}).get('mean_s'),
                     overlap_s=m.get('overlap',{}).get('mean_s'),
                     selected_mode='overlap' if p>1 else 'back_to_back',
                     rps=selected['rps'],fps=selected['fps'],
                     gain_vs_b2b=m['back_to_back']['mean_s']/selected['mean_s'],
                     gain_vs_decomposed=m.get('decomposed_serial',selected)['mean_s']/selected['mean_s'],
                     efficiency=selected['rps']/(p/t1),peak_gib=selected['peak_allocated_gib']))
lines=['# Krea SF 14B: SP dual-request overlap (L20X NVLink)', '',
       '832×480, 81 frames/request, BF16, Torch SDPA, Ulysses SP, no DiT/KV offload or quantization.',
       'Two requests use the same prompt and separate latent trajectories; validation uses seeds 42/43. Timed runs use the original process-global RNG. Each mode has 1 warmup and 3 measurements; mode order alternates.',
       'Timing includes all 7 chunks × (4 denoise steps + KV rerun), cache allocation/reset, scheduler setup and final GPU/rank synchronization. Excludes model loading, T5, VAE and validation.',
       'Throughput = 2 / pair wall; aggregate FPS = 162 / pair wall. Pair wall is not half the per-request latency.',
       'SP=1 reports sequential two-request baseline; A2A overlap is not applicable.', '',
       '| SP | single s | 2-req b2b s | decomposed serial s | overlap s | req/s | aggregate FPS | vs b2b | vs decomposed | efficiency | peak GiB/rank |',
       '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    decomp='—' if r['decomposed_serial_s'] is None else f"{r['decomposed_serial_s']:.4f}"
    overlap='—' if r['overlap_s'] is None else f"{r['overlap_s']:.4f}"
    lines.append(f"| {r['sp']} | {r['single_s']:.4f} | {r['b2b_s']:.4f} | {decomp} | {overlap} | {r['rps']:.4f} | {r['fps']:.2f} | {r['gain_vs_b2b']:.3f}× | {r['gain_vs_decomposed']:.3f}× | {r['efficiency']:.1%} | {r['peak_gib']:.2f} |")
lines += ['', f'Efficiency denominator: P / T1, T1 = {t1:.4f} s measured in this sweep. This is an ideal independent-replica reference, not measured data-parallel throughput.', '', '## Correctness', '']
for p in (2,4,8):
    for name,v in data[p]['validation'].items():
        max_abs=max(x['max_abs'] for x in v['requests'])
        rel=max(x['relative_l2'] for x in v['requests'])
        lines.append(f"- SP={p} {name}: passed={v['passed']}; rank-0 max absolute error={max_abs:.6g}, relative L2={rel:.6g}; pass checked across all ranks.")
lines += ['', 'Validation uses per-request/per-step deterministic RNG only outside the timed runs, because the legacy scheduler consumes a process-global generator.', '', '## Reproduce', '', '```bash', 'bash scripts/self_forcing/benchmark_sf_14b_dual_scaling.sh', '```', '', 'Raw data: p1.json, p2.json, p4.json, p8.json and corresponding logs in this directory.']
(out/'summary.json').write_text(json.dumps({'baseline_t1_s':t1,'results':rows},indent=2)+'\n')
(out/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
