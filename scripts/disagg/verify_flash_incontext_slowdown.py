#!/usr/bin/env python3
"""Verify why Flash self-attn is slower in-model vs isolated microbench.

Hypotheses tested (chunk0 q=k=4680 unless noted):
  H1  KV-cache layout   — flash on k_cur vs k from cache buffer vs contiguous clone
  H2  L2 pollution      — flash after QKV GEMM / FFN GEMM vs cold flash
  H3  In-model tensors  — same q,k,v captured pre-flash: isolated vs after layer prefix
  H4  Launch gap        — 30L flash back-to-back vs 30L with simulated inter-layer gap
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from flash_attn import flash_attn_func

from lightx2v.common.ops import *  # noqa: F401, F403
from lightx2v.disagg.sf_support import DisaggSFKVCacheManager
from lightx2v.disagg.utils import load_wan_scheduler, load_wan_text_encoder, load_wan_transformer, set_config

PROMPT = "A stylish woman strolls down a bustling Tokyo street."
Q = 4680
H, D = 12, 128
HEADS_DIM = H * D


@dataclass
class BenchResult:
    name: str
    ms_per_call: float
    ms_30l: float
    tflops: float
    notes: str = ""


def attn_tflops(q_len: int, k_len: int, ms: float, layers: int = 1) -> float:
    flops = 4 * q_len * k_len * H * D * layers
    return flops / (ms * 1e-3) / 1e12


def bench_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    name: str,
    warmup: int = 15,
    iters: int = 50,
    layers: int = 1,
    gap_ms: float = 0.0,
    pre_hook=None,
    notes: str = "",
) -> BenchResult:
    """Benchmark flash_attn_func; q,k,v are [seq, heads, dim] or [1,seq,heads,dim]."""
    if q.dim() == 3:
        q4 = q.unsqueeze(0)
        k4 = k.unsqueeze(0)
        v4 = v.unsqueeze(0)
    else:
        q4, k4, v4 = q, k, v

    def one():
        if pre_hook is not None:
            pre_hook()
        if layers == 1:
            flash_attn_func(q4, k4, v4)
        else:
            for _ in range(layers):
                flash_attn_func(q4, k4, v4)
                if gap_ms > 0:
                    _simulate_gap(gap_ms)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000 / iters
    per_ms = total_ms / layers if layers > 1 else total_ms
    k_len = k4.shape[1]
    return BenchResult(
        name=name,
        ms_per_call=round(per_ms, 4),
        ms_30l=round(total_ms if layers > 1 else per_ms * 30, 4),
        tflops=round(attn_tflops(Q, k_len, per_ms), 2),
        notes=notes,
    )



def _simulate_gap(target_ms: float) -> None:
    """Burn ~target_ms on GPU with one FFN-shaped GEMM (~1.5ms on A10)."""
    x = torch.randn(4680, 1536, device="cuda", dtype=torch.bfloat16)
    w_ffn = torch.randn(1536, 8960, device="cuda", dtype=torch.bfloat16)
    for _ in range(max(1, int(target_ms / 1.5))):
        y = x @ w_ffn  # always [4680,1536] @ [1536,8960]
        del y


def _rand_qkv(device="cuda", dtype=torch.bfloat16):
    q = torch.randn(Q, H, D, device=device, dtype=dtype)
    k = torch.randn(Q, H, D, device=device, dtype=dtype)
    v = torch.randn(Q, H, D, device=device, dtype=dtype)
    return q, k, v


def test_h1_kv_layout(kv_cache, layer_id: int, q, k_cur, v_cur) -> list[BenchResult]:
    """H1: compare flash input layouts."""
    results = []
    results.append(bench_flash(q, k_cur, v_cur, name="H1a_k_cur_fresh", notes="q,k,v from model compute path"))

    attn_k = kv_cache.k_cache(layer_id, 0, k_cur.shape[0])
    attn_v = kv_cache.v_cache(layer_id, 0, v_cur.shape[0])
    results.append(
        bench_flash(
            q, attn_k, attn_v,
            name="H1b_k_cache_slice",
            notes=f"from cache buffer, k.stride={list(attn_k.stride())}, contiguous={attn_k.is_contiguous()}",
        )
    )
    results.append(
        bench_flash(
            q, attn_k.contiguous(), attn_v.contiguous(),
            name="H1c_k_cache_contiguous_clone",
            notes="force contiguous copy from cache",
        )
    )
    # Simulate store path: write then read
    kb = kv_cache._k_layer(layer_id)
    vb = kv_cache._v_layer(layer_id)
    kb[: Q].copy_(k_cur)
    vb[: Q].copy_(v_cur)
    k_view = kb[:Q]
    v_view = vb[:Q]
    results.append(
        bench_flash(
            q, k_view, v_view,
            name="H1d_kv_buffer_view",
            notes=f"direct buffer view stride={list(k_view.stride())}",
        )
    )
    return results


def test_h2_l2_pollution(q, k, v) -> list[BenchResult]:
    """H2: flash after GEMM warms L2."""
    results = []
    results.append(bench_flash(q, k, v, name="H2a_cold_flash"))

    qkv_w = torch.randn(D * H, 1536, device="cuda", dtype=torch.bfloat16)  # wrong shape - use realistic
    # Realistic QKV GEMM: [4680,1536] @ [1536,1536]
    w_qkv = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Q, 1536, device="cuda", dtype=torch.bfloat16)

    def after_qkv_gemm():
        y = x @ w_qkv
        y = y @ w_qkv  # 2x to ~2ms

    results.append(
        bench_flash(q, k, v, name="H2b_after_qkv_gemm", pre_hook=after_qkv_gemm, notes="2x [4680,1536] GEMM before flash")
    )

    w_ffn = torch.randn(1536, 8960, device="cuda", dtype=torch.bfloat16)

    def after_ffn_gemm():
        y = x @ w_ffn

    results.append(
        bench_flash(q, k, v, name="H2c_after_ffn_gemm", pre_hook=after_ffn_gemm, notes="FFN-shaped GEMM before flash")
    )

    def after_heavy_gemm():
        y = x
        for _ in range(4):
            y = y @ w_ffn
            y = y @ w_ffn.T[:, :1536]

    results.append(
        bench_flash(q, k, v, name="H2d_after_heavy_gemm", pre_hook=after_heavy_gemm, notes="4x heavy GEMM to pollute L2")
    )
    del qkv_w, w_qkv, w_ffn, x
    return results


def test_h4_launch_gap(q, k, v) -> list[BenchResult]:
    """H4: 30L flash with/without simulated inter-layer gap."""
    results = []
    results.append(bench_flash(q, k, v, name="H4a_30L_back2back", layers=30, iters=20))
    results.append(
        bench_flash(q, k, v, name="H4b_30L_gap_8ms", layers=30, iters=10, gap_ms=8.0, notes="~8ms gap/layer (cross+FFN)")
    )
    return results


def capture_model_tensors(cfg: dict[str, Any], layer_id: int = 15) -> dict[str, Any]:
    """Run chunk0 forward; hook layer `layer_id` right before flash."""
    captured: dict[str, Any] = {}

    text_encoder = load_wan_text_encoder(cfg)[0]
    ctx = text_encoder.infer([PROMPT])
    ctx = torch.stack([torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in ctx])
    del text_encoder
    torch.cuda.empty_cache()

    ls = [16, 21, 60, 104]
    inputs = {
        "text_encoder_output": {"context": ctx.cuda(), "context_null": None},
        "image_encoder_output": None,
    }
    model = load_wan_transformer(cfg)
    scheduler = load_wan_scheduler(cfg)
    model.set_scheduler(scheduler)
    ls_adj, num_out, _ = DisaggSFKVCacheManager.setup(model, cfg, ls)
    scheduler.num_output_frames = num_out
    scheduler.prepare(seed=42, latent_shape=list(ls_adj), image_encoder_output=None)
    model.kv_cache_manager.current_step = 0
    scheduler.step_pre(seg_index=0, step_index=0, is_rerun=False)

    phase = model.transformer_weights.blocks[layer_id].compute_phases[0]
    orig_apply = phase.self_attn_1.apply

    def hooked_apply(q, k, v, **kwargs):
        captured["q"] = q.detach()
        captured["k"] = k.detach()
        captured["v"] = v.detach()
        captured["k_stride"] = list(k.stride())
        captured["k_contiguous"] = k.is_contiguous()
        captured["layer_id"] = layer_id
        return orig_apply(q=q, k=k, v=v, **kwargs)

    phase.self_attn_1.apply = hooked_apply

    # Warmup
    model.infer(inputs)
    captured.clear()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.infer(inputs)
    torch.cuda.synchronize()
    captured["full_forward_ms"] = (time.perf_counter() - t0) * 1000

    kv_cache = model.kv_cache_manager.self_attn_kv_cache
    if "q" in captured:
        q, k, v = captured["q"], captured["k"], captured["v"]
        captured["h1"] = test_h1_kv_layout(kv_cache, layer_id, q, k, v)
        captured["h2_model_tensors"] = test_h2_l2_pollution(q, k, v)
        captured["h3_isolated"] = bench_flash(q, k, v, name="H3_model_tensors_isolated")

    DisaggSFKVCacheManager.teardown(model)
    return captured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="save_results/sf_qk_opt/flash_incontext_verify.json")
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    cfg = set_config(
        model_path="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B",
        task="t2v",
        model_cls="wan2.1_sf",
        config_path="configs/self_forcing/wan_t2v_sf_sp_bench.json",
    )
    cfg["parallel"] = False

    out: dict[str, Any] = {"q": Q, "k": Q, "heads": H, "head_dim": D}

    # --- Synthetic baselines (no model) ---
    q0, k0, v0 = _rand_qkv()
    out["synthetic"] = {
        "baseline_randn": bench_flash(q0, k0, v0, name="S0_randn_baseline").__dict__,
        "h2_l2": [r.__dict__ for r in test_h2_l2_pollution(q0, k0, v0)],
        "h4_launch_gap": [r.__dict__ for r in test_h4_launch_gap(q0, k0, v0)],
    }

    # --- Model-captured tensors ---
    print("Capturing model tensors (chunk0 forward)...")
    cap = capture_model_tensors(cfg, layer_id=args.layer)
    out["model_capture"] = {
        "layer_id": cap.get("layer_id"),
        "full_forward_ms": round(cap.get("full_forward_ms", 0), 2),
        "k_stride": cap.get("k_stride"),
        "k_contiguous": cap.get("k_contiguous"),
    }
    if "h3_isolated" in cap:
        out["model_capture"]["h3_isolated"] = cap["h3_isolated"].__dict__
        out["model_capture"]["h1_kv_layout"] = [r.__dict__ for r in cap["h1"]]
        out["model_capture"]["h2_on_model_tensors"] = [r.__dict__ for r in cap["h2_model_tensors"]]

    # --- Summary interpretation ---
    syn_base = out["synthetic"]["baseline_randn"]["ms_per_call"]
    lines = ["\n=== Flash in-context slowdown verification ===\n"]
    lines.append(f"Synthetic baseline (randn): {syn_base:.3f} ms/layer, {out['synthetic']['baseline_randn']['tflops']:.0f} TFLOPS")

    if "h3_isolated" in out["model_capture"]:
        m = out["model_capture"]["h3_isolated"]["ms_per_call"]
        lines.append(f"Model tensors isolated:     {m:.3f} ms/layer, {out['model_capture']['h3_isolated']['tflops']:.0f} TFLOPS")
        lines.append(f"  → layout/tensor diff vs randn: {(m/syn_base - 1)*100:+.1f}%")

    for r in out["synthetic"]["h2_l2"]:
        delta = (r["ms_per_call"] / syn_base - 1) * 100
        lines.append(f"  {r['name']}: {r['ms_per_call']:.3f} ms ({delta:+.1f}% vs baseline)")

    if "h1_kv_layout" in out["model_capture"]:
        lines.append("\nH1 KV layout (model tensors):")
        base = out["model_capture"]["h3_isolated"]["ms_per_call"]
        for r in out["model_capture"]["h1_kv_layout"]:
            delta = (r["ms_per_call"] / base - 1) * 100
            lines.append(f"  {r['name']}: {r['ms_per_call']:.3f} ms ({delta:+.1f}% vs k_cur) — {r['notes']}")

    for r in out["synthetic"]["h4_launch_gap"]:
        lines.append(f"  {r['name']}: {r['ms_30l']:.2f} ms total 30L")

    summary = "\n".join(lines)
    print(summary)
    out["summary_text"] = summary

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
