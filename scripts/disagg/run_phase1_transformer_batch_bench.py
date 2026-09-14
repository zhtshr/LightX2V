#!/usr/bin/env python3
"""Phase 1 batch micro-benchmark: compare batch=1 vs fused batch>1 (t2v, P=1).

Fused mode concatenates B samples along the token dim for linear/FFN GEMMs, then
runs SLA self-attn per sample to avoid cross-sample attention.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from lightx2v.disagg.utils import load_wan_text_encoder, load_wan_transformer, set_config
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.set_config import set_parallel_config
from lightx2v.utils.utils import is_main_process, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Reuse cache helpers from phase1 bench
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_phase1_transformer_bench import (  # noqa: E402
    _bench_transformer_once,
    _init_distributed,
    _latent_shape_from_config,
    _load_config,
    _move_tensor_tree,
    _prepare_inputs_cache,
    _prepare_payload_on_device,
    _sync_device,
)


class _BatchedWanScheduler(WanScheduler):
    def __init__(self, config, batch_size: int):
        super().__init__(config)
        self.batch_size = batch_size

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        if self.batch_size <= 1:
            return super().prepare_latents(seed, latent_shape, dtype=dtype)
        if self.generator is None:
            self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        c, t, h, w = latent_shape
        self.latents = torch.randn(
            self.batch_size,
            c,
            t,
            h,
            w,
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )


class _BatchedWanPreInfer:
    """t2v batch path: batched patch-embed + text-embed, concat tokens."""

    def __init__(self, base_pre_infer):
        self._base = base_pre_infer
        self.scheduler = base_pre_infer.scheduler
        self.config = base_pre_infer.config
        self.batch_size = 1
        self.tokens_per_sample = 0

    def __getattr__(self, name):
        return getattr(self._base, name)

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        if self.batch_size <= 1:
            return self._base.infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)

        from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
        from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d

        latents = self.scheduler.latents  # [B, C, T, H, W]
        t = self.scheduler.timestep_input
        context_in = inputs["text_encoder_output"]["context"]  # [B, text_len, dim]
        B = self.batch_size
        text_len = context_in.shape[1]

        # Batched patch embedding (single conv3d over batch)
        x = weights.patch_embedding.apply(latents)
        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, L, D]
        self.tokens_per_sample = x.shape[1]
        x = x.reshape(B * self.tokens_per_sample, -1)

        embed = sinusoidal_embedding_1d(self._base.freq_dim, t.flatten())
        if self._base.sensitive_layer_dtype != self._base.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self._base.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self._base.dim))

        # Batched text embedding
        ctx_flat = context_in.reshape(B * text_len, -1)
        if self._base.sensitive_layer_dtype != self._base.infer_dtype:
            out = weights.text_embedding_0.apply(ctx_flat.to(self._base.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(ctx_flat)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)  # [B*text_len, dim]

        grid_sizes = (grid_sizes_t, grid_sizes_h, grid_sizes_w)
        grid_sizes_out = GridOutput(
            tensor=torch.tensor([[grid_sizes[0], grid_sizes[1], grid_sizes[2]]], dtype=torch.int32, device=x.device),
            tuple=grid_sizes,
        )
        if self._base.cos_sin is None or self._base.grid_sizes != grid_sizes_out.tuple:
            freqs = self._base.freqs.clone()
            self._base.grid_sizes = grid_sizes_out.tuple
            self._base.cos_sin = self._base.prepare_cos_sin(grid_sizes_out.tuple, freqs)

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes_out,
            x=x,
            embed0=embed0.squeeze(0),
            context=context,
            cos_sin=self._base.cos_sin,
            adapter_args={"motion_vec": None},
        )


class _BatchedWanTransformerInfer(WanTransformerInfer):
    def __init__(self, config, batch_size: int, tokens_per_sample: int):
        super().__init__(config)
        self.batch_size = batch_size
        self.tokens_per_sample = tokens_per_sample
        self._use_sla = config.get("self_attn_1_type") == "sla_attn"

    def _norm1_modulate(self, phase, x, shift_msa, scale_msa):
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
        else:
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)
        return norm1_out

    def _batched_qkv(self, phase, norm1_out):
        s, n, d = norm1_out.shape[0], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)
        return q, k, v

    def _apply_sla_batched(self, phase, q, k, v):
        from lightx2v.common.ops.attn.kernels.sla_kernel import _attention
        from lightx2v.common.ops.attn.utils.sla_util import get_block_map

        B, L = self.batch_size, self.tokens_per_sample
        n, d = q.shape[1], q.shape[2]
        sla = phase.self_attn_1
        q_b = q.view(B, L, n, d).transpose(1, 2).contiguous()
        k_b = k.view(B, L, n, d).transpose(1, 2).contiguous()
        v_b = v.view(B, L, n, d).transpose(1, 2).contiguous()
        sparse_map, lut, real_topk = get_block_map(
            q_b, k_b, topk_ratio=sla.topk, BLKQ=sla.BLKQ, BLKK=sla.BLKK
        )
        out_b = _attention.apply(q_b, k_b, v_b, sparse_map, lut, real_topk, sla.BLKQ, sla.BLKK)
        return out_b.transpose(1, 2).reshape(B * L, n * d)

    def _apply_dense_self_batched(self, phase, q, k, v):
        try:
            from lightx2v.common.ops.attn.sage_attn import sageattn
        except ImportError:
            sageattn = None

        B, L = self.batch_size, self.tokens_per_sample
        n, d = q.shape[1], q.shape[2]
        q_b = q.view(B, L, n, d)
        k_b = k.view(B, L, n, d)
        v_b = v.view(B, L, n, d)
        if sageattn is not None:
            out = sageattn(q_b, k_b, v_b, tensor_layout="NHD")
        else:
            import torch.nn.functional as F

            out = F.scaled_dot_product_attention(
                q_b.transpose(1, 2), k_b.transpose(1, 2), v_b.transpose(1, 2)
            ).transpose(1, 2)
        return out.reshape(B * L, n * d)

    def infer_self_attn(self, phase, x, shift_msa, scale_msa):
        if self.batch_size <= 1:
            return super().infer_self_attn(phase, x, shift_msa, scale_msa)

        norm1_out = self._norm1_modulate(phase, x, shift_msa, scale_msa)
        q, k, v = self._batched_qkv(phase, norm1_out)
        q, k = self.apply_rope_func(q, k, self.cos_sin)

        if self._use_sla:
            attn_out = self._apply_sla_batched(phase, q, k, v)
        else:
            attn_out = self._apply_dense_self_batched(phase, q, k, v)

        y = phase.self_attn_o.apply(attn_out)
        if self.clean_cuda_cache:
            torch_device_module = getattr(torch, AI_DEVICE)
            del q, k, v, attn_out, norm1_out
            torch_device_module.empty_cache()
        return y

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        if self.batch_size <= 1:
            return super().infer_cross_attn(phase, x, context, y_out, gate_msa)

        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        B = self.batch_size
        L = self.tokens_per_sample
        text_len = context.shape[0] // B
        n, d = self.num_heads, self.head_dim

        norm3_out = phase.norm3.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)

        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k_full = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, self.global_num_heads, d)
        v_full = phase.cross_attn_v.apply(context).view(-1, self.global_num_heads, d)
        if self.tp_size > 1:
            head_start = self.tp_rank * n
            k = k_full[:, head_start : head_start + n, :]
            v = v_full[:, head_start : head_start + n, :]
        else:
            k, v = k_full, v_full

        q_b = q.view(B, L, n, d)
        k_b = k.view(B, text_len, n, d)
        v_b = v.view(B, text_len, n, d)

        try:
            from lightx2v.common.ops.attn.sage_attn import sageattn

            attn_out = sageattn(q_b, k_b, v_b, tensor_layout="NHD").reshape(B * L, n * d)
        except ImportError:
            import torch.nn.functional as F

            attn_out = F.scaled_dot_product_attention(
                q_b.transpose(1, 2), k_b.transpose(1, 2), v_b.transpose(1, 2)
            ).transpose(1, 2).reshape(B * L, n * d)

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            torch_device_module = getattr(torch, AI_DEVICE)
            del q, k, v, norm3_out, context
            torch_device_module.empty_cache()
        return x, attn_out


class _BatchedWanPostInfer:
    def __init__(self, base_post_infer, batch_size: int, tokens_per_sample: int):
        self._base = base_post_infer
        self.batch_size = batch_size
        self.tokens_per_sample = tokens_per_sample

    def __getattr__(self, name):
        return getattr(self._base, name)

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        if self.batch_size <= 1:
            return self._base.infer(x, pre_infer_out)
        import math

        L = self.tokens_per_sample
        grid_sizes = pre_infer_out.grid_sizes.tuple
        outs = []
        for b in range(self.batch_size):
            x_b = x[b * L : (b + 1) * L]
            c = self._base.out_dim
            x_b = x_b[: math.prod(grid_sizes)].view(*grid_sizes, *self._base.patch_size, c)
            x_b = torch.einsum("fhwpqrc->cfphqwr", x_b)
            x_b = x_b.reshape(c, *[i * j for i, j in zip(grid_sizes, self._base.patch_size)])
            outs.append(x_b.float())
        return outs


def _stack_batch_payload(payload: dict[str, Any], batch_size: int) -> dict[str, Any]:
    if batch_size <= 1:
        return payload
    out = copy.deepcopy(payload)
    te = out["inputs"]["text_encoder_output"]
    ctx = te["context"]
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0)
    te["context"] = ctx.repeat(batch_size, 1, 1)
    if te.get("context_null") is not None:
        cn = te["context_null"]
        if cn.dim() == 2:
            cn = cn.unsqueeze(0)
        te["context_null"] = cn.repeat(batch_size, 1, 1)
    return out


def _patch_model_for_batch(model: Any, batch_size: int) -> None:
    if batch_size <= 1:
        return
    model.scheduler = _BatchedWanScheduler(model.config, batch_size)
    model.set_scheduler(model.scheduler)

    batched_pre = _BatchedWanPreInfer(model.pre_infer)
    batched_pre.batch_size = batch_size
    model.pre_infer = batched_pre

    orig_infer = model._infer_cond_uncond

    def _infer_batched(inputs, infer_condition=True):
        model.scheduler.infer_condition = infer_condition
        pre_infer_out = model.pre_infer.infer(model.pre_weight, inputs)
        model.transformer_infer.batch_size = batch_size
        model.transformer_infer.tokens_per_sample = model.pre_infer.tokens_per_sample
        x = model.transformer_infer.infer(model.transformer_weights, pre_infer_out)
        grid = pre_infer_out.grid_sizes.tuple
        post = _BatchedWanPostInfer(
            model.post_infer,
            batch_size,
            model.pre_infer.tokens_per_sample,
        )
        noise_preds = post.infer(x, pre_infer_out)
        return torch.stack(noise_preds, dim=0)

    model._infer_cond_uncond = lambda inputs, infer_condition=True: _infer_batched(inputs, infer_condition)
    model.transformer_infer = _BatchedWanTransformerInfer(
        model.config, batch_size, tokens_per_sample=0
    )
    model.transformer_infer.set_scheduler(model.scheduler)
    model.transformer_infer.cos_sin = model.pre_infer.cos_sin


def _run_serial_batch(
    scheduler: WanScheduler,
    model: Any,
    payload: dict[str, Any],
    batch_size: int,
) -> None:
    for _ in range(batch_size):
        single = copy.deepcopy(payload)
        _bench_transformer_once(scheduler, model, single)


def _bench_serial_batch(scheduler, model, payload, batch_size) -> float:
    _sync_device()
    start = time.perf_counter()
    _run_serial_batch(scheduler, model, payload, batch_size)
    return time.perf_counter() - start


def _write_sla_config(dense_cfg: Path, sla_cfg: Path) -> None:
    data = json.loads(dense_cfg.read_text(encoding="utf-8"))
    data["self_attn_1_type"] = "sla_attn"
    data["sla_attn_setting"] = {"sparsity_ratio": 0.8, "operator": "triton"}
    data.pop("general_sparse_attn_setting", None)
    sla_cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", default="/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--model_cls", default="wan2.1")
    parser.add_argument("--inputs_cache", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--batch_mode", choices=("fused", "serial"), default="fused")
    parser.add_argument("--seq_p_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=3)
    parser.add_argument("--prompt", default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_path", default="")
    args = parser.parse_args()

    if args.batch_size > 1 and args.seq_p_size > 1:
        raise SystemExit("batch benchmark currently supports seq_p_size=1 only")

    config = _load_config(argparse.Namespace(**{**vars(args), "tensor_p_size": 0, "encoder_only": False}))
    config["enable_cfg"] = False
    seed_all(args.seed)
    _init_distributed(config)

    payload = _prepare_inputs_cache(
        config=config,
        cache_path=Path(args.inputs_cache),
        prompt=args.prompt,
        image_path=args.image_path or "/root/zht/LightX2V/assets/inputs/imgs/img_0.jpg",
        seed=args.seed,
        force=False,
        task=args.task,
        negative_prompt=args.negative_prompt,
    )
    payload = _prepare_payload_on_device(payload)

    load_start = time.perf_counter()
    model = load_wan_transformer(config)
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)
    model_load_s = time.perf_counter() - load_start

    results: dict[str, Any] = {
        "metric": "transformer_batch_throughput",
        "config_json": args.config_json,
        "batch_size": args.batch_size,
        "batch_mode": args.batch_mode,
        "seq_p_size": args.seq_p_size,
        "model_load_s_excluded": model_load_s,
        "target_height": config.get("target_height"),
        "target_width": config.get("target_width"),
        "cases": {},
    }

    # batch=1 baseline
    for _ in range(args.warmup):
        _bench_transformer_once(scheduler, model, payload)
    b1_samples = [_bench_transformer_once(scheduler, model, payload) for _ in range(args.measure_iters)]
    b1_avg = sum(b1_samples) / len(b1_samples)
    results["cases"]["batch1"] = {
        "transformer_compute_s": b1_avg,
        "throughput_samples_per_s": 1.0 / b1_avg,
        "samples": b1_samples,
    }

    if args.batch_size > 1:
        if args.batch_mode == "serial":
            batch_payload = payload
            for _ in range(args.warmup):
                _bench_serial_batch(scheduler, model, batch_payload, args.batch_size)
            b_samples = [
                _bench_serial_batch(scheduler, model, batch_payload, args.batch_size)
                for _ in range(args.measure_iters)
            ]
        else:
            _patch_model_for_batch(model, args.batch_size)
            batch_payload = _stack_batch_payload(payload, args.batch_size)
            sched = model.scheduler
            for _ in range(args.warmup):
                _bench_transformer_once(sched, model, batch_payload)
            b_samples = [
                _bench_transformer_once(sched, model, batch_payload) for _ in range(args.measure_iters)
            ]
        b_avg = sum(b_samples) / len(b_samples)
        results["cases"][f"batch{args.batch_size}_{args.batch_mode}"] = {
            "transformer_compute_s": b_avg,
            "throughput_samples_per_s": args.batch_size / b_avg,
            "ms_per_sample": b_avg / args.batch_size * 1000.0,
            "samples": b_samples,
            "speedup_vs_serial_b1": (args.batch_size / b_avg) / (1.0 / b1_avg),
            "efficiency_vs_ideal": b_avg / (b1_avg * args.batch_size),
        }

    if is_main_process():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        print(f"wrote {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
