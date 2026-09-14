#!/usr/bin/env python3
"""After equal Flash (no skew) vs after skew — what lifts gather from 1.5ms toward E2E 5.5ms."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

try:
    import flash_attn
except ImportError:
    flash_attn = None

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


def main() -> int:
    if "3" in {x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()}:
        raise RuntimeError("GPU3 forbidden")
    if flash_attn is None:
        return 1
    dist.init_process_group("nccl")
    r = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(r)
    Ql, H, D, Kl = 1170, 12, 128, 4680
    Q = Ql * ws
    q = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    qf = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    # seed qf
    dist.all_gather_into_tensor(qf, q)
    k = torch.randn(Kl, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(Ql, 1536, device="cuda", dtype=torch.bfloat16)
    scale = D**-0.5

    def gather():
        dist.all_gather_into_tensor(qf, q)

    def a2a():
        all2all_seq2head(q, group=dist.group.WORLD)

    def flash():
        flash_attn.flash_attn_interface._flash_attn_forward(
            qf.unsqueeze(0).contiguous(),
            k.unsqueeze(0).contiguous(),
            v.unsqueeze(0).contiguous(),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

    def measure(prep, op, n=100, w=15):
        for _ in range(w):
            prep()
            op()
        torch.cuda.synchronize()
        dist.barrier()
        acc = 0.0
        for _ in range(n):
            prep()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            op()
            e.record()
            torch.cuda.synchronize()
            acc += float(s.elapsed_time(e))
        return acc / n

    def noop():
        return None

    def imb():
        if r != 0:
            for _ in range(4):
                x.copy_(x @ w)

    def flash_bal():
        flash()

    arms = {
        "gather_after_noop": measure(noop, gather),
        "a2a_after_noop": measure(noop, a2a),
        "gather_after_equal_flash": measure(flash_bal, gather),
        "a2a_after_equal_flash": measure(flash_bal, a2a),
        "gather_after_skew_gemm": measure(imb, gather),
        "a2a_after_skew_gemm": measure(imb, a2a),
    }
    keys = list(arms.keys())
    t = torch.tensor([arms[k] for k in keys], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if r == 0:
        m = {k: float(t[i]) for i, k in enumerate(keys)}
        out = {
            "ms_maxrank": m,
            "delta_gather_vs_noop": {
                "after_equal_flash": m["gather_after_equal_flash"] - m["gather_after_noop"],
                "after_skew_gemm": m["gather_after_skew_gemm"] - m["gather_after_noop"],
            },
            "delta_a2a_vs_noop": {
                "after_equal_flash": m["a2a_after_equal_flash"] - m["a2a_after_noop"],
                "after_skew_gemm": m["a2a_after_skew_gemm"] - m["a2a_after_noop"],
            },
            "ratios_gather_over_a2a": {
                "noop": m["gather_after_noop"] / max(m["a2a_after_noop"], 1e-9),
                "equal_flash": m["gather_after_equal_flash"] / max(m["a2a_after_equal_flash"], 1e-9),
                "skew": m["gather_after_skew_gemm"] / max(m["a2a_after_skew_gemm"], 1e-9),
            },
            "e2e_ref_gather": 5.55,
            "e2e_ref_a2a": 0.412,
        }
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_q_comm_context.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", path)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
