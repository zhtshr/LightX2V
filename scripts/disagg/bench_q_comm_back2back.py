#!/usr/bin/env python3
"""Back-to-back ops only — factor bytes vs collective kind."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


def main() -> int:
    if "3" in {x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()}:
        raise RuntimeError("GPU3 forbidden")
    dist.init_process_group("nccl")
    r = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(r)
    Ql, H, D = 1170, 12, 128
    Hp, Q = H // ws, Ql * ws
    q = torch.randn(Ql, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    qf = torch.empty(Q, H, D, device="cuda", dtype=torch.bfloat16)
    qt = torch.randn(Ql, Hp, D, device="cuda", dtype=torch.bfloat16).contiguous()
    qtf = torch.empty(Q, Hp, D, device="cuda", dtype=torch.bfloat16)

    def tmean(fn, n=150, w=30):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        acc = 0.0
        for _ in range(n):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            acc += float(s.elapsed_time(e))
        return acc / n

    def g12():
        dist.all_gather_into_tensor(qf, q)

    def g3():
        dist.all_gather_into_tensor(qtf, qt)

    def a2a():
        all2all_seq2head(q, group=dist.group.WORLD)

    m12, m3, ma = tmean(g12), tmean(g3), tmean(a2a)
    t = torch.tensor([m12, m3, ma], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if r == 0:
        g12m, g3m, a2am = (float(t[0]), float(t[1]), float(t[2]))
        out = {
            "ms_back2back_maxrank": {
                "allgather_H12_14MB": g12m,
                "allgather_H3_3.6MB": g3m,
                "ulysses_all2all_3.6MB": a2am,
            },
            "decomp": {
                "gap_H12_minus_a2a": g12m - a2am,
                "bytes_effect_H12_minus_H3": g12m - g3m,
                "kind_effect_H3gather_minus_a2a": g3m - a2am,
                "pct_bytes": 100 * (g12m - g3m) / max(g12m - a2am, 1e-9),
                "pct_kind": 100 * (g3m - a2am) / max(g12m - a2am, 1e-9),
                "ratio_H12_a2a": g12m / max(a2am, 1e-9),
            },
            "e2e_ref": {"stripe_gather": 5.55, "ulysses_a2a": 0.412, "ratio": 13.46},
        }
        print(json.dumps(out, indent=2))
        path = "save_results/optimization_study/sf_q_comm_back2back.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", path)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
