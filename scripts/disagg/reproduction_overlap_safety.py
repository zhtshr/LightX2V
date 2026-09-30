"""Explicit CUDA stream dependencies for reproducing legacy overlap schedules.

Opt-in benchmark adapter; leaves the historical scheduler implementations intact.
"""

import torch


def install():
    from scripts.disagg import run_phase3_dual_overlap_bench as overlap
    from scripts.disagg import pp_sp_quad_overlap as quad

    original = overlap.A2AOrchestrator

    class OrderedOrchestrator(original):
        def _maybe_overlap(self, comm_launch, *, kind):
            producer = torch.cuda.current_stream(self.comm_stream.device)
            self.comm_stream.wait_stream(producer)
            self.compute_stream.wait_stream(producer)
            ti = self.ti_ref
            state = {}
            if ti is not None:
                for name in ("scheduler", "block_idx", "cos_sin"):
                    if hasattr(ti, name):
                        state[name] = getattr(ti, name)
            try:
                work = super()._maybe_overlap(comm_launch, kind=kind)
            finally:
                if ti is not None:
                    for name, value in state.items():
                        setattr(ti, name, value)
            # Collective outputs and the peer's next-layer inputs must be ready
            # before either tenant consumes them on the caller's stream.
            producer.wait_stream(self.comm_stream)
            producer.wait_stream(self.compute_stream)
            return work

    overlap.A2AOrchestrator = OrderedOrchestrator
    quad.A2AOrchestrator = OrderedOrchestrator
