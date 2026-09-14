#!/usr/bin/env python3
"""Ensure legacy ``pyverbs.enums`` shim exposes SEND/MTU/WC symbols (pyverbs 59+)."""
from __future__ import annotations

from pathlib import Path


SHIM = '''"""Shim for legacy ``import pyverbs.enums`` (module removed in pyverbs 59+)."""
from pyverbs.libibverbs_enums import (
    ibv_access_flags,
    ibv_mtu,
    ibv_qp_type,
    ibv_send_flags,
    ibv_wc_status,
    ibv_wr_opcode,
)

IBV_QPT_RC = ibv_qp_type.IBV_QPT_RC

IBV_WR_RDMA_WRITE = ibv_wr_opcode.IBV_WR_RDMA_WRITE
IBV_WR_RDMA_READ = ibv_wr_opcode.IBV_WR_RDMA_READ
IBV_WR_ATOMIC_FETCH_AND_ADD = ibv_wr_opcode.IBV_WR_ATOMIC_FETCH_AND_ADD
IBV_WR_ATOMIC_CMP_AND_SWP = ibv_wr_opcode.IBV_WR_ATOMIC_CMP_AND_SWP

IBV_ACCESS_LOCAL_WRITE = ibv_access_flags.IBV_ACCESS_LOCAL_WRITE
IBV_ACCESS_REMOTE_WRITE = ibv_access_flags.IBV_ACCESS_REMOTE_WRITE
IBV_ACCESS_REMOTE_READ = ibv_access_flags.IBV_ACCESS_REMOTE_READ
IBV_ACCESS_REMOTE_ATOMIC = ibv_access_flags.IBV_ACCESS_REMOTE_ATOMIC

IBV_SEND_FENCE = ibv_send_flags.IBV_SEND_FENCE
IBV_SEND_SIGNALED = ibv_send_flags.IBV_SEND_SIGNALED
IBV_SEND_SOLICITED = ibv_send_flags.IBV_SEND_SOLICITED
IBV_SEND_INLINE = ibv_send_flags.IBV_SEND_INLINE
IBV_SEND_IP_CSUM = ibv_send_flags.IBV_SEND_IP_CSUM

IBV_MTU_256 = ibv_mtu.IBV_MTU_256
IBV_MTU_512 = ibv_mtu.IBV_MTU_512
IBV_MTU_1024 = ibv_mtu.IBV_MTU_1024
IBV_MTU_2048 = ibv_mtu.IBV_MTU_2048
IBV_MTU_4096 = ibv_mtu.IBV_MTU_4096

IBV_WC_SUCCESS = ibv_wc_status.IBV_WC_SUCCESS
'''


def main() -> int:
    try:
        import pyverbs
        import pyverbs.enums as e
    except Exception as exc:
        print(f"skip pyverbs enums patch: import failed: {exc}")
        return 0

    need = ("IBV_SEND_SIGNALED", "IBV_MTU_1024", "IBV_WC_SUCCESS")
    if all(hasattr(e, name) for name in need):
        print("pyverbs.enums already complete")
        return 0

    enums_path = Path(pyverbs.__file__).resolve().parent / "enums.py"
    if not enums_path.exists():
        print(f"skip pyverbs enums patch: missing {enums_path}")
        return 0

    bak = enums_path.with_suffix(enums_path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(enums_path.read_text(encoding="utf-8"), encoding="utf-8")
    enums_path.write_text(SHIM, encoding="utf-8")
    print(f"patched {enums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
