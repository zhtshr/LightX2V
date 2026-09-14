#!/usr/bin/env python3
"""Rebuild 1:5:2 and autoscale controllers from the default 8-GPU multi_node config."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path("/root/zht/LightX2V/configs/disagg/multi_node")
BASE = ROOT / "wan22_i2v_distill_controller.json"


def slot(itype: str, host: str, rank: int, cuda: int, autostart: bool = True) -> dict:
    return {
        "instance_type": itype,
        "host": host,
        "engine_rank": rank,
        "cuda_device": cuda,
        "autostart": autostart,
        "env": {
            "MOONCAKE_DEVICE_NAME": "eth0",
            "MOONCAKE_LOCAL_HOSTNAME": host,
        },
    }


def main() -> None:
    base = json.loads(BASE.read_text())
    h166 = "192.168.0.166"
    h139 = "192.168.0.139"

    # 1:5:2 — same physical map as default 1:6:1, but last transformer becomes 2nd decoder on 139:2
    cfg152 = deepcopy(base)
    cfg152["disagg_config"]["static_instance_slots"] = [
        slot("encoder", h139, 0, 0),
        slot("transformer", h166, 1, 0),
        slot("transformer", h166, 2, 1),
        slot("transformer", h166, 3, 2),
        slot("transformer", h166, 4, 3),
        slot("transformer", h166, 5, 4),
        slot("decoder", h139, 6, 1),
        slot("decoder", h139, 7, 2),
    ]
    cfg152["disagg_config"]["decoder_engine_rank"] = 6
    (ROOT / "wan22_i2v_distill_controller_152.json").write_text(json.dumps(cfg152, indent=4) + "\n")

    # Autoscale pool on the same 8 local+remote cards: start 1:4:1, spare 2T+1D
    cfg_as = deepcopy(base)
    cfg_as["disagg_config"]["ranks"] = 9
    cfg_as["disagg_config"]["static_instance_slots"] = [
        slot("encoder", h139, 0, 0, True),
        slot("transformer", h166, 1, 0, True),
        slot("transformer", h166, 2, 1, True),
        slot("transformer", h166, 3, 2, True),
        slot("transformer", h166, 4, 3, True),
        slot("transformer", h166, 5, 4, False),
        slot("transformer", h166, 6, 5, False),
        slot("decoder", h139, 7, 1, True),
        slot("decoder", h139, 8, 2, False),
    ]
    cfg_as["disagg_config"]["decoder_engine_rank"] = 7
    (ROOT / "wan22_i2v_distill_controller_autoscale.json").write_text(json.dumps(cfg_as, indent=4) + "\n")

    for name in (
        "wan22_i2v_distill_controller.json",
        "wan22_i2v_distill_controller_152.json",
        "wan22_i2v_distill_controller_autoscale.json",
    ):
        d = json.loads((ROOT / name).read_text())
        slots = d["disagg_config"]["static_instance_slots"]
        print(
            name,
            [(s["instance_type"], s["cuda_device"], s.get("autostart", True)) for s in slots],
        )


if __name__ == "__main__":
    main()
