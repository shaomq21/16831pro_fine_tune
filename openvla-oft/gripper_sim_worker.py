#!/usr/bin/env python3
"""Long-lived LIBERO sim worker: joint_state -> gripper finger pixels (stdin/stdout JSONL)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_OPENVLA_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _OPENVLA_ROOT.parent
sys.path.insert(0, str(_OPENVLA_ROOT))
_libero_root = _REPO_ROOT / "LIBERO"
if _libero_root.exists():
    sys.path.insert(0, str(_libero_root))

from libero_sim_mask import gripper_finger_pixels  # noqa: E402


def _default_bddl() -> str:
    env_path = os.environ.get("GRIPPER_SIM_BDDL")
    if env_path and Path(env_path).exists():
        return env_path
    spatial = _REPO_ROOT / "LIBERO/libero/libero/bddl_files/libero_spatial"
    if spatial.exists():
        first = sorted(spatial.glob("*.bddl"))
        if first:
            return str(first[0])
    raise FileNotFoundError("No LIBERO spatial BDDL found for gripper sim worker")


def main():
    from libero.libero.envs import OffScreenRenderEnv

    bddl = _default_bddl()
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    env.reset()
    sim = env.env.sim
    print(json.dumps({"ready": True, "bddl": bddl}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "ping":
            print(json.dumps({"pong": True}), flush=True)
            continue
        joint = req.get("joint")
        gripper = req.get("gripper")
        if joint is None or gripper is None:
            print(json.dumps({"pts": [], "error": "missing joint/gripper"}), flush=True)
            continue
        h = int(req.get("h", 256))
        w = int(req.get("w", 256))
        js = list(joint)
        g = list(gripper)
        sim.data.qpos[:7] = js[:7]
        if sim.model.nq >= 9:
            sim.data.qpos[7:9] = g[:2]
        sim.forward()
        pts = gripper_finger_pixels(sim, height=h, width=w)
        print(json.dumps({"pts": pts}), flush=True)


if __name__ == "__main__":
    main()
