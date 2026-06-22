"""Perception backend selection for LIBERO masking."""

from __future__ import annotations

import os

# ``sim`` (default): instance segmentation + gripper finger tips from LIBERO sim.
# ``real_perception``: Grounded-DINO / SAM3 masks + Roboflow gripper detection.
PERCEPTION_MODE = os.environ.get("PERCEPTION_MODE", "sim").strip().lower()


def is_sim_mode() -> bool:
    return PERCEPTION_MODE != "real_perception"


def is_real_perception_mode() -> bool:
    return PERCEPTION_MODE == "real_perception"
