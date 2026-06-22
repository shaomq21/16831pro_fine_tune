"""Spatial disambiguation for LIBERO spatial / libero_90 pick-and-place tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Per-task calibrated click points (normalized xy, 256x256 agentview RLDS frames).
# Red = source bowl to pick; green = destination plate.
# Tuned on debug frame0 per task (see tools/debug_spatial_masks.py --overlay).
LIBERO_SPATIAL_TASK_POINTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": {
        "red": (0.26, 0.55),
        "green": (0.22, 0.74),
        "bowl_rank": "between_plate_ramekin",
    },
    "pick up the black bowl from table center and place it on the plate": {
        "red": (0.49, 0.57),
        "green": (0.22, 0.74),
        "bowl_rank": "center",
    },
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": {
        "red": (0.59, 0.64),
        "green": (0.22, 0.74),
        "bowl_rank": "in_drawer",
    },
    "pick up the black bowl next to the cookie box and place it on the plate": {
        "red": (0.58, 0.76),
        "green": (0.22, 0.74),
        "bowl_rank": "near_cookie_box",
    },
    "pick up the black bowl next to the plate and place it on the plate": {
        "red": (0.28, 0.50),
        "green": (0.22, 0.74),
        "bowl_rank": "near_plate",
    },
    "pick up the black bowl next to the ramekin and place it on the plate": {
        "red": (0.13, 0.48),
        "green": (0.22, 0.74),
        "bowl_rank": "leftmost_top",
    },
    "pick up the black bowl on the cookie box and place it on the plate": {
        "red": (0.46, 0.69),
        "green": (0.22, 0.74),
        "bowl_rank": "on_cookie_box",
    },
    "pick up the black bowl on the ramekin and place it on the plate": {
        "red": (0.26, 0.53),
        "green": (0.22, 0.74),
        "bowl_rank": "on_ramekin",
    },
    "pick up the black bowl on the stove and place it on the plate": {
        "red": (0.65, 0.40),
        "green": (0.22, 0.74),
        "bowl_rank": "on_stove",
    },
    "pick up the black bowl on the wooden cabinet and place it on the plate": {
        "red": (0.88, 0.32),
        "green": (0.22, 0.74),
        "bowl_rank": "on_cabinet",
    },
}

# Default plate destination for all libero_spatial pick tasks
LIBERO_SPATIAL_PLATE_POINT = (0.22, 0.74)


def get_libero_spatial_task_points(lang: str) -> Optional[Dict[str, Tuple[float, float]]]:
    s = lang.strip().lower().replace("_", " ")
    return LIBERO_SPATIAL_TASK_POINTS.get(s)


LIBERO_90_STUDY_SCENE4_TASKS = [
    "pick up the book in the middle and place it on the cabinet shelf",
    "pick up the book on the left and place it on top of the shelf",
    "pick up the book on the right and place it on the cabinet shelf",
    "pick up the book on the right and place it under the cabinet shelf",
]


@dataclass
class SpatialPickSpec:
    """Parsed pick-up task: red=source object, green=destination."""

    source_phrase: str
    dest_phrase: str
    spatial_rule: str
    dest_spatial_rule: str = "default"
    anchor_phrases: List[str] = field(default_factory=list)
    source_object: str = "object"  # bowl | book | object


def parse_pick_up_lang(lang: str) -> Optional[SpatialPickSpec]:
    """Parse pick-up-and-place instructions (libero_spatial, libero_90 STUDY_SCENE4, etc.)."""
    s = lang.strip().lower().replace("_", " ")

    patterns = [
        (r"pick up the (.+?) and place it on top of the (.+)$", "on_top"),
        (r"pick up the (.+?) and place it under the (.+)$", "under"),
        (r"pick up the (.+?) and place it on the (.+)$", "on"),
    ]
    source = dest = None
    dest_kind = "on"
    for pat, kind in patterns:
        m = re.match(pat, s)
        if m:
            source, dest = m.group(1).strip(), m.group(2).strip()
            dest_kind = kind
            break
    if source is None:
        return None

    rule, anchors, src_obj = _infer_source_rule(source)
    dest_rule = _infer_dest_rule(dest, dest_kind)
    return SpatialPickSpec(
        source_phrase=source,
        dest_phrase=dest,
        spatial_rule=rule,
        dest_spatial_rule=dest_rule,
        anchor_phrases=anchors,
        source_object=src_obj,
    )


def _infer_source_rule(source: str) -> Tuple[str, List[str], str]:
    s = source.lower()
    src_obj = "book" if "book" in s else ("bowl" if "bowl" in s else "object")

    # ---- libero_90 STUDY_SCENE4 books (check before generic left/right) ----
    if "book in the middle" in s:
        return "book_middle", ["cabinet shelf"], src_obj
    if "book on the left" in s:
        return "book_leftmost", ["cabinet shelf"], src_obj
    if "book on the right" in s:
        return "book_rightmost", ["cabinet shelf"], src_obj

    # ---- libero_spatial bowls ----
    if "between the plate and the ramekin" in s:
        return "between_plate_ramekin", ["plate", "ramekin"], src_obj
    if "from table center" in s:
        return "table_center", [], src_obj
    if "next to the cookie box" in s or "next to the cookies box" in s:
        return "near_cookie_box", ["cookie box"], src_obj
    if "next to the plate" in s:
        return "near_plate", ["plate"], src_obj
    if "next to the ramekin" in s:
        return "near_ramekin", ["ramekin"], src_obj
    if "on the cookie box" in s:
        return "on_cookie_box", ["cookie box"], src_obj
    if "in the top drawer" in s:
        return "in_top_drawer", ["wooden cabinet", "top drawer"], src_obj
    if "on the ramekin" in s:
        return "on_ramekin", ["ramekin"], src_obj
    if "on the stove" in s:
        return "on_stove", ["stove"], src_obj
    if "on the wooden cabinet" in s:
        return "on_cabinet", ["wooden cabinet"], src_obj
    if "left" in s:
        return "leftmost", [], src_obj
    if "right" in s:
        return "rightmost", [], src_obj
    return "default", [], src_obj


def _infer_dest_rule(dest: str, dest_kind: str) -> str:
    d = dest.lower()
    if dest_kind == "under" or "under" in d:
        return "dest_under_shelf"
    if dest_kind == "on_top" or "top of" in d:
        return "dest_on_top_shelf"
    if "cabinet shelf" in d or d == "shelf":
        return "dest_cabinet_shelf"
    if "plate" in d:
        return "dest_plate"
    return "default"


def _centers(boxes: List[np.ndarray]) -> np.ndarray:
    return np.array([[(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5] for b in boxes], dtype=np.float64)


def _areas(boxes: List[np.ndarray]) -> np.ndarray:
    return np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=np.float64)


def _filter_book_candidates(
    boxes: List[np.ndarray], scores: List[float], image_shape: Tuple[int, int],
) -> List[int]:
    """Keep book-like detections: small, on desk region (from test_seg_gdino_sam heuristics)."""
    H, W = image_shape
    img_area = float(H * W)
    centers = _centers(boxes)
    areas = _areas(boxes)
    keep = []
    for i, b in enumerate(boxes):
        cx, cy = centers[i]
        if areas[i] / img_area > 0.20:
            continue
        if cy < 0.40 * H or cy > 0.92 * H:
            continue
        keep.append(i)
    if not keep:
        return list(range(len(boxes)))
    return keep


def select_instance_index(
    boxes: List[np.ndarray],
    scores: List[float],
    rule: str,
    anchor_boxes: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
) -> int:
    """Pick one detection index among multiple source-object candidates."""
    if not boxes:
        return 0
    if len(boxes) == 1:
        return 0

    H, W = image_shape
    centers = _centers(boxes)
    cx, cy = centers[:, 0], centers[:, 1]

    def _anchor(name: str) -> Optional[np.ndarray]:
        for k, v in anchor_boxes.items():
            if name in k.lower():
                return v
        return None

    # Book disambiguation (STUDY_SCENE4)
    if rule.startswith("book_"):
        cand = _filter_book_candidates(boxes, scores, image_shape)
        cx_c = cx[cand]
        if rule == "book_leftmost":
            return cand[int(np.argmin(cx_c))]
        if rule == "book_rightmost":
            return cand[int(np.argmax(cx_c))]
        if rule == "book_middle":
            return cand[int(np.argmin(np.abs(cx_c - W * 0.5)))]

    if rule == "table_center":
        d = (cx - W * 0.5) ** 2 + (cy - H * 0.55) ** 2
        return int(np.argmin(d))

    if rule == "leftmost":
        return int(np.argmin(cx))

    if rule == "rightmost":
        return int(np.argmax(cx))

    if rule == "between_plate_ramekin":
        pb = _anchor("plate")
        rb = _anchor("ramekin")
        if pb is not None and rb is not None:
            pcx = (pb[0] + pb[2]) * 0.5
            rcx = (rb[0] + rb[2]) * 0.5
            lo, hi = min(pcx, rcx), max(pcx, rcx)
            mid = 0.5 * (lo + hi)
            valid = (cx >= lo - 0.05 * W) & (cx <= hi + 0.05 * W)
            if valid.any():
                idxs = np.where(valid)[0]
                return int(idxs[np.argmin(np.abs(cx[idxs] - mid))])
        return int(np.argmin(np.abs(cx - W * 0.5)))

    if rule in ("near_cookie_box", "on_cookie_box"):
        ab = _anchor("cookie")
        if ab is not None:
            acx, acy = (ab[0] + ab[2]) * 0.5, (ab[1] + ab[3]) * 0.5
            d = (cx - acx) ** 2 + (cy - acy) ** 2
            if rule == "on_cookie_box":
                d = d + np.maximum(0, cy - acy) * 2.0
            return int(np.argmin(d))

    if rule == "near_plate":
        pb = _anchor("plate")
        if pb is not None:
            pcx, pcy = (pb[0] + pb[2]) * 0.5, (pb[1] + pb[3]) * 0.5
            d = (cx - pcx) ** 2 + (cy - pcy) ** 2
            areas = _areas(boxes)
            plate_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            for i in range(len(boxes)):
                ix0, iy0, ix1, iy1 = boxes[i]
                overlap = max(0, min(ix1, pb[2]) - max(ix0, pb[0])) * max(0, min(iy1, pb[3]) - max(iy0, pb[1]))
                if overlap > 0.3 * min(areas[i], plate_area):
                    d[i] += 1e6
            return int(np.argmin(d))

    if rule == "near_ramekin":
        rb = _anchor("ramekin")
        if rb is not None:
            rcx, rcy = (rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5
            d = (cx - rcx) ** 2 + (cy - rcy) ** 2
            return int(np.argmin(d))

    if rule == "on_ramekin":
        rb = _anchor("ramekin")
        if rb is not None:
            rcx, rcy = (rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5
            d = (cx - rcx) ** 2 + (cy - rcy) ** 2 + np.maximum(0, cy - rcy) * 3.0
            return int(np.argmin(d))

    if rule == "on_stove":
        sb = _anchor("stove")
        if sb is not None:
            scx, scy = (sb[0] + sb[2]) * 0.5, (sb[1] + sb[3]) * 0.5
            d = (cx - scx) ** 2 + (cy - scy) ** 2
            return int(np.argmin(d))

    if rule in ("on_cabinet", "in_top_drawer"):
        cb = _anchor("cabinet") or _anchor("wooden")
        if cb is not None:
            ccx, ccy = (cb[0] + cb[2]) * 0.5, (cb[1] + cb[3]) * 0.5
            d = (cx - ccx) ** 2 + (cy - ccy) ** 2
            if rule == "in_top_drawer":
                d += np.abs(cy - (ccy + 0.05 * H)) * 2.0
            return int(np.argmin(d))
        if rule == "on_cabinet":
            right = np.where(cx > 0.55 * W)[0]
            if len(right):
                return int(right[np.argmax(cx[right])])

    return int(np.argmax(scores))


def select_tracked_bowl(
    masks: List[np.ndarray],
    boxes: List[np.ndarray],
    scores: List[float],
    image_shape: Tuple[int, int],
    *,
    click_xy: Optional[Tuple[float, float]] = None,
    green_xy: Optional[Tuple[float, float]] = None,
    prev_box: Optional[np.ndarray] = None,
    spatial_rule: Optional[str] = None,
    anchor_boxes: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """
    Pick the correct bowl among SAM3 detections for pick-and-place tracking.

    - Init: closest to calibrated click (cabinet-top bowl).
    - Carry/place: left-side bowl nearest the plate (green click).
    - Mid-flight: closest to previous box when still on the right half.
    """
    if not masks:
        return None

    H, W = image_shape
    centers = _centers(boxes)
    cx, cy = centers[:, 0], centers[:, 1]

    def _dist(i: int, px: float, py: float) -> float:
        return float((cx[i] - px) ** 2 + (cy[i] - py) ** 2)

    gcx = gcy = None
    if green_xy is not None:
        gcx, gcy = green_xy[0] * W, green_xy[1] * H

    pbc_x = ((prev_box[0] + prev_box[2]) * 0.5) if prev_box is not None else None

    if len(masks) == 1:
        i = 0
        # Single detection on the right while plate exists → distractor bowl on scale/cabinet
        if gcx is not None and cx[i] > 0.42 * W:
            return None
        return masks[0]

    # Bowl being carried / placed: left-side detection near plate wins over stale right-side track
    if gcx is not None:
        left = np.where(cx < 0.45 * W)[0]
        if len(left):
            near_plate = int(left[np.argmin([_dist(i, gcx, gcy) for i in left])])
            if pbc_x is None or pbc_x > 0.42 * W:
                return masks[near_plate]

    # Mid-carry: only stale right-side dets (scale/cabinet) — force caller fallback
    if (
        gcx is not None
        and pbc_x is not None
        and pbc_x > 0.45 * W
        and np.all(cx > 0.42 * W)
    ):
        return None

    if prev_box is not None:
        pbx = (prev_box[0] + prev_box[2]) * 0.5
        pby = (prev_box[1] + prev_box[3]) * 0.5
        idx = int(np.argmin([_dist(i, pbx, pby) for i in range(len(masks))]))
        if gcx is not None and _dist(idx, pbx, pby) ** 0.5 > 70:
            return None
        return masks[idx]

    if click_xy is not None:
        ccx, ccy = click_xy[0] * W, click_xy[1] * H
        return masks[int(np.argmin([_dist(i, ccx, ccy) for i in range(len(masks))]))]

    if spatial_rule and anchor_boxes is not None:
        idx = select_instance_index(list(boxes), list(scores), spatial_rule, anchor_boxes, (H, W))
        return masks[idx]

    return masks[int(np.argmax(scores))]


def select_dest_index(
    boxes: List[np.ndarray],
    scores: List[float],
    rule: str,
    image_shape: Tuple[int, int],
) -> int:
    """Pick destination (green) among multiple shelf/plate detections."""
    if not boxes:
        return 0
    if len(boxes) == 1:
        return 0

    H, W = image_shape
    cy = _centers(boxes)[:, 1]

    if rule == "dest_plate":
        return int(np.argmax([b[3] for b in boxes]))

    if rule == "dest_cabinet_shelf":
        # shelf surface: mid-height among detections
        return int(np.argmin(np.abs(cy - H * 0.42)))

    if rule == "dest_on_top_shelf":
        return int(np.argmin(cy))

    if rule == "dest_under_shelf":
        return int(np.argmax(cy))

    return int(np.argmax(scores))


def sam3_text_for_source(source_phrase: str) -> str:
    s = source_phrase.lower()
    if "book" in s:
        return "book"
    if "black bowl" in s or "bowl" in s:
        return "black bowl"
    return source_phrase


def sam3_text_for_dest(dest_phrase: str, dest_rule: str = "default") -> str:
    d = dest_phrase.lower()
    if "cabinet shelf" in d or dest_rule.startswith("dest_") and "shelf" in d:
        return "cabinet shelf"
    if d == "shelf" or "top of the shelf" in d:
        return "cabinet shelf"
    if "plate" in d:
        return "plate"
    return dest_phrase


def sam3_text_for_anchor(anchor: str) -> str:
    a = anchor.lower()
    if "cookie" in a:
        return "cookie box"
    if "stove" in a:
        return "white rectangular box on the left"
    if "shelf" in a or "cabinet" in a:
        return "cabinet shelf"
    return anchor


def primitive_skill_for_lang(lang: str) -> str:
    s = lang.strip().lower()
    if s.startswith("pick up "):
        return "pick_and_place"
    if s.startswith("put "):
        return "put"
    if s.startswith("push "):
        return "push"
    if s.startswith("open "):
        return "open"
    if s.startswith("turn on "):
        return "turn_on"
    return "unknown"


def mask_role_description(lang: str) -> Tuple[str, str]:
    """Return (red_description, green_description) for debug task.txt."""
    pts = get_libero_spatial_task_points(lang)
    if pts:
        return f"source bowl @ {pts['red']}", f"plate @ {pts['green']}"
    pick = parse_pick_up_lang(lang)
    if pick:
        red = f"source {pick.source_object} ({pick.spatial_rule})"
        green = f"destination ({pick.dest_spatial_rule}): {pick.dest_phrase}"
        return red, green
    s = lang.lower()
    if s.startswith("put "):
        return "grasped object", "destination"
    if s.startswith("push "):
        return "plate/source", "stove/front region"
    if s.startswith("open "):
        return "", "drawer handle"
    if s.startswith("turn on "):
        return "", "stove"
    return "source", "destination"
