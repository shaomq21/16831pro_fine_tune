# masked_grounded_sam.py
# ------------------------------------------------------------
# Grounded-SAM masking for OpenVLA RLDS images
# - background black
# - source objects painted RED, target objects painted GREEN
# - gripper centers from Roboflow model as white dots (optional)
#
# Gripper detection requires: pip install inference
# Set ROBOFLOW_API_KEY for hosted model: https://app.roboflow.com/margarets-workspace/gripper_box/models
# ------------------------------------------------------------

from __future__ import annotations

import os

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
#from debian.debtags import output

# ========== GroundingDINO ==========
# You need to have GroundingDINO repo installed/importable.
# Common import pattern:
#   git clone https://github.com/IDEA-Research/GroundingDINO.git
#   pip install -e GroundingDINO
from groundingdino.util.inference import Model as GroundingDINOModel
#from orca.orca_state import device
#from reportlab.rl_settings import imageReaderFlags

# ========== SAM ==========
# You need SAM installed:
#   git clone https://github.com/facebookresearch/segment-anything.git
#   pip install -e segment-anything
from segment_anything import sam_model_registry, SamPredictor

from mask_spatial import (
    SpatialPickSpec,
    get_libero_spatial_task_points,
    parse_pick_up_lang,
    sam3_text_for_anchor,
    sam3_text_for_dest,
    sam3_text_for_source,
    select_dest_index,
    select_instance_index,
    select_tracked_bowl,
)
from sam3_backend import DEFAULT_SAM3_CKPT, SAM3Segmenter, SAM3VideoTracker, _mask_to_box


def _groundingdino_has_cuda_ops() -> bool:
    try:
        from groundingdino import _C  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------
# 1) Language parsing rules
# --------------------------

@dataclass
class MaskSpec:
    red_phrases: List[str]
    green_phrases: List[str]
    red_points_xy: List[Tuple[float, float]] = None
    green_points_xy: List[Tuple[float, float]] = None
    spatial_pick: Optional[SpatialPickSpec] = None




def build_mask_spec_from_lang(lang: str) -> MaskSpec:
    """
    Your rules (as you described):
      - open ... : green mask drawers (middle/top drawer)
      - push ... : red mask plate + table region in front of stove (approx via 'plate' + 'table'/'stove front')
      - put ...  : red mask the grasped object (bowl/cream cheese/wine bottle),
                  green mask destination (stove/cabinet/rack/plate/bowl) depending on phrase
      - turn on  : mask stove (green)
    """
    s = lang.strip().lower()

    # OPEN
    if lang.startswith("open "):
        
        green_points = []

        
        TOP_HANDLE    = (0.71, 0.58)
       
        MIDDLE_HANDLE = (0.71, 0.63)

        if "top drawer" in lang:
            green_points = [TOP_HANDLE]
        elif "middle drawer" in lang:
            green_points = [MIDDLE_HANDLE]
        elif "drawer" in lang:
            green_points = [MIDDLE_HANDLE]

        return MaskSpec(
            red_phrases=[],
            green_phrases=[],          
            red_points_xy=[],
            green_points_xy=green_points,
        )

    # PUSH
    if lang.startswith("push "):
        # Your request: red mask plate + the table area before "front of the stove"
        # In practice: Grounded-SAM can't segment "table region in front of stove" perfectly via text,
        # so we approximate with "plate" (strong) and "table" or "stove" (weak).
        reds = []
        greens = []

        if "plate" in lang:
            reds.append("plate")
        # Stove: try both "white rectangular" (light) and "stove" (LIBERO stove is dark brown/black)
        greens += ["white rectangular box on the left", "stove"]
        return MaskSpec(red_phrases=reds, green_phrases=greens)

    # PUT
    if lang.startswith("put "):
        # parse: "put the X on/in/inside/on top of the Y"
        # red := X
        # green := Y (destination)
        # handle:
        #   "put the cream cheese in the bowl"
        #   "put the wine bottle on the rack"
        #   "put the bowl on the stove" etc.
        # We'll do a simple regex.
        # Note: dataset language is "put the bowl on the plate" style.

        red_obj = None
        green_obj = None

        m = re.match(r"put the (.+?) (on top of|on|in|inside) the (.+)$", lang)
        if m:
            red_obj = m.group(1).strip()
            green_obj = m.group(3).strip()
        else:
            # fallback: try "put the X" only
            m2 = re.match(r"put the (.+)$", lang)
            if m2:
                red_obj = m2.group(1).strip()
                
        reds = [red_obj] if red_obj else []
        greens = [green_obj] if green_obj else []

        red_points = []
        if red_obj and "cream cheese" in red_obj:
            reds = [] 
            red_points = [(0.33, 0.6)] 


        return MaskSpec(
        red_phrases=reds,
        green_phrases=greens,
        red_points_xy=red_points,
        green_points_xy=[],
    )

    # TURN ON  (stove -> 左边的扁方块)
    if lang.startswith("turn on "):
        return MaskSpec(red_phrases=[], green_phrases=["white rectangular box on the left"])

    # PICK UP (libero_spatial / libero_90): per-task point overrides when calibrated
    if lang.startswith("pick up "):
        task_pts = get_libero_spatial_task_points(lang)
        pick = parse_pick_up_lang(lang)
        if task_pts:
            return MaskSpec(
                red_phrases=[],
                green_phrases=[],
                red_points_xy=[task_pts["red"]],
                green_points_xy=[task_pts["green"]],
                spatial_pick=pick,
            )
        if pick:
            return MaskSpec(
                red_phrases=[sam3_text_for_source(pick.source_phrase)],
                green_phrases=[sam3_text_for_dest(pick.dest_phrase, pick.dest_spatial_rule)],
                spatial_pick=pick,
            )

    # default: no masks
    return MaskSpec(red_phrases=[], green_phrases=[])


# --------------------------
# 2) Grounded-SAM wrapper
# --------------------------

@dataclass
class GroundedSAMConfig:
    # GroundingDINO
    dino_config_path: str
    dino_checkpoint_path: str
    box_threshold: float = 0.30
    text_threshold: float = 0.25

    # SAM
    sam_type: str = "vit_h"  # vit_h / vit_l / vit_b
    sam_checkpoint_path: str = ""
    sam_backend: str = "sam1"  # sam1 (Grounded-DINO+SAM, libero_goal) | sam3 (spatial tracking)
    sam3_checkpoint_path: str = DEFAULT_SAM3_CKPT
    temporal_frames: int = 8  # prior frames for SAM3 video tracking on occlusion
    fast_mode: bool = False  # skip video clip + reduce SAM3 retries (RLDS batch)

    # Gripper detection via Roboflow (real_perception mode only)
    gripper_model_id: Optional[str] = "gripper_box/1"
    gripper_enabled: bool = True
    perception_mode: str = "sim"  # sim | real_perception

    device: str = "cuda"


from PIL import Image, ImageDraw

from perception_config import is_real_perception_mode

# Lazy-load Roboflow inference for gripper detection
_gripper_model = None

def _get_gripper_model(model_id: str):
    """Lazy load Roboflow gripper model. Requires: pip install inference"""
    global _gripper_model
    if _gripper_model is None:
        try:
            from inference import get_model
            _gripper_model = get_model(model_id=model_id)
        except ImportError:
            raise ImportError(
                "Gripper detection requires 'inference' package. Install with: pip install inference"
            )
    return _gripper_model


def _detect_gripper_centers(image_rgb: np.ndarray, model_id: str) -> List[Tuple[int, int]]:
    """
    Run Roboflow gripper model; returns center points (x,y) of bounding boxes in pixel coords.
    Typically returns 2 boxes (sometimes 1). Uses ObjectDetectionPrediction.x, .y (center coords).
    """
    model = _get_gripper_model(model_id)
    result = model.infer(image_rgb)
    # infer() can return list for batch; take first element
    if isinstance(result, (list, tuple)) and len(result) > 0:
        result = result[0]
    centers = []
    if hasattr(result, "predictions") and result.predictions:
        for p in result.predictions:
            if hasattr(p, "x") and hasattr(p, "y"):
                cx = int(round(p.x))
                cy = int(round(p.y))
                centers.append((cx, cy))
            elif hasattr(p, "bbox"):
                b = p.bbox  # x_min, y_min, x_max, y_max
                cx = int(round((b[0] + b[2]) / 2))
                cy = int(round((b[1] + b[3]) / 2))
                centers.append((cx, cy))
    return centers


def _draw_white_dots(img_arr: np.ndarray, centers_xy: List[Tuple[int, int]], radius: int = 12) -> np.ndarray:
    """Draw filled white circles at center points on image. Returns modified array."""
    if not centers_xy:
        return img_arr
    img_pil = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img_pil)
    for (cx, cy) in centers_xy:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255), outline=(255, 255, 255))
    return np.array(img_pil)


def _draw_points_overlay(img_pil: Image.Image, points_xy, *, color=(255, 0, 0), r=8, w=3):
    
    if not points_xy:
        return img_pil

    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    for (px, py) in points_xy:
        x = int(round(px * W))
        y = int(round(py * H))

        
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=w)

        
        draw.line((x - 2*r, y, x + 2*r, y), fill=color, width=w)
        draw.line((x, y - 2*r, x, y + 2*r), fill=color, width=w)

        
        draw.text((x + r + 2, y + r + 2), f"({x},{y})", fill=color)

    return img


class GroundedSAMMasker:
    def __init__(self, cfg: GroundedSAMConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Resolve GroundingDINO config path (use package config if local path does not exist)
        dino_config_path = cfg.dino_config_path
        if not os.path.isfile(dino_config_path):
            import groundingdino
            _pkg_dir = os.path.dirname(os.path.abspath(groundingdino.__file__))
            dino_config_path = os.path.join(_pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
        if not os.path.isfile(dino_config_path):
            raise FileNotFoundError(
                f"GroundingDINO config not found at {cfg.dino_config_path} nor at {dino_config_path}"
            )

        self._dino_config_path = dino_config_path
        self.dino = None  # lazy-loaded for sam1 backend only

        # SAM / SAM3
        self.sam_backend = getattr(cfg, "sam_backend", "sam1")
        self._dino_device = self.device
        if self.sam_backend != "sam3" and not _groundingdino_has_cuda_ops():
            self._dino_device = torch.device("cpu")
            if self.device.type == "cuda":
                import warnings
                warnings.warn(
                    "GroundingDINO CUDA ops not built; running DINO on CPU, SAM on GPU.",
                    stacklevel=2,
                )

        self.sam3 = None
        self.sam3_video = None
        self.sam_predictor = None
        if self.sam_backend == "sam3":
            self.sam3 = SAM3Segmenter(
                checkpoint=getattr(cfg, "sam3_checkpoint_path", DEFAULT_SAM3_CKPT),
                device=str(self.device),
                save=not getattr(cfg, "fast_mode", False),
            )
            if not getattr(cfg, "fast_mode", False):
                self.sam3_video = SAM3VideoTracker(
                    checkpoint=getattr(cfg, "sam3_checkpoint_path", DEFAULT_SAM3_CKPT),
                    device=str(self.device),
                )
        else:
            sam = sam_model_registry[cfg.sam_type](checkpoint=cfg.sam_checkpoint_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)

    def _ensure_dino(self):
        if self.dino is None:
            self.dino = GroundingDINOModel(
                model_config_path=self._dino_config_path,
                model_checkpoint_path=self.cfg.dino_checkpoint_path,
                device=str(self._dino_device),
            )

    def _segment_phrases_sam3(
        self,
        image_rgb: np.ndarray,
        phrases: List[str],
        lang: str = "",
        spatial_pick: Optional[SpatialPickSpec] = None,
        prev_box: Optional[np.ndarray] = None,
        prev_frames: Optional[List[np.ndarray]] = None,
        is_green: bool = False,
    ) -> np.ndarray:
        """SAM3 text/bbox segmentation with spatial disambiguation and temporal tracking."""
        if not phrases and prev_box is None:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        H, W = image_rgb.shape[:2]
        self.sam3.set_image(image_rgb)

        if prev_box is not None:
            mask, _ = self.sam3.segment_from_prev_box(prev_box)
            if mask.any() and mask.shape == (H, W):
                return mask

        if prev_frames and spatial_pick and len(prev_frames) >= 2 and not getattr(self.cfg, "fast_mode", False):
            text = sam3_text_for_source(spatial_pick.source_phrase)
            try:
                mask, _ = self.sam3_video.track_text_on_clip(prev_frames + [image_rgb], text)
                if mask is not None and mask.any() and mask.shape == (H, W):
                    return mask
            except Exception:
                pass

        anchor_boxes: Dict[str, np.ndarray] = {}
        if spatial_pick and spatial_pick.anchor_phrases:
            for a in spatial_pick.anchor_phrases:
                _, box = self.sam3.segment_best_text([sam3_text_for_anchor(a)])
                if box is not None:
                    anchor_boxes[a] = box

        union = np.zeros((H, W), dtype=bool)
        skip_right = ("cabinet" not in lang.lower()) and ("rack" not in lang.lower())

        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            masks, boxes, scores = self.sam3.segment_text([phrase])
            if not masks:
                continue

            boxes_list = list(boxes)
            scores_list = list(scores)

            if "plate" in phrase.lower() and len(boxes_list) > 1:
                idx = int(np.argmax([b[3] for b in boxes_list]))
                union |= masks[idx]
                continue

            if ("on the left" in phrase.lower() or phrase.lower() == "stove") and len(boxes_list) > 1:
                cx = [(b[0] + b[2]) * 0.5 for b in boxes_list]
                idx = int(np.argmin(cx))
                union |= masks[idx]
                continue

            if skip_right and boxes_list:
                cx = np.array([(b[0] + b[2]) * 0.5 for b in boxes_list])
                keep = cx <= 0.6 * W
                if keep.any():
                    masks = [m for m, k in zip(masks, keep) if k]
                    boxes_list = [b for b, k in zip(boxes_list, keep) if k]
                    scores_list = [s for s, k in zip(scores_list, keep) if k]

            if spatial_pick and len(boxes_list) > 1:
                if is_green:
                    idx = select_dest_index(
                        boxes_list, scores_list, spatial_pick.dest_spatial_rule, (H, W),
                    )
                elif (
                    spatial_pick.source_object in ("bowl", "book")
                    or "bowl" in phrase.lower()
                    or "book" in phrase.lower()
                ):
                    idx = select_instance_index(
                        boxes_list, scores_list, spatial_pick.spatial_rule, anchor_boxes, (H, W),
                    )
                else:
                    idx = int(np.argmax(scores_list))
                union |= masks[idx]
                continue

            for m in masks:
                union |= m

        return union

    @torch.inference_mode()
    def _segment_phrases(
        self,
        image_rgb: np.ndarray,
        phrases: List[str],
        lang: str = "",
        shift_box_x_pixels: Optional[int] = None,
        shift_phrase: Optional[str] = None,
        spatial_pick: Optional[SpatialPickSpec] = None,
        prev_red_box: Optional[np.ndarray] = None,
        prev_green_box: Optional[np.ndarray] = None,
        prev_frames: Optional[List[np.ndarray]] = None,
        is_green: bool = False,
    ) -> np.ndarray:
        """
        Returns a union mask (H,W) bool for all phrases.
        When shift_box_x_pixels and shift_phrase are set, shift the detected box
        for that phrase (e.g. "plate") by offset before SAM - used for "plate beside" mask.
        """
        if self.sam_backend == "sam3" and self.sam3 is not None:
            prev_box = prev_green_box if is_green else prev_red_box
            return self._segment_phrases_sam3(
                image_rgb,
                phrases,
                lang=lang,
                spatial_pick=spatial_pick,
                prev_box=prev_box,
                prev_frames=prev_frames if not is_green else None,
                is_green=is_green,
            )

        if not phrases:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        self._ensure_dino()
        H, W = image_rgb.shape[:2]
        union = np.zeros((H, W), dtype=bool)
        # 任务里没有 cabinet/rack 时，不 box 右边的物体（避免误框柜子）
        skip_right = ("cabinet" not in lang.lower()) and ("rack" not in lang.lower())

        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue

            # GroundingDINO expects "phrase." sometimes; keep it robust
            prompt = phrase if phrase.endswith(".") else (phrase + ".")

            # Predict boxes with GroundingDINO
            detections = self.dino.predict_with_classes(
                image=image_rgb,
                classes=[prompt],
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
            )
            # detections.xyxy is (N,4) in pixel coords
            if detections is None or len(detections) == 0:
                continue

            boxes_xyxy = detections.xyxy
            if boxes_xyxy is None or len(boxes_xyxy) == 0:
                continue

            # plate 检测出两个（盘+碗）时，取下面的 mask（盘子）
            if "plate" in phrase.lower() and len(boxes_xyxy) > 1:
                bottommost_idx = np.argmax(boxes_xyxy[:, 3])  # y2 最大 = 最下面
                boxes_xyxy = np.array([boxes_xyxy[bottommost_idx]], dtype=boxes_xyxy.dtype)

            # stove/左边扁方块：多个框时取最左边的一个
            if ("on the left" in phrase.lower() or "left" in phrase.lower()) and len(boxes_xyxy) > 1:
                leftmost_idx = np.argmin((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2)
                boxes_xyxy = np.array([boxes_xyxy[leftmost_idx]], dtype=boxes_xyxy.dtype)

            # 任务无 cabinet/rack 时，排除靠右的 box（多是柜子误检）
            if skip_right and len(boxes_xyxy) > 0:
                cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
                keep_mask = cx <= 0.6 * W
                boxes_xyxy = boxes_xyxy[keep_mask]
            if len(boxes_xyxy) == 0:
                continue

            # Shift box for "plate beside" mask: same shape, different location
            if (
                shift_box_x_pixels is not None
                and shift_phrase is not None
                and shift_phrase.lower() in phrase.lower()
            ):
                boxes_xyxy = boxes_xyxy.copy()
                boxes_xyxy[:, [0, 2]] = np.clip(
                    boxes_xyxy[:, [0, 2]] + shift_box_x_pixels, 0, W
                )

            # SAM expects boxes as torch tensor on device in XYXY
            boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device)

            # Transform boxes to SAM input space
            transformed = self.sam_predictor.transform.apply_boxes_torch(boxes_t, (H, W))

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            # masks: (N,1,H,W) bool tensor
            m = masks.squeeze(1).detach().cpu().numpy().astype(bool)  # (N,H,W)
            union |= m.any(axis=0)

        return union

    @torch.inference_mode()
    def _segment_points(self, image_rgb: np.ndarray, points_xy, *, box_half: int = 28):
        """Segment with a normalized click point (+ small box constraint)."""
        if not points_xy:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        H, W = image_rgb.shape[:2]
        px, py = points_xy[0]
        cx = int(px * W)
        cy = int(py * H)

        x1 = max(0, cx - box_half)
        y1 = max(0, cy - box_half)
        x2 = min(W, cx + box_half)
        y2 = min(H, cy + box_half)

        if self.sam_backend == "sam3" and self.sam3 is not None:
            self.sam3.set_image(image_rgb)
            masks, _, _ = self.sam3.segment_bboxes([[float(x1), float(y1), float(x2), float(y2)]])
            if masks:
                return masks[0].astype(bool)
            return np.zeros((H, W), dtype=bool)

        if self.sam_predictor is None:
            return np.zeros((H, W), dtype=bool)

        self.sam_predictor.set_image(image_rgb)
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        pts = np.array([[cx, cy]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=pts,
            point_labels=labels,
            box=box,
            multimask_output=False,
        )
        return masks[0].astype(bool)

    @torch.inference_mode()
    def _segment_source_tracked(
        self,
        image_rgb: np.ndarray,
        points_xy: List[Tuple[float, float]],
        *,
        spatial_pick: Optional[SpatialPickSpec] = None,
        green_points_xy: Optional[List[Tuple[float, float]]] = None,
        prev_box: Optional[np.ndarray] = None,
        prev_red_center: Optional[Tuple[float, float]] = None,
        init_red_center: Optional[Tuple[float, float]] = None,
        prev_frames: Optional[List[np.ndarray]] = None,
        lang: str = "",
    ) -> np.ndarray:
        """
        Red/source mask for pick tasks.

        - Init (no prev_box): click + spatial disambiguation to lock the target bowl.
        - Track (prev_box set): SAM3 bbox propagation on every frame; video clip if needed.
          Do NOT re-anchor to the init click — follow the carried object.
        """
        H, W = image_rgb.shape[:2]
        if self.sam_backend != "sam3" or self.sam3 is None:
            return self._segment_points(image_rgb, points_xy)

        click_xy = points_xy[0] if points_xy else None
        green_xy = green_points_xy[0] if green_points_xy else None
        spatial_rule = spatial_pick.spatial_rule if spatial_pick else None
        fast = getattr(self.cfg, "fast_mode", False)
        click_halves = (36,) if fast else (28, 36, 44)
        bbox_margins = (0,) if fast else (0, 32, 64)

        self.sam3.set_image(image_rgb)

        # ---- TRACK: follow the object locked on frame 0 ----
        if prev_box is not None:
            ref_x = (prev_red_center[0] * W) if prev_red_center else (prev_box[0] + prev_box[2]) * 0.5
            ref_y = (prev_red_center[1] * H) if prev_red_center else (prev_box[1] + prev_box[3]) * 0.5

            def _center_ok(mask: np.ndarray, max_dist: float = 80.0) -> bool:
                if not mask.any():
                    return False
                ys, xs = np.where(mask)
                cx, cy = float(xs.mean()), float(ys.mean())
                return float(np.hypot(cx - ref_x, cy - ref_y)) <= max_dist

            # 1) Click at previous mask centroid (cheap, follows motion frame-to-frame)
            if prev_red_center is not None:
                for half in click_halves:
                    clicked = self._segment_points(image_rgb, [prev_red_center], box_half=half)
                    if _center_ok(clicked, max_dist=half + 20):
                        return clicked.astype(bool)

            # 2) Bbox propagation (+ expanded search if object moved)
            for margin in bbox_margins:
                mask, _ = self.sam3.segment_from_prev_box(prev_box, margin=margin)
                if _center_ok(mask, max_dist=margin + 50):
                    return mask.astype(bool)

            if fast:
                mask, _ = self.sam3.segment_from_prev_box(prev_box, margin=0)
                if mask.any() and mask.shape == (H, W):
                    return mask.astype(bool)
                return np.zeros((H, W), dtype=bool)

            # 3) Text detect: prefer bowl that left init spot if multiple (carry phase)
            masks, boxes, scores = self.sam3.segment_text(["black bowl"])
            if masks:
                centers = np.array([((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in boxes])
                dists_prev = np.hypot(centers[:, 0] - ref_x, centers[:, 1] - ref_y)
                if init_red_center is not None and len(masks) > 1:
                    icx = init_red_center[0] * W
                    icy = init_red_center[1] * H
                    dists_init = np.hypot(centers[:, 0] - icx, centers[:, 1] - icy)
                    moved = dists_init > 45
                    if moved.any():
                        cand = np.where(moved)[0]
                        idx = int(cand[np.argmin(dists_prev[cand])])
                        if float(dists_prev[idx]) < 150:
                            return masks[idx].astype(bool)
                idx = int(np.argmin(dists_prev))
                if float(dists_prev[idx]) < 120:
                    return masks[idx].astype(bool)

            # 4) Video clip fallback (slow — only when bbox/point track lost)
            if prev_frames and len(prev_frames) >= 2 and self.sam3_video is not None:
                clip = prev_frames[-3:] + [image_rgb]
                try:
                    mask, _ = self.sam3_video.track_text_on_clip(
                        clip, "black bowl", target_frame_idx=-1,
                    )
                    if mask is not None and mask.any() and mask.shape == (H, W):
                        return mask.astype(bool)
                except Exception:
                    pass

            return np.zeros((H, W), dtype=bool)

        # ---- INIT: first frame — click / spatial pick ----
        anchor_boxes: Dict[str, np.ndarray] = {}
        if spatial_pick and spatial_pick.anchor_phrases:
            for a in spatial_pick.anchor_phrases:
                _, box = self.sam3.segment_best_text([sam3_text_for_anchor(a)])
                if box is not None:
                    anchor_boxes[a] = box

        if click_xy is not None and spatial_rule == "on_cabinet":
            clicked = self._segment_points(image_rgb, points_xy, box_half=12)
            if clicked.any():
                return clicked.astype(bool)

        masks, boxes, scores = self.sam3.segment_text(["black bowl"])
        picked = select_tracked_bowl(
            masks, boxes, scores, (H, W),
            click_xy=click_xy,
            green_xy=green_xy,
            prev_box=None,
            spatial_rule=spatial_rule,
            anchor_boxes=anchor_boxes or None,
        )
        if picked is not None and picked.any():
            return picked.astype(bool)

        if click_xy is not None:
            return self._segment_points(image_rgb, points_xy, box_half=18)

        return np.zeros((H, W), dtype=bool)



    def mask_image_from_lang(
        self,
        img_pil: Image.Image,
        lang: str,
        *,
        return_masks: bool = False,
        alpha: float = 0.35,
        shift_green_plate_pixels: Optional[int] = None,
        draw_green: bool = True,
        prev_red_box: Optional[np.ndarray] = None,
        prev_green_box: Optional[np.ndarray] = None,
        prev_red_center: Optional[Tuple[float, float]] = None,
        init_red_center: Optional[Tuple[float, float]] = None,
        prev_frames: Optional[List[np.ndarray]] = None,
        proprio_state: Optional[np.ndarray] = None,
        joint_state: Optional[np.ndarray] = None,
        draw_click_overlay: bool = False,
    ):
        """
        Output: PIL RGB image with black background and colored masks:
          - red objects -> (255,0,0)
          - green objects -> (0,255,0)  (omitted if draw_green=False, e.g. hide plate mask)

        If return_masks=True, returns (out_pil, red_mask, green_mask).
        """
        # stove -> 左边的扁方块（白色矩形盒）
        lang = lang.replace("stove", "white rectangular box on the left")

        lang = lang.replace("rack",
                          "the yellow and white striped rack near the edge of the table")
        lang = lang.replace("wine bottle","right wine bottle")
        spec = build_mask_spec_from_lang(lang)

        image_rgb = np.array(img_pil.convert("RGB"),dtype=np.uint8)
        if self.sam_backend != "sam3" and self.sam_predictor is not None:
            self.sam_predictor.set_image(image_rgb)

        # === RED ===
        if spec.red_points_xy:
            red_mask = self._segment_source_tracked(
                image_rgb,
                spec.red_points_xy,
                spatial_pick=getattr(spec, "spatial_pick", None),
                green_points_xy=getattr(spec, "green_points_xy", None),
                prev_box=prev_red_box,
                prev_red_center=prev_red_center,
                init_red_center=init_red_center,
                prev_frames=prev_frames,
                lang=lang,
            )
        else:
            red_mask = self._segment_phrases(
                image_rgb,
                spec.red_phrases,
                lang=lang,
                spatial_pick=getattr(spec, "spatial_pick", None),
                prev_red_box=prev_red_box,
                prev_frames=prev_frames,
                is_green=False,
            )

        # === GREEN ===
        if spec.green_points_xy:
            green_mask = self._segment_points(image_rgb, spec.green_points_xy)
        else:
            green_mask = self._segment_phrases(
                image_rgb,
                spec.green_phrases,
                lang=lang,
                shift_box_x_pixels=shift_green_plate_pixels,
                shift_phrase="plate" if shift_green_plate_pixels is not None else None,
                prev_green_box=prev_green_box,
                is_green=True,
            )

        # When shifting green (plate): exclude original plate region from red so plate is not painted red
        if shift_green_plate_pixels is not None and spec.green_phrases:
            original_plate_mask = self._segment_phrases(
                image_rgb, spec.green_phrases, lang=lang,
            )
            red_mask = red_mask & (~original_plate_mask)

        if spec.red_points_xy or spec.green_points_xy:
            debug_rgb = Image.fromarray(image_rgb.copy(), mode="RGB")
            if getattr(spec, "green_points_xy", None):
                debug_rgb = _draw_points_overlay(debug_rgb, spec.green_points_xy, color=(0, 255, 0), r=10, w=3)
            if getattr(spec, "red_points_xy", None):
                debug_rgb = _draw_points_overlay(debug_rgb, spec.red_points_xy, color=(255, 0, 0), r=10, w=3)

            os.makedirs("debug_points", exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", lang)[:120]
            debug_rgb.save(f"debug_points/{safe}_points.png")


        # Compose: black background
        out = np.zeros_like(image_rgb, dtype=np.uint8)
        light_red = np.array([255, 120, 120], dtype=np.float32)
        light_green = np.array([120, 255, 120], dtype=np.float32)
        red_mask = red_mask & (~green_mask)
        if red_mask.any():
            tinted = (1.0 - alpha) * image_rgb[red_mask] + alpha * light_red
            out[red_mask] = np.clip(tinted, 0, 255).astype(np.uint8)
        if draw_green and green_mask.any():
            tinted = (1.0 - alpha) * image_rgb[green_mask] + alpha * light_green
            out[green_mask] = np.clip(tinted, 0, 255).astype(np.uint8)

        # Gripper white dots: RLDS proprio (sim) or Roboflow (real_perception)
        gripper_centers = []
        if getattr(self.cfg, "gripper_enabled", True):
            if proprio_state is not None or joint_state is not None:
                try:
                    from gripper_project import gripper_pixels_from_obs

                    gripper_centers = gripper_pixels_from_obs(
                        proprio_state, joint_state=joint_state
                    )
                    if gripper_centers:
                        H, W = image_rgb.shape[:2]
                        radius = max(2, min(5, min(H, W) // 70))
                        out = _draw_white_dots(out, gripper_centers, radius=radius)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Gripper projection from proprio failed: {e}")
            else:
                use_roboflow = (
                    is_real_perception_mode()
                    or getattr(self.cfg, "perception_mode", "sim") == "real_perception"
                )
                if use_roboflow and getattr(self.cfg, "gripper_model_id", None):
                    try:
                        gripper_centers = _detect_gripper_centers(image_rgb, self.cfg.gripper_model_id)
                        if gripper_centers:
                            H, W = image_rgb.shape[:2]
                            radius = max(2, min(5, min(H, W) // 70))
                            out = _draw_white_dots(out, gripper_centers, radius=radius)
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Gripper detection failed: {e}")

        # If nothing was drawn (all black): return original image
        if not red_mask.any() and not green_mask.any() and not gripper_centers:
            if return_masks:
                return img_pil, red_mask, green_mask
            return img_pil

        out_pil = Image.fromarray(out, mode="RGB")

        if draw_click_overlay:
            if getattr(spec, "green_points_xy", None):
                out_pil = _draw_points_overlay(out_pil, spec.green_points_xy, color=(0, 255, 0), r=10, w=3)
            if getattr(spec, "red_points_xy", None) and prev_red_box is None:
                out_pil = _draw_points_overlay(out_pil, spec.red_points_xy, color=(255, 0, 0), r=10, w=3)
            elif red_mask.any():
                ys, xs = np.where(red_mask)
                cx, cy = int(xs.mean()), int(ys.mean())
                W, H = out_pil.size
                out_pil = _draw_points_overlay(
                    out_pil, [(cx / W, cy / H)], color=(255, 0, 0), r=8, w=2
                )

        if return_masks:
            return out_pil, red_mask, green_mask
        return out_pil


class EpisodeMaskTracker:
    """Per-episode temporal mask tracking: SAM3 bbox tracking + short video clip on occlusion."""

    def __init__(self, masker: GroundedSAMMasker, max_history: int = 8):
        self.masker = masker
        self.max_history = max_history
        self.reset()

    def reset(self):
        self.frames_rgb: List[np.ndarray] = []
        self.prev_red_box: Optional[np.ndarray] = None
        self.prev_green_box: Optional[np.ndarray] = None
        self.prev_red_center: Optional[Tuple[float, float]] = None
        self.init_red_center: Optional[Tuple[float, float]] = None
        self.lang: str = ""

    def mask_image_from_lang(self, img_pil: Image.Image, lang: str, proprio_state=None, joint_state=None, **kwargs):
        image_rgb = np.array(img_pil.convert("RGB"))
        self.frames_rgb.append(image_rgb)
        if len(self.frames_rgb) > self.max_history:
            self.frames_rgb.pop(0)
        self.lang = lang

        want_masks = kwargs.pop("return_masks", False)
        out_pil, red_mask, green_mask = self.masker.mask_image_from_lang(
            img_pil,
            lang,
            prev_red_box=self.prev_red_box,
            prev_green_box=self.prev_green_box,
            prev_red_center=self.prev_red_center,
            init_red_center=self.init_red_center,
            prev_frames=list(self.frames_rgb[:-1]),
            proprio_state=proprio_state,
            joint_state=joint_state,
            return_masks=True,
            **kwargs,
        )

        rb = _mask_to_box(red_mask) if red_mask is not None else None
        gb = _mask_to_box(green_mask) if green_mask is not None else None
        if rb is not None:
            self.prev_red_box = rb
        if gb is not None:
            self.prev_green_box = gb
        if red_mask is not None and red_mask.any():
            ys, xs = np.where(red_mask)
            H, W = red_mask.shape
            self.prev_red_center = (float(xs.mean()) / W, float(ys.mean()) / H)
            if self.init_red_center is None:
                self.init_red_center = self.prev_red_center

        if want_masks:
            return out_pil, red_mask, green_mask
        return out_pil


# --------------------------
# 3) Minimal CLI test
# --------------------------

def main():
    """
    Example:
      python masked_grounded_sam.py --image /path/to/frame.png --lang "put the bowl on the stove"
    """
    import argparse

    ap = argparse.ArgumentParser()






    dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_ckpt = "groundingdino_swint_ogc.pth"
    sam_ckpt = "sam_vit_h_4b8939.pth"
    sam_type = "vit_h"
    #sam_type = "vit_b"
    device = "cuda"
    out_path = "/home/ubuntu/16831pro_fine_tune/zz/masked_cheese.png"

    cfg = GroundedSAMConfig(
        dino_config_path=dino_config,
        dino_checkpoint_path=dino_ckpt,
        sam_checkpoint_path=sam_ckpt,
        sam_type=sam_type,
        device=device,
    )
    lang = "put the cream cheese in the bowl"
    image = "/home/ubuntu/16831pro_fine_tune/zt/plate.jpg"
    masker = GroundedSAMMasker(cfg)
    img = Image.open(image).convert("RGB")
    out = masker.mask_image_from_lang(img, lang)

    out.save(out_path)
    print(f"[OK] Saved masked image to: {out_path}")


if __name__ == "__main__":
    main()


