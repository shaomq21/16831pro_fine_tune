"""SAM3 backend (Ultralytics) for text/bbox concept segmentation and temporal tracking."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_SAM3_CKPT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/var/lib/docker/data/hf_cache/hub/models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
)


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _box_center(box: np.ndarray) -> Tuple[float, float]:
    return float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5)


class SAM3Segmenter:
    """Text/bbox concept segmentation via Ultralytics SAM3SemanticPredictor."""

    def __init__(self, checkpoint: str = DEFAULT_SAM3_CKPT, device: str = "cuda", conf: float = 0.20, save: bool = True):
        from ultralytics.models.sam import SAM3SemanticPredictor

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {checkpoint}. "
                "Request access at https://huggingface.co/facebook/sam3 and set SAM3_CHECKPOINT."
            )
        dev = device if torch.cuda.is_available() else "cpu"
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=checkpoint,
            verbose=False,
            save=save,
            device=dev,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self._current_path: Optional[str] = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Cache image features for repeated queries on the same frame."""
        tmp = os.path.join("/tmp", f"sam3_frame_{id(self)}.png")
        Image.fromarray(image_rgb).save(tmp)
        self.predictor.set_image(tmp)
        self._current_path = tmp

    @staticmethod
    def _parse_results(results) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        if results is None:
            return [], [], []
        r = results[0] if isinstance(results, list) and results else results
        if r is None or r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return [], [], []
        masks_t = r.masks.data.detach().cpu().numpy().astype(bool)  # (N,H,W)
        boxes_t = r.boxes.xyxy.detach().cpu().numpy()  # (N,4)
        scores_t = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes_t))
        masks = [masks_t[i] for i in range(masks_t.shape[0])]
        boxes = [boxes_t[i] for i in range(boxes_t.shape[0])]
        scores = [float(scores_t[i]) for i in range(len(boxes))]
        return masks, boxes, scores

    def segment_text(self, text_phrases: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        if not text_phrases:
            return [], [], []
        results = self.predictor(text=[p if p.endswith(".") else p + "." for p in text_phrases])
        return self._parse_results(results)

    def segment_bboxes(self, bboxes_xyxy: List[List[float]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        if not bboxes_xyxy:
            return [], [], []
        results = self.predictor(bboxes=bboxes_xyxy)
        return self._parse_results(results)

    def segment_union_text(self, text_phrases: List[str]) -> np.ndarray:
        masks, _, _ = self.segment_text(text_phrases)
        if not masks:
            return None  # type: ignore
        union = np.zeros_like(masks[0], dtype=bool)
        for m in masks:
            union |= m
        return union

    def segment_best_text(
        self,
        text_phrases: List[str],
        *,
        select_idx=None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (mask, box) for one instance; select_idx picks among multiple detections."""
        masks, boxes, scores = self.segment_text(text_phrases)
        if not masks:
            return np.zeros((1, 1), dtype=bool), None
        if select_idx is not None:
            idx = int(select_idx(masks, boxes, scores))
            idx = max(0, min(idx, len(masks) - 1))
            return masks[idx], boxes[idx]
        # default: highest score
        idx = int(np.argmax(scores))
        return masks[idx], boxes[idx]

    def segment_from_prev_box(self, prev_box: np.ndarray, margin: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Track object in current frame using bbox exemplar from previous frame."""
        x1, y1, x2, y2 = [float(v) for v in prev_box.tolist()]
        if margin > 0:
            x1, y1 = x1 - margin, y1 - margin
            x2, y2 = x2 + margin, y2 + margin
        masks, boxes, scores = self.segment_bboxes([[x1, y1, x2, y2]])
        if not masks:
            return np.zeros((1, 1), dtype=bool), None
        idx = int(np.argmax(scores))
        return masks[idx], boxes[idx]


class SAM3VideoTracker:
    """Short-clip temporal tracking: write recent frames to mp4 and run SAM3 video semantic predictor."""

    def __init__(self, checkpoint: str = DEFAULT_SAM3_CKPT, device: str = "cuda", conf: float = 0.20, save: bool = True):
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        dev = device if torch.cuda.is_available() else "cpu"
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            imgsz=640,
            model=checkpoint,
            verbose=False,
            save=save,
            device=dev,
        )
        self.predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        self._session_path: Optional[str] = None

    @staticmethod
    def _write_clip(frames_rgb: List[np.ndarray], out_path: str, fps: int = 5) -> str:
        import imageio

        imageio.mimsave(out_path, frames_rgb, fps=fps)
        return out_path

    def track_text_on_clip(
        self,
        frames_rgb: List[np.ndarray],
        text: str,
        target_frame_idx: int = -1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run video semantic tracking; return mask+box on target_frame_idx (default last)."""
        if not frames_rgb:
            return np.zeros((1, 1), dtype=bool), None
        clip_path = os.path.join("/tmp", f"sam3_clip_{id(self)}.mp4")
        self._write_clip(frames_rgb, clip_path)
        results = self.predictor(source=clip_path, text=[text], stream=True)
        frame_masks: List[np.ndarray] = []
        frame_boxes: List[Optional[np.ndarray]] = []
        for r in results:
            masks, boxes, _ = SAM3Segmenter._parse_results([r])
            if masks:
                frame_masks.append(masks[0])
                frame_boxes.append(boxes[0])
            else:
                frame_masks.append(np.zeros(frames_rgb[0].shape[:2], dtype=bool))
                frame_boxes.append(None)
        idx = target_frame_idx if target_frame_idx >= 0 else len(frame_masks) - 1
        idx = min(idx, len(frame_masks) - 1)
        return frame_masks[idx], frame_boxes[idx]
