"""RGB frames -> per-frame instance masks via SAM2 automatic mask generation.

Identity is no longer SAM2's job — the Fuser associates masks into objects by
appearance + occupancy across frames (see fusion.py). So SAM2 is a stateless
per-frame segmenter: `segment(rgb)` runs the automatic mask generator on one
frame and returns its boolean instance masks. No video propagation, chunking,
or cross-chunk id carry-over — which also means only the mask-generation model
is loaded (not the video model), and there is no CPU/GPU state offload.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image
from transformers import pipeline

from herald.scene.recon.utils import as_rgb, to_bool

DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"


class Sam2Segmenter:
    """Per-frame SAM2 automatic mask generation.

    segment(rgb) -> list of boolean masks (RGB resolution), largest first, each
    at least `min_area` pixels. run_stream / run_video wrap it for iterables and
    whole sequences of RGB arrays or image paths.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        points_per_side: int = 16,
        points_per_batch: int = 32,
        min_area: int = 200,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.9,
    ) -> None:
        self.maskgen = pipeline("mask-generation", model=model_name, device=device)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.min_area = min_area
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

    def segment(self, rgb) -> list[np.ndarray]:
        """One RGB frame (array or path) -> its boolean instance masks."""
        out = self.maskgen(
            Image.fromarray(as_rgb(rgb)),
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        masks = [m for m in map(to_bool, out["masks"]) if int(m.sum()) >= self.min_area]
        masks.sort(key=lambda m: int(m.sum()), reverse=True)
        return masks

    def run_stream(self, frames: Iterable) -> Iterator[list[np.ndarray]]:
        """Lazily yield the mask list for each frame (RGB array or path)."""
        for f in frames:
            yield self.segment(f)

    def run_video(self, frames: Sequence) -> list[list[np.ndarray]]:
        """Whole-sequence convenience: one mask list per frame."""
        return [self.segment(f) for f in frames]
