"""RGB frames -> instance masks via SAM2 (Segment Anything 2).

Service-side home of the SAM2 model wrapper. A single `Sam2` class covers both
uses; the `track` constructor flag decides which machinery loads:

- track=False -- stateless per-frame automatic mask generation (AMG).
  `segment(rgb)` collapses the frame's masks into one uint16 label image (ids
  1..N largest-first, 0 = background); `return_masks=True` yields the raw
  boolean-mask list instead. No cross-frame identity (ids are per-frame;
  association is left to the recon-side merge downstream).
- track=True  -- additionally loads the SAM2 *video* model and becomes a
  chunk-buffered tracker. `push(rgb)` buffers frames and, when a chunk fills,
  seeds the chunk's first frame with AMG masks and propagates them through the
  video model, returning per-frame (label, conf) pairs; `finish()` flushes the
  tail; `reset()` clears the buffer for a new stream. Ids are FRESH local ids
  (1..N) each chunk -- a lifelong session OOMs (every discovery adds a
  conditioning frame SAM2 never prunes), so memory is bounded by re-creating a
  session per chunk; cross-chunk identity is handled by the recon-side merge.

The AMG loads in both modes, so `segment` works even when track=True. Heavy CUDA
deps (torch + transformers>=4.56, `uv sync --group seg`), imported only by recon
code that has opted into segmentation, so importing `services` stays cheap.
Self-contained: no imports from herald (the tiny RGB/erode helpers live here).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import (
    Sam2VideoModel,
    Sam2VideoProcessor,
    pipeline
)

DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"


def _as_rgb(frame) -> np.ndarray:
    """RGB ndarray passthrough, or image path -> HxWx3 uint8 array."""
    if isinstance(frame, np.ndarray):
        return frame
    return np.asarray(Image.open(frame).convert("RGB"))


def _erode(lab: np.ndarray, iterations: int) -> np.ndarray:
    """Shrink each label region so mixed-object border pixels become 0 (bg) --
    shaves the flicker-prone mask edges before the label image leaves here."""
    for _ in range(iterations):
        p = np.pad(lab, 1, mode="edge")
        c = p[1:-1, 1:-1]
        keep = ((c == p[:-2, 1:-1]) & (c == p[2:, 1:-1])
                & (c == p[1:-1, :-2]) & (c == p[1:-1, 2:]))
        lab = np.where(keep, lab, 0).astype(lab.dtype)
    return lab


def _automatic_masks(
    maskgen,
    image_arr: np.ndarray,
    *,
    points_per_side: int,
    points_per_batch: int,
    min_area: int,
    iou_thresh: float,
    stability_thresh: float,
) -> list[np.ndarray]:
    """AMG on one frame -> boolean masks, largest-first, each >= min_area."""
    out = maskgen(
        Image.fromarray(image_arr),
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=iou_thresh,             # HF AMG kwarg names
        stability_score_thresh=stability_thresh,
    )
    masks = [b for b in (np.asarray(m).astype(bool) for m in out["masks"])
             if int(b.sum()) >= min_area]
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks


def _collapse(masks: list[np.ndarray], shape) -> np.ndarray:
    """Paint a per-frame mask list (largest-first) into one uint16 label image:
    ids 1..N by descending area, later (smaller) masks overwrite earlier ones so
    finer masks win; 0 = background. Same largest-first rule the tracker uses."""
    lab = np.zeros(shape, np.uint16)
    for i, m in enumerate(masks, 1):
        lab[m] = i
    return lab


class Sam2:
    """SAM2 mask generation -- per-frame (track=False) or chunk-buffered video
    tracking (track=True). See the module docstring for the two modes; the AMG
    (`segment`) is available in both."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        points_per_side: int = 16,
        points_per_batch: int = 32,
        min_area: int = 200,
        iou_thresh: float = 0.7,
        stability_thresh: float = 0.9,
        erode: int = 0,         # >0: shave N px off collapsed label borders (segment mode)
        track: bool = False,
        chunk: int = 16,        # track=True only: frames per SAM2 session (bounds memory)
        offload: bool = True,   # track=True only: keep session state on CPU (else GPU)
    ) -> None:
        self.device = device
        self.track = track
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.min_area = min_area
        self.iou_thresh = iou_thresh
        self.stability_thresh = stability_thresh
        self.erode = max(0, erode)
        self.maskgen = pipeline("mask-generation", model=model_name, device=device)
        self.model = None
        self.processor = None
        if track:
            self.model = Sam2VideoModel.from_pretrained(model_name).to(device).eval()
            self.processor = Sam2VideoProcessor.from_pretrained(model_name)
            self.chunk = max(1, chunk)
            self.offload = offload
            self.reset()

    def _seed(self, image_arr: np.ndarray) -> list[np.ndarray]:
        """Automatic masks on one frame, largest-first, >= min_area."""
        return _automatic_masks(
            self.maskgen, image_arr,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            min_area=self.min_area,
            iou_thresh=self.iou_thresh,
            stability_thresh=self.stability_thresh,
        )

    # ------------------------------------------------ per-frame (both modes) ---
    def segment(self, rgb, *, return_masks: bool = False):
        """One RGB frame (array or path) -> a uint16 label image with its masks
        collapsed largest-first (ids 1..N, 0 = background). Pass return_masks=True
        to get the raw list of boolean masks instead (e.g. for per-mask 3D
        segments / crops downstream)."""
        arr = _as_rgb(rgb)
        masks = self._seed(arr)
        if return_masks:
            return masks
        lab = _collapse(masks, arr.shape[:2])
        return _erode(lab, self.erode) if self.erode else lab

    def run_stream(self, frames: Iterable, *, return_masks: bool = False) -> Iterator:
        """Lazily yield segment(f) for each frame (label image, or mask list)."""
        for f in frames:
            yield self.segment(f, return_masks=return_masks)

    def run_video(self, frames: Sequence, *, return_masks: bool = False) -> list:
        """Whole-sequence convenience: one segment(f) per frame."""
        return [self.segment(f, return_masks=return_masks) for f in frames]

    # ------------------------------------------------- video tracking (track) ---
    def reset(self) -> None:
        """Clear the chunk buffer so the same model can track a new stream."""
        self._buffer: list[np.ndarray] = []

    def push(self, rgb) -> list[tuple[np.ndarray, np.ndarray]]:
        """Buffer a frame; return this chunk's (label uint16, conf float32) pairs
        once it fills, else []. Requires track=True."""
        self._require_track()
        self._buffer.append(_as_rgb(rgb))
        if len(self._buffer) >= self.chunk:
            frames, self._buffer = self._buffer, []
            return self._run_chunk(frames)
        return []

    def finish(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Flush the partial trailing chunk. Requires track=True."""
        self._require_track()
        frames, self._buffer = self._buffer, []
        return self._run_chunk(frames) if frames else []

    def _require_track(self) -> None:
        if not self.track:
            raise RuntimeError(
                "Sam2 was created with track=False; pass track=True for video "
                "tracking (push/finish)."
            )

    def _run_chunk(self, frames) -> list[tuple[np.ndarray, np.ndarray]]:
        frames = [_as_rgb(f) for f in frames]
        n = len(frames)
        h, w = frames[0].shape[:2]
        labels = [np.zeros((h, w), np.uint16) for _ in range(n)]
        confs = [np.zeros((h, w), np.float32) for _ in range(n)]

        masks = self._seed(frames[0])
        obj_ids = list(range(1, len(masks) + 1))   # fresh local ids per chunk
        if not masks:
            return list(zip(labels, confs))

        state_dev = "cpu" if self.offload else self.device
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            inference_state_device=state_dev,
            video_storage_device=state_dev,
            dtype=torch.float32,
        )
        self.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=obj_ids,
            input_masks=[m.astype(np.float32) for m in masks],
        )
        with torch.inference_mode():
            self.model(inference_session=session, frame_idx=0)
            for out in self.model.propagate_in_video_iterator(session):
                logits = self.processor.post_process_masks(
                    [out.pred_masks], original_sizes=[[h, w]], binarize=False,
                )[0][:, 0].float().cpu().numpy()          # (num_obj, h, w) logits
                res = logits > 0
                ids = [int(x) for x in list(out.object_ids)]
                lab, cf = labels[out.frame_idx], confs[out.frame_idx]
                for k in np.argsort([-int(m.sum()) for m in res]):   # largest first: finer wins
                    lab[res[k]] = ids[k]
                    cf[res[k]] = 1.0 / (1.0 + np.exp(-logits[k][res[k]]))
        return list(zip(labels, confs))
