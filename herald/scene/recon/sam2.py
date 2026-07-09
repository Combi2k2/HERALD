"""RGB frames -> framewise instance label masks via SAM2 video propagation."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor, pipeline

from herald.scene.recon.utils import as_rgb, iou, to_bool

DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"

class Sam2Stream:
    """Push RGB frames, receive (uint16 label map, float32 conf map) pairs
    back one chunk at a time.

    Frames are buffered into chunks; each chunk is seeded with automatic masks
    on its first frame and propagated with SAM2 video (run_chunk). Object
    identity is carried across chunks by IoU between new seeds and the
    previous chunk's final-frame masks. run_stream/run_video wrap push/finish
    for iterables and whole sequences of RGB arrays or image paths.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        chunk: int = 32,
        points_per_side: int = 16,
        points_per_batch: int = 32,
        min_area: int = 200,
        carry_iou: float = 0.3,
        offload: bool = True,
    ) -> None:
        self.model = Sam2VideoModel.from_pretrained(model_name).to(device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(model_name)
        self.maskgen = pipeline("mask-generation", model=model_name, device=device)
        self.chunk = max(1, chunk)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.min_area = min_area
        self.carry_iou = carry_iou
        self.offload = offload
        self._buffer: list[np.ndarray] = []
        self._prev: dict[int, np.ndarray] = {}
        self._next_id = 1

    def _seed(self, image) -> list[np.ndarray]:
        out = self.maskgen(
            image,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.9,
        )
        masks = [m for m in map(to_bool, out["masks"]) if int(m.sum()) >= self.min_area]
        masks.sort(key=lambda m: int(m.sum()), reverse=True)
        return masks

    def _seed_ids(self, seeds: list[np.ndarray]) -> tuple[list[int], list[np.ndarray]]:
        obj_ids: list[int] = []
        masks: list[np.ndarray] = []
        for mask in seeds:
            gid, best = None, self.carry_iou
            for pid, pmask in self._prev.items():
                score = iou(mask, pmask)
                if score > best:
                    best, gid = score, pid
            if gid is None:
                gid, self._next_id = self._next_id, self._next_id + 1
            if gid in obj_ids:
                continue
            obj_ids.append(gid)
            masks.append(mask)
        return obj_ids, masks

    def run_chunk(self, frames: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
        """Seed + propagate one chunk (RGB arrays or paths) in a single pass.

        Returns (label, conf) pairs — conf being sigmoid(mask logit) of the
        winning object per pixel (0 on background). Logits are relative
        confidence, not calibrated probabilities.
        """
        frames = [as_rgb(f) for f in frames]
        n = len(frames)
        h, w = frames[0].shape[:2]
        labels = [np.zeros((h, w), dtype=np.uint16) for _ in range(n)]
        confs = [np.zeros((h, w), dtype=np.float32) for _ in range(n)]

        obj_ids, masks = self._seed_ids(self._seed(Image.fromarray(frames[0])))
        if not masks:
            self._prev = {}
            return list(zip(labels, confs))

        state_device = "cpu" if self.offload else self.model.device
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.model.device,
            inference_state_device=state_device,
            video_storage_device=state_device,
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
                    [out.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=False,
                )[0][:, 0].float().cpu().numpy()
                res = logits > 0
                ids = list(getattr(out, "object_ids", None) or session.obj_ids)
                lab = labels[out.frame_idx]
                cf = confs[out.frame_idx]
                for k in np.argsort([-int(m.sum()) for m in res]):
                    lab[res[k]] = ids[k]
                    cf[res[k]] = 1.0 / (1.0 + np.exp(-logits[k][res[k]]))
                if out.frame_idx == n - 1:
                    self._prev = {ids[k]: res[k] for k in range(len(ids))}
        return list(zip(labels, confs))

    def push(self, rgb: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        self._buffer.append(np.asarray(rgb))
        if len(self._buffer) >= self.chunk:
            frames, self._buffer = self._buffer, []
            return self.run_chunk(frames)
        return []

    def finish(self) -> list[tuple[np.ndarray, np.ndarray]]:
        frames, self._buffer = self._buffer, []
        return self.run_chunk(frames) if frames else []

    def run_stream(self, frames: Iterable) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Lazily yield one (label, conf) pair per frame (RGB array or path), a chunk at a time."""
        for f in frames:
            yield from self.push(as_rgb(f))
        yield from self.finish()

    def run_video(self, frames: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
        """Whole-sequence convenience: (label, conf) pairs for RGB arrays or image paths."""
        return list(self.run_stream(frames))
