"""Per-frame object proposals (perception seam).

M1 ships a deterministic, dependency-free :class:`StubObjectSource` so the refine
pipeline runs end-to-end without a GPU, SAM, or CLIP. The real open-vocabulary
detector + segmenter (and CLIP/VLM features) swap in at M2 behind the same
:class:`ObjectSource` protocol. Tests typically inject their own fake source for
precise assertions, exactly as ``FakeBackend`` stands in for VGGT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from services.reconstruction.base import FrameGeometry


@dataclass
class Detection:
    """A single 2D object proposal in one frame."""

    label: str
    score: float = 1.0
    bbox: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 (pixels)
    mask: np.ndarray | None = None  # (H, W) bool, optional

    def pixel_mask(self, h: int, w: int) -> np.ndarray:
        """Boolean (H, W) support: explicit mask, else bbox rect, else full frame."""
        if self.mask is not None:
            return np.asarray(self.mask, dtype=bool)
        m = np.zeros((h, w), dtype=bool)
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            x0, x1 = sorted((max(0, int(x0)), min(w, int(x1))))
            y0, y1 = sorted((max(0, int(y0)), min(h, int(y1))))
            m[y0:y1, x0:x1] = True
        else:
            m[:] = True
        return m


@runtime_checkable
class ObjectSource(Protocol):
    """Detect object proposals in a frame (optionally using its geometry)."""

    name: str

    def detect(
        self, frame: np.ndarray, geometry: FrameGeometry | None = None
    ) -> list[Detection]: ...


class StubObjectSource:
    """Deterministic grid proposals over pixels that carry valid depth.

    Placeholder only: it tiles each frame into a ``grid`` of cells and emits one
    proposal per cell whose center has finite, positive depth. This produces
    stable, distinct 3D blobs that associate across frames so the rest of the
    pipeline can be exercised, but it does no real recognition.
    """

    name = "stub-grid"

    def __init__(self, grid: tuple[int, int] = (2, 2), label_prefix: str = "object") -> None:
        self.grid = grid
        self.label_prefix = label_prefix

    def detect(
        self, frame: np.ndarray, geometry: FrameGeometry | None = None
    ) -> list[Detection]:
        if geometry is not None:
            h, w = geometry.hw
        else:
            arr = np.asarray(frame)
            h, w = int(arr.shape[0]), int(arr.shape[1])
        rows, cols = self.grid
        out: list[Detection] = []
        for r in range(rows):
            for c in range(cols):
                y0, y1 = h * r // rows, h * (r + 1) // rows
                x0, x1 = w * c // cols, w * (c + 1) // cols
                if geometry is not None:
                    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                    d = geometry.depth[cy, cx]
                    if not (np.isfinite(d) and d > 0):
                        continue
                out.append(
                    Detection(
                        label=f"{self.label_prefix}_r{r}c{c}",
                        score=1.0,
                        bbox=(x0, y0, x1, y1),
                    )
                )
        return out
