"""Per-frame geometry from a feed-forward reconstruction model.

A geometry backend turns an ordered RGB stream into, for each frame, a metric
depth map, pinhole intrinsics, and a camera-to-world pose. VGGT is the only real
backend (see :mod:`services.reconstruction.vggt`); the :class:`GeometryBackend`
protocol exists purely as a test seam so a lightweight fake can stand in for VGGT
without torch or a GPU.

Output conventions follow VGGT/R3: the model emits *camera-from-world* (w2c)
extrinsics in OpenCV convention, which we invert to *camera-to-world* (``c2w``);
``depth`` is a per-pixel range map; ``conf`` is the model's per-pixel score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class FrameGeometry:
    """Depth, intrinsics, and pose for a single RGB frame.

    Coordinates are in the reconstruction's own (arbitrary, up-to-scale) world
    frame until :mod:`herald.scene.refine.register` aligns them to the site ENU
    frame.
    """

    index: int
    K: np.ndarray  # (3, 3) pinhole intrinsics, pixels
    c2w: np.ndarray  # (4, 4) camera-to-world, OpenCV convention
    depth: np.ndarray  # (H, W) float32; <= 0 or NaN marks invalid pixels
    conf: np.ndarray | None = None  # (H, W) per-pixel confidence, if available
    rgb_ref: str | None = None  # path to source RGB frame (render-only)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.c2w = np.asarray(self.c2w, dtype=np.float64).reshape(4, 4)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {self.depth.shape}")
        if self.conf is not None:
            self.conf = np.asarray(self.conf, dtype=np.float32)
            if self.conf.shape != self.depth.shape:
                raise ValueError("conf must match depth shape")

    @property
    def hw(self) -> tuple[int, int]:
        """Image (height, width)."""
        return (int(self.depth.shape[0]), int(self.depth.shape[1]))

    @property
    def cam_center(self) -> np.ndarray:
        """Camera origin in the reconstruction world frame, shape (3,)."""
        return self.c2w[:3, 3].astype(np.float64)


@runtime_checkable
class GeometryBackend(Protocol):
    """Feed-forward depth + pose estimator over an ordered set of frames.

    Implemented by :class:`~services.reconstruction.vggt.VGGTBackend` (real) and
    by a fake in the test suite. ``frames`` is whatever the stream loader yields
    (image paths for VGGT); a backend is free to ignore it (the fake does).
    """

    model_id: str

    def infer(self, frames: Sequence[object]) -> list[FrameGeometry]: ...
