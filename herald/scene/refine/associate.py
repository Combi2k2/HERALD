"""Multi-view association: per-frame detections -> 3D object instances.

Each detection is lifted to 3D using the frame's depth, then matched to an
existing instance by nearest centroid within a distance gate (optionally
label-constrained). Matched detections accumulate points so each instance grows
a 3D support set from which a centroid and an axis-aligned bounding box are read.
This is the geometry-only association used in M1; appearance/embedding-based
association is an M2 refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from services.reconstruction.base import FrameGeometry
from herald.scene.refine.perceive import Detection


@dataclass
class ObjectInstance:
    """A multi-view-fused object: label, 3D support points, and observations."""

    id: int
    label: str
    points: np.ndarray  # (M, 3) in the reconstruction frame
    observation_count: int = 1
    score: float = 1.0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def size(self) -> np.ndarray:
        lo, hi = self.aabb
        return hi - lo

    def add(self, pts: np.ndarray, score: float, *, max_points: int = 20000) -> None:
        self.points = np.concatenate([self.points, np.asarray(pts, dtype=np.float64)], axis=0)
        self.observation_count += 1
        self.score = max(self.score, float(score))
        if len(self.points) > max_points:
            idx = np.linspace(0, len(self.points) - 1, max_points).astype(int)
            self.points = self.points[idx]


def _unproject_mask(
    fg: FrameGeometry,
    mask: np.ndarray,
    *,
    conf_thresh: float = 0.0,
    max_points: int = 2000,
) -> np.ndarray | None:
    """Unproject the valid depth pixels under ``mask`` into the recon world frame."""
    depth = fg.depth
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if fg.conf is not None and conf_thresh > 0:
        valid &= fg.conf >= conf_thresh
    if not np.any(valid):
        return None
    vs, us = np.nonzero(valid)  # rows (y), cols (x)
    z = depth[valid].astype(np.float64)
    fx, fy = fg.K[0, 0], fg.K[1, 1]
    cx, cy = fg.K[0, 2], fg.K[1, 2]
    cam = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=1)
    world = cam @ fg.c2w[:3, :3].T + fg.c2w[:3, 3]
    if len(world) > max_points:
        idx = np.linspace(0, len(world) - 1, max_points).astype(int)
        world = world[idx]
    return world


def associate_objects(
    geometries: Sequence[FrameGeometry],
    detections_per_frame: Sequence[Sequence[Detection]],
    *,
    assoc_dist: float = 0.5,
    conf_thresh: float = 0.0,
    max_points_per_det: int = 2000,
    max_points_per_instance: int = 20000,
    min_observations: int = 1,
    same_label_only: bool = False,
) -> list[ObjectInstance]:
    """Fuse per-frame detections into 3D :class:`ObjectInstance` tracks."""
    instances: list[ObjectInstance] = []
    for fg, dets in zip(geometries, detections_per_frame):
        h, w = fg.hw
        for det in dets:
            pts = _unproject_mask(
                fg, det.pixel_mask(h, w),
                conf_thresh=conf_thresh, max_points=max_points_per_det,
            )
            if pts is None or len(pts) == 0:
                continue
            centroid = pts.mean(axis=0)
            best: ObjectInstance | None = None
            best_d = assoc_dist
            for inst in instances:
                if same_label_only and inst.label != det.label:
                    continue
                d = float(np.linalg.norm(inst.centroid - centroid))
                if d < best_d:
                    best_d = d
                    best = inst
            if best is None:
                instances.append(
                    ObjectInstance(
                        id=len(instances), label=det.label,
                        points=pts, observation_count=1, score=float(det.score),
                    )
                )
            else:
                best.add(pts, float(det.score), max_points=max_points_per_instance)
    return [i for i in instances if i.observation_count >= min_observations]
