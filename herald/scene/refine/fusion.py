"""Depth -> world point cloud fusion (render-only sidecar).

The fused cloud is used **only for visualization**. It is written to a ``.ply``
sidecar and referenced by URI from the refine metadata; it is never stored in the
scene graph schema. Implementation is pure NumPy (no open3d) so it runs in the
torch-free env and in tests: a small voxel-grid downsampler and a minimal binary
PLY writer cover what we need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from services.reconstruction.base import FrameGeometry


@dataclass
class PointCloud:
    """A render-only fused point cloud. ``points`` is (N, 3), ``colors`` (N, 3) uint8."""

    points: np.ndarray
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(min_xyz, max_xyz) over all points; zeros for an empty cloud."""
        if len(self) == 0:
            return np.zeros(3), np.zeros(3)
        return self.points.min(axis=0), self.points.max(axis=0)

    def write_ply(self, path: Path | str) -> None:
        write_ply(path, self.points, self.colors)


def unproject(
    fg: FrameGeometry,
    rgb: np.ndarray | None = None,
    *,
    conf_thresh: float = 0.0,
    stride: int = 1,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a frame's valid depth pixels into the recon world frame.

    Returns ``(points (M, 3), colors (M, 3) uint8 | None)``. Uses the pinhole
    model with VGGT's z-depth convention: ``X_c = ((u-cx)/fx, (v-cy)/fy, 1) * d``,
    then ``X_w = R_c2w @ X_c + t_c2w``.
    """
    h, w = fg.hw
    depth = fg.depth[::stride, ::stride]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]

    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    if fg.conf is not None and conf_thresh > 0:
        valid &= fg.conf[::stride, ::stride] >= conf_thresh
    if not np.any(valid):
        return np.empty((0, 3)), (None if rgb is None else np.empty((0, 3), np.uint8))

    fx, fy = fg.K[0, 0], fg.K[1, 1]
    cx, cy = fg.K[0, 2], fg.K[1, 2]
    z = depth[valid].astype(np.float64)
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)

    r = fg.c2w[:3, :3]
    t = fg.c2w[:3, 3]
    world = cam @ r.T + t

    colors = None
    if rgb is not None:
        colors = np.asarray(rgb)[::stride, ::stride][valid][:, :3].astype(np.uint8)
    return world, colors


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    voxel: float = 0.05,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average points (and colors) that fall in the same ``voxel``-sized cell."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((counts.shape[0], 3))
    np.add.at(sums, inv, points)
    centroids = sums / counts[:, None]
    if colors is None:
        return centroids, None
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    csum = np.zeros((counts.shape[0], 3))
    np.add.at(csum, inv, colors)
    return centroids, (csum / counts[:, None]).round().astype(np.uint8)


def fuse_geometries(
    geometries: list[FrameGeometry],
    rgbs: list[np.ndarray] | None = None,
    *,
    conf_thresh: float = 0.0,
    stride: int = 4,
    voxel: float = 0.05,
    max_depth: float | None = None,
) -> PointCloud:
    """Unproject every frame, concatenate, and voxel-downsample into one cloud."""
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    have_color = rgbs is not None
    for i, fg in enumerate(geometries):
        rgb = rgbs[i] if rgbs is not None and i < len(rgbs) else None
        p, c = unproject(
            fg, rgb, conf_thresh=conf_thresh, stride=stride, max_depth=max_depth
        )
        if len(p):
            pts.append(p)
            if have_color and c is not None:
                cols.append(c)
    if not pts:
        return PointCloud(np.empty((0, 3)), None)
    points = np.concatenate(pts, axis=0)
    colors = np.concatenate(cols, axis=0) if have_color and cols else None
    points, colors = voxel_downsample(points, colors, voxel=voxel)
    return PointCloud(points, colors)


def write_ply(path: Path | str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a binary-little-endian PLY (xyz float32, optional rgb uchar)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]
    has_color = colors is not None and len(colors) == n
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header\n")

    if has_color:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        buf = np.empty(n, dtype=dtype)
        buf["x"], buf["y"], buf["z"] = points[:, 0], points[:, 1], points[:, 2]
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        buf["red"], buf["green"], buf["blue"] = c[:, 0], c[:, 1], c[:, 2]
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        buf = np.empty(n, dtype=dtype)
        buf["x"], buf["y"], buf["z"] = points[:, 0], points[:, 1], points[:, 2]

    with path.open("wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        f.write(buf.tobytes())
