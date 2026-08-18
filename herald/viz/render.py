from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from herald.scene.common.geometry import FrameGeometry
from herald.viz.frames import colorize_labels

LoadRGB = Callable[[int], np.ndarray]

_ZUP_FROM_NED = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

def unproject(
    geometry: FrameGeometry,
    rgb: np.ndarray | None = None,
    *,
    pixel_stride: int = 4,
    max_depth: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray | None]:
    d = geometry.depth[::pixel_stride, ::pixel_stride]
    h, w = geometry.depth.shape
    vs, us = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    d, us, vs = d[valid], us[valid], vs[valid]

    fx, fy = geometry.K[0, 0], geometry.K[1, 1]
    cx, cy = geometry.K[0, 2], geometry.K[1, 2]
    cam = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], axis=1)
    world = cam @ geometry.c2w[:3, :3].T + geometry.c2w[:3, 3]

    colors = None
    if rgb is not None:
        colors = rgb[vs, us]
    return world.astype(np.float32), colors

def accumulate_cloud(
    geo,
    load_rgb: LoadRGB,
    indices: Sequence[int],
    *,
    pixel_stride: int = 8,
    max_depth: float = 60.0,
    voxel: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    pts, cols = [], []
    for i in indices:
        p, c = unproject(geo.geometry(i), load_rgb(i), pixel_stride=pixel_stride, max_depth=max_depth)
        pts.append(p)
        cols.append(c)
    if not pts:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    keys = np.floor(pts / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return pts[keep], cols[keep]

def log_rerun(
    out_rrd: Path,
    geo,
    seg,
    load_rgb: LoadRGB,
    indices: Sequence[int],
    *,
    cloud: tuple[np.ndarray, np.ndarray],
    max_depth: float = 60.0,
) -> None:
    import rerun as rr

    rr.init("herald.viz")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    cpts, ccols = cloud
    rr.log("world/cloud", rr.Points3D(cpts @ _ZUP_FROM_NED, colors=ccols, radii=0.03), static=True)

    centers: list[np.ndarray] = []
    for i in indices:
        g = geo.geometry(i)
        h, w = g.hw
        rr.set_time("frame", sequence=i)
        centers.append(_ZUP_FROM_NED @ g.cam_center)
        rr.log("world/camera", rr.Transform3D(
            translation=_ZUP_FROM_NED @ g.c2w[:3, 3], mat3x3=_ZUP_FROM_NED @ g.c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(
            image_from_camera=g.K, resolution=[w, h], camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/image/rgb", rr.Image(load_rgb(i)))
        depth = g.depth.copy()
        depth[depth >= max_depth] = 0.0
        rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1.0))
        rr.log("world/camera/image/seg", rr.SegmentationImage(seg.labels(i).astype(np.uint16)))
        if len(centers) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers)], colors=[(255, 0, 255)]))
    out_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_rrd))

def render_stream(
    geo,
    seg,
    load_rgb: LoadRGB,
    *,
    out_rrd: Path,
    indices: Sequence[int] | None = None,
    cloud_frame_stride: int = 10,
    cloud_pixel_stride: int = 8,
    max_depth: float = 60.0,
    voxel: float = 0.1,
) -> Path:
    if indices is None:
        indices = list(range(len(geo)))
    indices = list(indices)
    cloud_idx = indices[::cloud_frame_stride] or indices[:1]
    cloud = accumulate_cloud(
        geo, load_rgb, cloud_idx, pixel_stride=cloud_pixel_stride, max_depth=max_depth, voxel=voxel)
    log_rerun(out_rrd, geo, seg, load_rgb, indices, cloud=cloud, max_depth=max_depth)
    return out_rrd

def render_clouds(
    cloud_paths: Sequence[Path | str],
    out_rrd: Path | str,
    *,
    radii: float = 0.03,
) -> Path:
    import rerun as rr

    clouds = sorted((dict(np.load(p)) for p in cloud_paths), key=lambda c: int(c.get("frame", -1)))
    rr.init("herald.viz")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for cloud in clouds:
        rr.set_time("frame", sequence=max(int(cloud.get("frame", -1)), 0))
        rr.log("world/cloud", rr.Points3D(
            cloud["points"] @ _ZUP_FROM_NED, colors=colorize_labels(cloud["labels"]), radii=radii))
    out_rrd = Path(out_rrd)
    out_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_rrd))
    return out_rrd
