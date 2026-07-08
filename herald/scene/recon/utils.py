"""Small array converters, geometry helpers, and cloud I/O shared by the recon submodules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

def to_bool(mask) -> np.ndarray:
    """Torch tensor or array-like -> boolean numpy mask."""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    return np.asarray(mask).astype(bool)

def as_rgb(frame) -> np.ndarray:
    """RGB ndarray passthrough, or image path -> HxWx3 uint8 array."""
    if isinstance(frame, np.ndarray):
        return frame
    return np.asarray(Image.open(frame).convert("RGB"))

def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0

def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """(S,3,4) world-to-camera extrinsics -> (S,4,4) camera-to-world poses."""
    rt = w2c[:, :3, :3].transpose(0, 2, 1)  # R^T per frame
    c2w = np.tile(np.eye(4), (len(w2c), 1, 1))
    c2w[:, :3, :3] = rt
    c2w[:, :3, 3] = -np.einsum("sij,sj->si", rt, w2c[:, :3, 3])
    return c2w

def resize_nearest(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an integer label map to (H, W)."""
    h, w = shape
    img = Image.fromarray(np.asarray(labels).astype(np.int32), mode="I")
    return np.asarray(img.resize((w, h), Image.NEAREST)).astype(np.int64)

def erode_labels(labels: np.ndarray, iterations: int = 1, fill: int = -1) -> np.ndarray:
    """Shrink each label region so border pixels (mixed-object) become `fill`."""
    lab = np.asarray(labels).astype(np.int64)
    for _ in range(max(0, iterations)):
        p = np.pad(lab, 1, mode="edge")
        c = p[1:-1, 1:-1]
        keep = (
            (c == p[:-2, 1:-1]) & (c == p[2:, 1:-1])
            & (c == p[1:-1, :-2]) & (c == p[1:-1, 2:])
        )
        lab = np.where(keep, lab, fill)
    return lab

def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    """Mask of pixels at depth discontinuities ("flying pixels" once unprojected)."""
    pad = kernel_size // 2
    p = np.pad(depth, pad, mode="edge")
    h, w = depth.shape
    dmax = np.full_like(depth, -np.inf)
    dmin = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            dmax = np.maximum(dmax, p[y:y + h, x:x + w])
            dmin = np.minimum(dmin, p[y:y + h, x:x + w])
    return (dmax - dmin) / np.maximum(np.abs(depth), 1e-6) > rtol

def unproject_labeled(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    labels: np.ndarray,
    *,
    stride: int = 1,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift labeled depth pixels to world points -> ((N,3) points, (N,) labels)."""
    depth, labels = np.asarray(depth), np.asarray(labels)
    if labels.shape != depth.shape:
        raise ValueError(f"labels {labels.shape} must match depth {depth.shape}")

    h, w = depth.shape
    d = depth[::stride, ::stride]
    lab = labels[::stride, ::stride]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]

    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid)[::stride, ::stride]
    if not np.any(ok):
        return np.empty((0, 3)), np.empty((0,), np.int64)

    z = d[ok].astype(np.float64)
    u = us[ok].astype(np.float64)
    v = vs[ok].astype(np.float64)
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    c2w = np.asarray(c2w)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world, lab[ok].astype(np.int64)

def filter_clusters(points: np.ndarray, *, eps: float, min_cluster: int) -> np.ndarray:
    """Keep-mask for points in connected clusters of at least min_cluster points."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if min_cluster <= 1:
        return np.ones(n, dtype=bool)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    return sizes[comp] >= min_cluster

def save_cloud(cloud: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cloud)

def load_cloud(path: Path | str) -> dict:
    data = np.load(Path(path))
    cloud = {k: data[k] for k in data.files}
    cloud["frame"] = int(cloud.get("frame", -1))
    return cloud

def write_ply(path: Path | str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]
    has_color = colors is not None and len(colors) == n
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    header.append("end_header\n")

    buf = np.empty(n, dtype=np.dtype(fields))
    buf["x"], buf["y"], buf["z"] = points[:, 0], points[:, 1], points[:, 2]
    if has_color:
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        buf["red"], buf["green"], buf["blue"] = c[:, 0], c[:, 1], c[:, 2]

    with path.open("wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        f.write(buf.tobytes())
