"""Geometry helpers, scene-cloud accumulation, and cloud I/O shared by recon."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_OFF = 1 << 20  # voxel-index bias so floor-divided coords stay non-negative before packing


def quat_to_R(q) -> np.ndarray:
    """Unit quaternion (x,y,z,w) -> 3x3 rotation matrix (object -> world)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


def obb_inter_vol(ci, hi, Ri, cj, hj, Rj) -> float:
    """Exact intersection volume of two gravity-aligned (yaw-only) OBBs: the boxes
    share the vertical (z) axis, so volume = (XY footprint intersection area) x (z
    interval overlap). Footprints are rotated rectangles; Shapely clips them exactly
    -- no Monte-Carlo. `c*`=center(3), `h*`=half-size(3), `R*`=rotation (object->world)."""
    from shapely.geometry import Polygon

    dz = min(ci[2] + hi[2], cj[2] + hj[2]) - max(ci[2] - hi[2], cj[2] - hj[2])
    if dz <= 0:
        return 0.0

    def rect(c, h, R):
        yaw = np.arctan2(R[1, 0], R[0, 0])                       # footprint rotation about z
        cs, sn = np.cos(yaw), np.sin(yaw)
        corners = np.array([[h[0], h[1]], [h[0], -h[1]], [-h[0], -h[1]], [-h[0], h[1]]], np.float64)
        return Polygon(corners @ np.array([[cs, sn], [-sn, cs]]) + np.asarray(c[:2], np.float64))

    return rect(ci, hi, Ri).intersection(rect(cj, hj, Rj)).area * float(dz)


def obb_vol(h) -> float:
    """Volume of an OBB from its half-size (3,): product of the full extents."""
    h = np.asarray(h, np.float64)
    return float(8.0 * h[0] * h[1] * h[2])


def obb_iou(ci, hi, Ri, cj, hj, Rj) -> float:
    """3D IoU of two gravity-aligned (yaw-only) OBBs = inter / (vi + vj - inter), reusing
    obb_inter_vol. A scale-free 'mergeable' metric (no absolute-distance threshold)."""
    inter = obb_inter_vol(ci, hi, Ri, cj, hj, Rj)
    if inter <= 0:
        return 0.0
    union = obb_vol(hi) + obb_vol(hj) - inter
    return float(inter / union) if union > 1e-12 else 0.0


def obb_ios(ci, hi, Ri, cj, hj, Rj) -> float:
    """Intersection-over-smaller of two gravity-aligned OBBs = inter / min(vol_i, vol_j).
    Unlike IoU it is robust to size mismatch: a small box fully nested in a much larger
    one scores ~1, so a duplicate whose two OBBs disagree on extent still reads as a
    strong match (IoU would drop it). Use as the cross-session 'mergeable' metric."""
    inter = obb_inter_vol(ci, hi, Ri, cj, hj, Rj)
    if inter <= 0:
        return 0.0
    vmin = min(obb_vol(hi), obb_vol(hj))
    return float(inter / vmin) if vmin > 1e-12 else 0.0


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """(S,3,4) world-to-camera extrinsics -> (S,4,4) camera-to-world poses."""
    rt = w2c[:, :3, :3].transpose(0, 2, 1)  # R^T per frame
    c2w = np.tile(np.eye(4), (len(w2c), 1, 1))
    c2w[:, :3, :3] = rt
    c2w[:, :3, 3] = -np.einsum("sij,sj->si", rt, w2c[:, :3, 3])
    return c2w


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


def unproject_rgb(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    rgb: np.ndarray,
    stride: int = 1,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift valid depth pixels to world points + their RGB (same pinhole/stride
    convention as unproject_labeled)."""
    h, w = depth.shape
    d = depth[::stride, ::stride]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid)[::stride, ::stride]
    if not np.any(ok):
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    z = d[ok].astype(np.float64)
    u = us[ok].astype(np.float64)
    v = vs[ok].astype(np.float64)
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    c2w = np.asarray(c2w)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world, rgb[::stride, ::stride][ok]


class SceneCloud:
    """Voxel-downsampled colored cloud of the whole scene (background included).
    One running centroid + mean RGB per occupied voxel, so memory is bounded by
    explored volume, not frame count."""

    def __init__(self, voxel: float = 0.1):
        self.voxel = voxel
        self._acc: dict[int, list] = {}   # code -> [sx,sy,sz, sr,sg,sb, n]

    def add(self, pts: np.ndarray, cols: np.ndarray) -> None:
        if len(pts) == 0:
            return
        k = np.floor(pts / self.voxel).astype(np.int64) + _OFF
        codes = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
        uniq, inv = np.unique(codes, return_inverse=True)
        ps = np.zeros((len(uniq), 3)); np.add.at(ps, inv, pts)
        cs = np.zeros((len(uniq), 3)); np.add.at(cs, inv, cols.astype(np.float64))
        cnt = np.bincount(inv).astype(np.float64)
        for j, c in enumerate(uniq.tolist()):
            e = self._acc.get(c)
            if e is None:
                self._acc[c] = [ps[j, 0], ps[j, 1], ps[j, 2], cs[j, 0], cs[j, 1], cs[j, 2], cnt[j]]
            else:
                e[0] += ps[j, 0]; e[1] += ps[j, 1]; e[2] += ps[j, 2]
                e[3] += cs[j, 0]; e[4] += cs[j, 1]; e[5] += cs[j, 2]; e[6] += cnt[j]

    def cloud(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._acc:
            return np.empty((0, 3)), np.empty((0, 3), np.uint8)
        v = np.asarray(list(self._acc.values()))
        return v[:, :3] / v[:, 6:7], (v[:, 3:6] / v[:, 6:7]).astype(np.uint8)


_DBSCAN = None  # cached (name, run) backend: cuML on GPU if present, else sklearn


def _dbscan_backend():
    """Resolve the DBSCAN backend once: RAPIDS cuML (GPU) when importable,
    otherwise sklearn (CPU). Returns (name, run(points, eps, min_samples))."""
    global _DBSCAN
    if _DBSCAN is not None:
        return _DBSCAN
    try:
        from cuml.cluster import DBSCAN as _cuDBSCAN

        def run(pts, eps, min_samples):
            comp = _cuDBSCAN(
                eps=eps, min_samples=min_samples, output_type="numpy"
            ).fit_predict(np.ascontiguousarray(pts, dtype=np.float32))
            return np.asarray(comp)

        _DBSCAN = ("cuml", run)
    except Exception:
        from sklearn.cluster import DBSCAN as _skDBSCAN

        def run(pts, eps, min_samples):
            return _skDBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)

        _DBSCAN = ("sklearn", run)
    return _DBSCAN


def filter_clusters(
    points: np.ndarray,
    *,
    eps: float,
    min_cluster: int,
    min_samples: int = 1,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Keep-mask for points in DBSCAN clusters of at least min_cluster points.

    min_samples=1 makes every point a core point, so DBSCAN reduces to plain
    eps-connectivity (no noise); raise it to break low-density bridges and drop
    sparse points as noise. Runs on GPU via cuML when available, else sklearn.
    Pass `labels` (one per point) to cluster all points in a single backend call:
    points get a 4th coordinate offset per label far enough apart that clusters
    can never bridge two labels (identical to clustering each label separately)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if min_cluster <= 1 and min_samples <= 1:
        return np.ones(n, dtype=bool)

    if labels is None:
        data = np.ascontiguousarray(points, dtype=np.float32)
    else:
        _, dense = np.unique(np.asarray(labels), return_inverse=True)
        sep = (dense.astype(np.float32) * (2.0 * eps))[:, None]
        data = np.ascontiguousarray(np.concatenate([points, sep], axis=1), dtype=np.float32)

    _, run = _dbscan_backend()
    comp = run(data, eps, min_samples)
    keep = comp >= 0  # DBSCAN marks noise as -1
    if keep.any():
        sizes = np.bincount(comp[keep])
        keep[keep] = sizes[comp[keep]] >= min_cluster
    return keep


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


# --------------------------------------------------------------------------- crop-ref resolver
# Render-only: turn a SceneObject's per-session crop references (frame_idx + 2D box) back into
# thumbnail images. Nothing in the processing pipeline calls this -- it reads source frames off
# disk on demand. `source` is a path: a frames folder (sorted images, indexed by frame_id) now;
# a video file later (swap `_frame_files`/`load_frame`).

import functools


@functools.lru_cache(maxsize=16)
def _frame_files(source: str) -> tuple:
    p = Path(source)
    files = sorted(p.glob("*.png")) or sorted(p.glob("*.jpg"))
    return tuple(str(f) for f in files)


def load_frame(source: str, frame_id: int) -> np.ndarray:
    """RGB uint8 frame `frame_id` from a source path (frames folder; `frame_id` indexes the
    sorted images, matching the recon loop's frame index)."""
    from PIL import Image
    return np.asarray(Image.open(_frame_files(str(source))[int(frame_id)]).convert("RGB"))


def crop_ref(source: str, ref, size: int = 112) -> np.ndarray:
    """Resolve one (frame_id, box_xyxy) reference to a `size`x`size` uint8 RGB thumbnail."""
    import cv2
    frame_id, box = ref
    img = load_frame(source, frame_id)
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    patch = img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def object_crops(obj, sources: dict | None, size: int = 112, max_n: int = 4) -> list:
    """All crop thumbnails for a SceneObject, resolved across its sessions -> list[uint8 img].
    `sources` = SceneMap.meta['sources'] (session_id -> path). Missing sources / broken refs
    are skipped, so this never raises during rendering."""
    out: list = []
    for sid, rec in obj.sessions.items():
        src = sources.get(sid) if sources else None
        if not src:
            continue
        for ref in rec.get("crops", []):
            try:
                out.append(crop_ref(src, ref, size))
            except Exception:
                continue
            if len(out) >= max_n:
                return out
    return out
