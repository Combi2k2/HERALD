"""Fuse posed depth + semantic label masks into a labeled point cloud.

Geometry can be a per-frame dict from VggtStream ({"K","c2w","depth","conf"})
or a dataset FrameGeometry (TartanGround GT); labels can come from Sam2Stream
or dataset GT seg, and sources may be mixed frame to frame. Every accepted
pixel casts a (voxel, label) vote and finalize() resolves each voxel by
majority, so disagreeing sources and per-frame mask noise are reconciled
instead of trusted blindly.

Low-confidence input is discarded before it can vote:
- pixels: non-finite/zero/far depth, depth discontinuities (utils.depth_edge,
  the "flying pixel" filter from vggt-omega), and a per-frame confidence
  percentile (VGGT conf has no absolute scale, so the threshold is relative);
- mask borders: label maps are eroded so points cannot bleed across objects;
- voxels/objects: finalize() drops voxels with few hits, labels seen in few
  frames, and small disconnected clusters (utils.filter_clusters).

The result is a dict of flat arrays — {"points": (N,3) f32, "labels": (N,) i64,
"hits": (N,) i64, "obj_labels"/"obj_frames": per-object i64, "frame": int} —
persisted with utils.save_cloud / utils.load_cloud.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from herald.scene.recon.utils import (
    depth_edge,
    erode_labels,
    filter_clusters,
    resize_nearest,
    unproject_labeled,
)

class Fuser:
    """Incrementally fuse (geometry, labels) frames into a voted voxel cloud."""

    def __init__(
        self,
        *,
        voxel: float = 0.1,
        stride: int = 2,
        max_depth: float = 60.0,
        conf_percentile: float = 0.0,
        edge_rtol: float = 0.0,
        erode: int = 1,
        ignore_ids: Sequence[int] = (),
    ) -> None:
        if voxel <= 0:
            raise ValueError("voxel must be > 0")
        self.voxel = voxel
        self.stride = max(1, stride)
        self.max_depth = max_depth
        self.conf_percentile = conf_percentile
        self.edge_rtol = edge_rtol
        self.erode = max(0, erode)
        self.ignore_ids = frozenset(int(i) for i in ignore_ids)
        # (voxel, label) vote table: keys (M,4) = ijk + label, point sums, hit counts
        self._keys = np.empty((0, 4), np.int64)
        self._sums = np.zeros((0, 3))
        self._counts = np.zeros(0, np.int64)
        self._label_frames: dict[int, int] = {}
        self.frames = 0

    def add(self, geo, labels) -> None:
        """Fuse one frame; `geo` is a VggtStream dict or a FrameGeometry."""
        get = geo.get if isinstance(geo, Mapping) else lambda k: getattr(geo, k, None)
        depth = np.asarray(get("depth"))
        lab = np.asarray(labels)
        if lab.shape != depth.shape:
            lab = resize_nearest(lab, depth.shape)
        if self.erode:
            lab = erode_labels(lab, self.erode)

        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        if self.edge_rtol > 0:
            valid &= ~depth_edge(depth, rtol=self.edge_rtol)
        conf = get("conf")
        if conf is not None and self.conf_percentile > 0 and np.any(valid):
            conf = np.asarray(conf)
            valid &= conf >= np.percentile(conf[valid], self.conf_percentile)

        points, point_labels = unproject_labeled(
            depth, get("K"), get("c2w"), lab, stride=self.stride, valid=valid)
        keep = point_labels >= 0
        for ignored in self.ignore_ids:
            keep &= point_labels != ignored
        points, point_labels = points[keep], point_labels[keep]

        keys = np.concatenate(
            [np.floor(points / self.voxel).astype(np.int64), point_labels[:, None]], axis=1)
        m = len(self._keys)
        uniq, inv = np.unique(np.concatenate([self._keys, keys]), axis=0, return_inverse=True)
        sums = np.zeros((len(uniq), 3))
        counts = np.zeros(len(uniq), np.int64)
        np.add.at(sums, inv[:m], self._sums)
        np.add.at(counts, inv[:m], self._counts)
        np.add.at(sums, inv[m:], points)
        np.add.at(counts, inv[m:], 1)
        self._keys, self._sums, self._counts = uniq, sums, counts
        for label in np.unique(point_labels):
            self._label_frames[int(label)] = self._label_frames.get(int(label), 0) + 1
        self.frames += 1

    def finalize(
        self,
        *,
        eps: float | None = None,
        min_cluster: int = 10,
        min_hits: int = 1,
        min_frames: int = 1,
        frame: int = -1,
    ) -> dict:
        eps = 2.0 * self.voxel if eps is None else eps
        keys, sums, counts = self._keys, self._sums, self._counts
        if len(keys):
            # majority vote: within each voxel keep the best-supported label
            order = np.lexsort((counts, keys[:, 2], keys[:, 1], keys[:, 0]))
            keys, sums, counts = keys[order], sums[order], counts[order]
            winner = np.r_[np.any(keys[1:, :3] != keys[:-1, :3], axis=1), True]
            keys, sums, counts = keys[winner], sums[winner], counts[winner]

        pts, labs, hits, obj_labels, obj_frames = [], [], [], [], []
        for label in (np.unique(keys[:, 3]) if len(keys) else []):
            seen = self._label_frames.get(int(label), 0)
            if seen < min_frames:
                continue
            sel = keys[:, 3] == label
            points = sums[sel] / counts[sel, None]
            cnt = counts[sel]
            keep = cnt >= min_hits
            points, cnt = points[keep], cnt[keep]
            if len(points) and eps > 0:
                keep = filter_clusters(points, eps=eps, min_cluster=min_cluster)
                points, cnt = points[keep], cnt[keep]
            if len(points) == 0:
                continue
            pts.append(points)
            labs.append(np.full(len(points), label, np.int64))
            hits.append(cnt)
            obj_labels.append(int(label))
            obj_frames.append(seen)
        return {
            "points": np.concatenate(pts).astype(np.float32) if pts else np.empty((0, 3), np.float32),
            "labels": np.concatenate(labs) if labs else np.empty(0, np.int64),
            "hits": np.concatenate(hits) if hits else np.empty(0, np.int64),
            "obj_labels": np.array(obj_labels, np.int64),
            "obj_frames": np.array(obj_frames, np.int64),
            "frame": int(frame),
        }
