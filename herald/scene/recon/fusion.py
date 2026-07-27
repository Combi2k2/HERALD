"""Fuse posed depth + per-frame instance masks into a labeled point cloud.

Geometry can be a per-frame dict from VggtStream ({"K","c2w","depth","conf"})
or a dataset FrameGeometry (TartanGround GT); masks are per-frame instance
masks from Sam2Segmenter (no persistent ids — identity is resolved here).

Objects are class-agnostic and identified by *appearance + occupancy*: each
mask is cropped from the RGB frame and embedded with SigLIP
(embed.SiglipEmbedder), then associated to an existing object when it is both
semantically similar (cosine >= `min_csim`) and geometrically overlapping — a
fraction >= `min_overlap` of the mask's voxels fall on (within `dilate` voxels
of) the object's own voxel cloud. The embedding gate keeps distinct kinds of
thing apart; the occupancy gate keeps two different instances of the same kind
(two chairs) from merging. A cheap point-to-AABB prefilter (`radius`) limits
the overlap test to nearby objects. A matched object updates its EMA embedding
and grows its voxel cloud; an unmatched mask starts a new object.

Per-frame association can only merge a mask into an object it *already*
overlaps, so one physical surface still fragments when it is revealed a strip
at a time (camera panning) or split into parts by SAM2 (sofa cushions, ceiling
tiles). flush() therefore runs a *merge pass* first: two existing objects are
unioned when they are geometrically adjacent (a fraction >= `merge_overlap` of
the smaller one's voxels lie within `merge_dilate` cells of the other) and
semantically similar (cosine >= `merge_csim`). `merge_dilate` > `dilate` so it
bridges the gap between fragments, and `merge_overlap` is small because two
co-planar surface fragments only meet along a seam; `merge_csim` is the guard
that stops it fusing e.g. ceiling into wall.

Each object carries a stable color id (`cid`) assigned at creation and kept
through merges, so a given physical object keeps one color across frames both
in the cloud and in the rendered per-frame masks (see `last_masks`/`last_cids`).

push() feeds one frame; flush() snapshots the (merged) objects as a cloud.

Low-confidence input is discarded before it contributes:
- pixels: non-finite/zero/far depth, depth discontinuities (utils.depth_edge,
  the "flying pixel" filter), a per-frame VGGT-confidence percentile, and an
  optional SAM2 mask-confidence threshold;
- masks: those smaller than `min_area` pixels are dropped before embedding;
- mask borders: the merged label map is eroded so points cannot bleed across
  objects;
- objects: flush() drops objects seen in few frames or with few voxels.

The result is a dict of flat arrays — {"points": (N,3) f32, "labels": (N,) i64
stable object color id, "hits": (N,) i64, "obj_labels"/"obj_frames": per-object
i64, "frame": int} — persisted with utils.save_cloud / utils.load_cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from herald.scene.recon.utils import (
    bbox_crop,
    depth_edge,
    erode_labels,
    resize_nearest,
    to_bool,
    unproject_labeled,
)


def _offsets(dilate: int) -> list[tuple[int, int, int]]:
    d = max(0, dilate)
    return [(a, b, c) for a in range(-d, d + 1)
            for b in range(-d, d + 1) for c in range(-d, d + 1)]


@dataclass
class _Object:
    """One accumulated object: an EMA appearance embedding, its own voxel cloud,
    a world-space AABB, and a stable color id. `vox` maps a voxel ijk tuple to
    [sum_x, sum_y, sum_z, count]."""

    emb: np.ndarray                       # (D,) L2-normalized EMA embedding
    bmin: np.ndarray                      # (3,) AABB min corner (world)
    bmax: np.ndarray                      # (3,) AABB max corner (world)
    cid: int                              # stable color id (kept through merges)
    vox: dict = field(default_factory=dict)
    frames: set = field(default_factory=set)


class Fuser:
    """Incrementally fuse (geometry, masks, rgb) frames into embedding-and-
    occupancy-associated objects, each a voted voxel cloud."""

    def __init__(
        self,
        embedder,
        *,
        voxel: float = 0.1,
        stride: int = 2,
        geo_conf_thresh: float = 0.0,
        max_depth: float = 60.0,
        edge_rtol: float = 0.0,
        sem_conf_thresh: float = 0.0,
        erode: int = 1,
        radius: float = 1.0,          # AABB prefilter slack: only test objects this near (m)
        min_overlap: float = 0.2,     # min fraction of mask voxels touching an object (spatial gate)
        dilate: int = 1,              # voxel dilation (in cells) when testing overlap
        min_area: int = 200,          # drop masks smaller than this before embedding
        bg_fade: float = 0.5,         # fade non-mask crop pixels toward white before embedding
                                      # (0 = raw bbox crop; keeps the embedding on the object,
                                      # not the surrounding room)
        min_csim: float = 0.5,        # min cosine similarity to associate (semantic gate)
        ema: float = 0.2,             # embedding update rate on a match
        merge_dilate: int = 2,        # voxel dilation for the object<->object merge pass
        merge_csim: float = 0.5,      # min cosine to merge two existing objects
        merge_overlap: float = 0.25,  # min contact (frac of smaller obj's voxels) to merge;
                                      # low values merge co-planar tiles but risk collapsing a
                                      # connected scene transitively, so default stays safe
    ) -> None:
        if voxel <= 0:
            raise ValueError("voxel must be > 0")
        self.embedder = embedder
        self.voxel = voxel
        self.stride = max(1, stride)
        self.geo_conf_thresh = geo_conf_thresh
        self.max_depth = max_depth
        self.edge_rtol = edge_rtol
        self.sem_conf_thresh = sem_conf_thresh
        self.erode = max(0, erode)
        self.radius = radius
        self.min_overlap = min_overlap
        self.min_area = min_area
        self.bg_fade = bg_fade
        self.min_csim = min_csim
        self.ema = ema
        self.merge_dilate = merge_dilate
        self.merge_csim = merge_csim
        self.merge_overlap = merge_overlap
        self._offsets = _offsets(dilate)
        self._merge_offsets = _offsets(merge_dilate)
        self._objs: list[_Object] = []
        # parallel arrays kept in sync with _objs for vectorized prefiltering
        self._emb = np.empty((0, embedder.dim), np.float32)
        self._bmin = np.empty((0, 3))
        self._bmax = np.empty((0, 3))
        self._next_cid = 1            # 0 is reserved for background in seg images
        self.frames = 0
        # per-frame association result, for rendering (masks + their color ids)
        self.last_masks: list[np.ndarray] = []
        self.last_cids: list[int] = []

    # -- association --------------------------------------------------------

    @staticmethod
    def _touch_frac(a: dict, b: dict, offsets) -> float:
        """Fraction of the smaller voxel set that lies within `offsets` of the
        larger set."""
        small, big = (a, b) if len(a) <= len(b) else (b, a)
        hit = sum(any((i + o0, j + o1, k + o2) in big for (o0, o1, o2) in offsets)
                  for (i, j, k) in small)
        return hit / len(small)

    def _overlap(self, keyset: list, vox: dict) -> float:
        """Fraction of the mask's voxels that fall within `dilate` cells of any
        of the object's occupied voxels."""
        off = self._offsets
        hit = sum(
            any((i + a, j + b, k + c) in vox for (a, b, c) in off)
            for (i, j, k) in keyset
        )
        return hit / len(keyset)

    def _match(self, emb: np.ndarray, centroid: np.ndarray, keyset: list) -> int | None:
        """Index of the best existing object that is near (AABB prefilter),
        semantically similar (cosine >= `min_csim`) and geometrically
        overlapping (>= `min_overlap`); highest overlap wins, else None."""
        if not self._objs:
            return None
        # point-to-AABB distance of the mask centroid to each object's box
        d = np.maximum(0.0, np.maximum(self._bmin - centroid, centroid - self._bmax))
        near = np.linalg.norm(d, axis=1) < self.radius
        if not near.any():
            return None
        best_j, best_ov = None, -1.0
        for j in np.where(near)[0]:
            if float(self._emb[j] @ emb) < self.min_csim:
                continue
            ov = self._overlap(keyset, self._objs[j].vox)
            if ov > best_ov:
                best_ov, best_j = ov, int(j)
        return best_j if best_ov >= self.min_overlap else None

    def _voxelize(self, points: np.ndarray):
        """(N,3) points -> (voxel keys as tuples, per-voxel point sums, counts)."""
        ijk = np.floor(points / self.voxel).astype(np.int64)
        keys, inv = np.unique(ijk, axis=0, return_inverse=True)
        sums = np.zeros((len(keys), 3))
        np.add.at(sums, inv, points)
        counts = np.bincount(inv, minlength=len(keys))
        return [tuple(r) for r in keys.tolist()], sums, counts

    def _accum(self, obj: _Object, keyset: list, sums: np.ndarray, counts: np.ndarray) -> None:
        for key, s, cnt in zip(keyset, sums, counts):
            cur = obj.vox.get(key)
            if cur is None:
                obj.vox[key] = np.array([s[0], s[1], s[2], float(cnt)])
            else:
                cur[:3] += s
                cur[3] += cnt

    def _update(self, k: int, emb: np.ndarray, pts: np.ndarray,
                keyset: list, sums: np.ndarray, counts: np.ndarray, frame: int) -> int:
        obj = self._objs[k]
        v = (1.0 - self.ema) * self._emb[k] + self.ema * emb
        v /= np.linalg.norm(v) + 1e-9
        self._emb[k] = v
        obj.emb = v
        obj.frames.add(frame)
        obj.bmin = np.minimum(obj.bmin, pts.min(0))
        obj.bmax = np.maximum(obj.bmax, pts.max(0))
        self._bmin[k], self._bmax[k] = obj.bmin, obj.bmax
        self._accum(obj, keyset, sums, counts)
        return obj.cid

    def _insert(self, emb: np.ndarray, pts: np.ndarray,
                keyset: list, sums: np.ndarray, counts: np.ndarray, frame: int) -> int:
        obj = _Object(emb=emb.copy(), bmin=pts.min(0).copy(), bmax=pts.max(0).copy(),
                      cid=self._next_cid)
        self._next_cid += 1
        obj.frames.add(frame)
        self._accum(obj, keyset, sums, counts)
        self._objs.append(obj)
        self._emb = np.vstack([self._emb, emb[None]])
        self._bmin = np.vstack([self._bmin, obj.bmin[None]])
        self._bmax = np.vstack([self._bmax, obj.bmax[None]])
        return obj.cid

    # -- merge pass ---------------------------------------------------------

    def _merge(self) -> None:
        """Union existing objects that are adjacent (>= `merge_overlap` of the
        smaller's voxels within `merge_dilate` of the other) and semantically
        similar (cosine >= `merge_csim`). Run before each flush to consolidate
        fragments of one surface that never co-occurred in a single mask."""
        n = len(self._objs)
        if n < 2:
            return
        reach = self.merge_dilate * self.voxel
        lo, hi = self._bmin - reach, self._bmax + reach
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            # broad phase: objects whose (inflated) AABBs intersect i's
            aabb = np.all((lo[i] <= hi[i + 1:]) & (hi[i] >= lo[i + 1:]), axis=1)
            for dj in np.where(aabb)[0]:
                j = i + 1 + int(dj)
                if find(i) == find(j):
                    continue
                if float(self._emb[i] @ self._emb[j]) < self.merge_csim:
                    continue
                if self._touch_frac(self._objs[i].vox, self._objs[j].vox,
                                    self._merge_offsets) >= self.merge_overlap:
                    parent[find(i)] = find(j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        if len(groups) == n:
            return

        new_objs: list[_Object] = []
        for members in groups.values():
            if len(members) == 1:
                new_objs.append(self._objs[members[0]])
                continue
            objs = [self._objs[m] for m in members]
            vox: dict = {}
            for o in objs:
                for key, val in o.vox.items():
                    cur = vox.get(key)
                    if cur is None:
                        vox[key] = val.copy()
                    else:
                        cur[:3] += val[:3]
                        cur[3] += val[3]
            w = np.array([len(o.vox) for o in objs], float)
            emb = (w[:, None] * np.stack([o.emb for o in objs])).sum(0)
            emb = (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)
            new_objs.append(_Object(
                emb=emb,
                bmin=np.min([o.bmin for o in objs], axis=0),
                bmax=np.max([o.bmax for o in objs], axis=0),
                cid=min(o.cid for o in objs),           # keep the oldest color
                vox=vox,
                frames=set().union(*(o.frames for o in objs)),
            ))

        self._objs = new_objs
        self._emb = (np.stack([o.emb for o in new_objs]) if new_objs
                     else np.empty((0, self._emb.shape[1]), np.float32))
        self._bmin = np.stack([o.bmin for o in new_objs]) if new_objs else np.empty((0, 3))
        self._bmax = np.stack([o.bmax for o in new_objs]) if new_objs else np.empty((0, 3))

    # -- streaming ----------------------------------------------------------

    def push(self, geo, masks, rgb, sem_conf=None) -> None:
        """Fuse one frame. `geo` is a VggtStream dict or FrameGeometry; `masks`
        is a sequence of per-instance boolean masks (RGB resolution); `rgb` is
        the frame the masks came from (for the appearance crops). `sem_conf` is
        an optional per-pixel mask confidence; pixels below sem_conf_thresh are
        dropped. Sets `last_masks`/`last_cids`: the masks kept this frame and the
        stable color id each was associated to (-1 if it contributed no points)."""
        get = geo.get if isinstance(geo, Mapping) else lambda k: getattr(geo, k, None)
        rgb = np.asarray(rgb)
        depth = np.asarray(get("depth"))
        
        self.last_masks = []
        self.last_cids = []
        
        masks = [m for m in (to_bool(m) for m in masks) if int(m.sum()) >= self.min_area]
        masks.sort(key=lambda m: int(m.sum()), reverse=True)
        
        if not masks:
            self.frames += 1
            return

        # appearance embeddings from RGB bbox crops, background faded toward
        # black so the embedding sees the object, not the surrounding room
        # (one batched forward)
        embs = self.embedder([bbox_crop(rgb, m, pad=3, bg_fade=self.bg_fade) for m in masks])

        # merged label map: paint larger masks first so smaller (finer) masks
        # win on overlap; index k -> label k+1, background 0.
        lab = np.zeros(masks[0].shape, np.int64)
        for k, m in enumerate(masks):
            lab[m] = k + 1
        if lab.shape != depth.shape:
            lab = resize_nearest(lab, depth.shape)
        if self.erode:
            lab = erode_labels(lab, self.erode)

        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        if self.edge_rtol > 0:
            valid &= ~depth_edge(depth, rtol=self.edge_rtol)
        conf = get("conf")
        if conf is not None and self.geo_conf_thresh > 0 and np.any(valid):
            conf = np.asarray(conf)
            valid &= conf >= np.percentile(conf[valid], self.geo_conf_thresh)
        if sem_conf is not None and self.sem_conf_thresh > 0:
            mc = np.asarray(sem_conf, np.float32)
            if mc.shape != depth.shape:
                mc = resize_nearest(mc, depth.shape)
            valid &= mc >= self.sem_conf_thresh

        points, plab = unproject_labeled(depth, get("K"), get("c2w"), lab, stride=self.stride, valid=valid)
        keep = plab > 0
        points, plab = points[keep], plab[keep]
        
        cids: list[int] = []
        for k in range(len(masks)):
            pts = points[plab == k + 1]
            if len(pts) == 0:
                cids.append(-1)
                continue
            keyset, sums, counts = self._voxelize(pts)
            emb, centroid = embs[k], pts.mean(0)
            j = self._match(emb, centroid, keyset)
            if j is None:
                cids.append(self._insert(emb, pts, keyset, sums, counts, self.frames))
            else:
                cids.append(self._update(j, emb, pts, keyset, sums, counts, self.frames))

        self.last_masks = masks
        self.last_cids = cids
        self.frames += 1

    def flush(self, *, min_obs: int = 1, min_points: int = 10, frame: int = -1) -> dict:
        """Consolidate objects (merge pass) then snapshot them as a labeled
        cloud. Drops objects seen in fewer than `min_obs` frames or with fewer
        than `min_points` occupied voxels. Each surviving voxel contributes its
        centroid; `labels` is the stable object color id."""
        self._merge()
        pts, labs, hits, obj_labels, obj_frames = [], [], [], [], []
        for obj in self._objs:
            if len(obj.frames) < min_obs or len(obj.vox) < min_points:
                continue
            vals = np.array(list(obj.vox.values()))          # (V,4): sum3 + count
            centroids = vals[:, :3] / vals[:, 3:4]
            pts.append(centroids)
            labs.append(np.full(len(centroids), obj.cid, np.int64))
            hits.append(vals[:, 3].astype(np.int64))
            obj_labels.append(obj.cid)
            obj_frames.append(len(obj.frames))
        return {
            "points": np.concatenate(pts).astype(np.float32) if pts else np.empty((0, 3), np.float32),
            "labels": np.concatenate(labs) if labs else np.empty(0, np.int64),
            "hits": np.concatenate(hits) if hits else np.empty(0, np.int64),
            "obj_labels": np.array(obj_labels, np.int64),
            "obj_frames": np.array(obj_frames, np.int64),
            "frame": int(frame),
        }
