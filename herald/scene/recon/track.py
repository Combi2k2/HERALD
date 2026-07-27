"""Chunk-based SAM2 video tracking + 3D voxel-overlap merge -> voxel-majority-vote
labeled cloud.

Two layers of identity, so neither over- nor under-fuses:

- WITHIN a chunk: SAM2 video propagation (`Sam2Tracker`). Fixed CHUNKS bound
  memory (a lifelong session OOM'd -- every discovery adds a conditioning frame
  SAM2 never prunes), and each chunk RE-SEEDS with automatic masks so newly
  visible objects keep getting discovered. Each chunk mints FRESH local ids;
  the old cross-chunk IoU carry is gone (it minted a new id whenever the 2D
  overlap missed, exploding one surface into hundreds of ids).
- ACROSS chunks: `TrackReconStream` unprojects each chunk's local objects to
  world voxels and matches them to persistent global objects by voxel
  co-occupancy (OVERLAP, not adjacency), remapping local->global before voting.
  Overlap (not touch) means the floor can't bridge distinct objects, and two
  identical objects at different places stay separate.

`VoteCloud` accumulates the remapped labels into one cloud: per (voxel,
global-id) votes resolved to the majority label at flush, with flicker-prone
pixels dropped BEFORE they vote (frame-border margin, depth discontinuities,
eroded mask edges, optional per-pixel confidence floor). No appearance
embedding, no union-find. Fragment-rejoining (parts, revisits) is left out.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Mapping

import numpy as np
from PIL import Image

from herald.scene.recon.utils import (
    as_rgb,
    depth_edge,
    erode_labels,
    filter_clusters,
    resize_nearest,
    to_bool,
    unproject_labeled,
)

DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"


class Sam2Tracker:
    """Chunk-buffered SAM2 video tracker. push(rgb) -> [] until a chunk fills,
    then [(label uint16, conf float32), ...] for that chunk's frames; finish()
    flushes the remainder. Ids are FRESH local ids (1..N) each chunk -- id 0 =
    background; cross-chunk identity is handled by TrackReconStream, not here."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        chunk: int = 16,
        points_per_side: int = 16,
        points_per_batch: int = 32,
        min_area: int = 400,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.9,
        offload: bool = True,
    ) -> None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor, pipeline

        self._torch = torch
        self.device = device
        self.model = Sam2VideoModel.from_pretrained(model_name).to(device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(model_name)
        self.maskgen = pipeline("mask-generation", model=model_name, device=device)
        self.chunk = max(1, chunk)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.min_area = min_area
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.offload = offload
        self.reset()

    def reset(self) -> None:
        self._buffer: list[np.ndarray] = []

    def _seed(self, image_arr: np.ndarray) -> list[np.ndarray]:
        """Automatic masks on one frame, largest-first, >= min_area."""
        out = self.maskgen(
            Image.fromarray(image_arr),
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        masks = [m for m in map(to_bool, out["masks"]) if int(m.sum()) >= self.min_area]
        masks.sort(key=lambda m: int(m.sum()), reverse=True)
        return masks

    def _run_chunk(self, frames) -> list[tuple[np.ndarray, np.ndarray]]:
        torch = self._torch
        frames = [as_rgb(f) for f in frames]
        n = len(frames)
        h, w = frames[0].shape[:2]
        labels = [np.zeros((h, w), np.uint16) for _ in range(n)]
        confs = [np.zeros((h, w), np.float32) for _ in range(n)]

        masks = self._seed(frames[0])
        obj_ids = list(range(1, len(masks) + 1))   # fresh local ids per chunk
        if not masks:
            return list(zip(labels, confs))

        state_dev = "cpu" if self.offload else self.device
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            inference_state_device=state_dev,
            video_storage_device=state_dev,
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
                    [out.pred_masks], original_sizes=[[h, w]], binarize=False,
                )[0][:, 0].float().cpu().numpy()          # (num_obj, h, w) logits
                res = logits > 0
                ids = [int(x) for x in list(out.object_ids)]
                lab, cf = labels[out.frame_idx], confs[out.frame_idx]
                for k in np.argsort([-int(m.sum()) for m in res]):   # largest first: finer wins
                    lab[res[k]] = ids[k]
                    cf[res[k]] = 1.0 / (1.0 + np.exp(-logits[k][res[k]]))
        return list(zip(labels, confs))

    def push(self, rgb) -> list[tuple[np.ndarray, np.ndarray]]:
        """Buffer a frame; return this chunk's (label, conf) pairs when it fills."""
        self._buffer.append(as_rgb(rgb))
        if len(self._buffer) >= self.chunk:
            frames, self._buffer = self._buffer, []
            return self._run_chunk(frames)
        return []

    def finish(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Flush the partial trailing chunk."""
        frames, self._buffer = self._buffer, []
        return self._run_chunk(frames) if frames else []


class VoteCloud:
    """Accumulate (geometry, track-label, conf) frames into a voxel-majority-vote
    cloud. Each accepted pixel casts a (voxel, label) vote; flush() resolves each
    voxel to its best-supported label, so occasional per-frame flicker is
    outvoted instead of trusted. Flicker-prone pixels are dropped before voting:
    a frame-border margin, depth discontinuities (depth_edge), eroded mask
    borders, and pixels below a confidence floor. No cross-object merge -- track
    ids are the objects. flush() is non-destructive."""

    def __init__(
        self,
        *,
        voxel: float = 0.05,
        stride: int = 2,
        max_depth: float = 60.0,
        sem_conf_thresh: float = 0.0,
        edge_rtol: float = 0.03,
        erode: int = 1,
        border: int = 12,
        ignore_ids=(0,),
    ) -> None:
        if voxel <= 0:
            raise ValueError("voxel must be > 0")
        self.voxel = voxel
        self.stride = max(1, stride)
        self.max_depth = max_depth
        self.sem_conf_thresh = sem_conf_thresh
        self.edge_rtol = edge_rtol
        self.erode = max(0, erode)
        self.border = max(0, border)
        self.ignore_ids = frozenset(int(i) for i in ignore_ids)
        self._keys = np.empty((0, 4), np.int64)     # ijk voxel + label
        self._sums = np.zeros((0, 3))               # point-coordinate sums per (voxel,label)
        self._counts = np.zeros(0, np.int64)        # votes per (voxel,label)
        self._label_frames: dict[int, int] = {}
        self.frames = 0

    def push(self, geo, labels, sem_conf=None) -> None:
        get = geo.get if isinstance(geo, Mapping) else lambda k: getattr(geo, k, None)
        depth = np.asarray(get("depth"))
        lab = np.asarray(labels)
        if lab.shape != depth.shape:
            lab = resize_nearest(lab, depth.shape)
        if self.erode:
            lab = erode_labels(lab, self.erode)

        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        if self.border > 0:                          # drop the unstable frame-border strip
            b = self.border
            m = np.zeros_like(valid)
            m[b:-b, b:-b] = True
            valid &= m
        if self.edge_rtol > 0:                        # drop flying pixels at depth discontinuities
            valid &= ~depth_edge(depth, rtol=self.edge_rtol)
        if sem_conf is not None and self.sem_conf_thresh > 0:
            mc = np.asarray(sem_conf, np.float32)
            if mc.shape != depth.shape:
                mc = resize_nearest(mc, depth.shape)
            valid &= mc >= self.sem_conf_thresh

        points, plabels = unproject_labeled(
            depth, get("K"), get("c2w"), lab, stride=self.stride, valid=valid)
        keep = plabels >= 0
        for ig in self.ignore_ids:
            keep &= plabels != ig
        points, plabels = points[keep], plabels[keep]
        self.frames += 1
        if len(points) == 0:
            return

        keys = np.concatenate(
            [np.floor(points / self.voxel).astype(np.int64), plabels[:, None]], axis=1)
        m = len(self._keys)
        uniq, inv = np.unique(np.concatenate([self._keys, keys]), axis=0, return_inverse=True)
        inv = inv.ravel()
        sums = np.zeros((len(uniq), 3))
        counts = np.zeros(len(uniq), np.int64)
        np.add.at(sums, inv[:m], self._sums)
        np.add.at(counts, inv[:m], self._counts)
        np.add.at(sums, inv[m:], points)
        np.add.at(counts, inv[m:], 1)
        self._keys, self._sums, self._counts = uniq, sums, counts
        for label in np.unique(plabels):
            self._label_frames[int(label)] = self._label_frames.get(int(label), 0) + 1

    def flush(
        self,
        *,
        eps: float | None = None,
        min_cluster: int = 10,
        min_samples: int = 1,
        min_hits: int = 1,
        min_frames: int = 1,
        frame: int = -1,
    ) -> dict:
        """Snapshot the accumulator as a labeled cloud (non-destructive)."""
        eps = 2.0 * self.voxel if eps is None else eps
        keys, sums, counts = self._keys, self._sums, self._counts
        if len(keys):
            # majority vote: sort so each voxel's highest-count label is last, keep it
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
                keep = filter_clusters(
                    points, eps=eps, min_cluster=min_cluster, min_samples=min_samples)
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


class TrackReconStream:
    """Wire Sam2Tracker (within-chunk ids) to VoteCloud through a 3D
    voxel-overlap cross-chunk merge. push(frame_idx, geo, rgb) buffers a frame
    and, when the tracker emits a chunk, unprojects that chunk's local objects
    to world voxels, matches each to a persistent GLOBAL object by voxel
    co-occupancy (fraction of the local object's voxels landing on an existing
    global object >= merge_overlap, else mint a new global id), remaps the
    labels, and votes them into VoteCloud under global ids. Overlap not
    adjacency, so the floor can't bridge objects. Optional on_frame(frame_idx,
    geo, rgb, global_label, conf) callback fires per frame for rendering."""

    def __init__(
        self,
        tracker: Sam2Tracker,
        vote: VoteCloud,
        *,
        merge_voxel: float = 0.1,
        merge_overlap: float = 0.2,
        max_depth: float = 60.0,
    ) -> None:
        self.tracker = tracker
        self.vote = vote
        self.merge_voxel = merge_voxel
        self.merge_overlap = merge_overlap
        self.max_depth = max_depth
        self._owner: dict[int, int] = {}       # world-voxel code -> global id
        self._next_gid = 1
        self._pending: deque = deque()         # (frame_idx, geo, rgb) buffered for the current chunk

    def push(self, frame_idx: int, geo, rgb, on_frame: Callable | None = None) -> None:
        self._pending.append((frame_idx, geo, rgb))
        chunk = self.tracker.push(rgb)
        if chunk:
            self._drain(chunk, on_frame)

    def finish(self, on_frame: Callable | None = None) -> None:
        chunk = self.tracker.finish()
        if chunk:
            self._drain(chunk, on_frame)

    def _drain(self, chunk, on_frame) -> None:
        frames = [self._pending.popleft() for _ in chunk]     # aligned 1:1 with chunk order
        local_vox: dict[int, set[int]] = defaultdict(set)     # local id -> world-voxel codes (chunk union)
        for (_, g, _), (label, _) in zip(frames, chunk):
            for lid, codes in self._voxels(g, label).items():
                local_vox[lid].update(codes)
        remap = self._match(local_vox)                        # local id -> global id (matched vs. pre-chunk state)
        for lid, gid in remap.items():
            for c in local_vox[lid]:
                self._owner[c] = gid
        for (fi, g, rgb), (label, conf) in zip(frames, chunk):
            glabel = self._relabel(label, remap)
            self.vote.push(g, glabel, conf)
            if on_frame is not None:
                on_frame(fi, g, rgb, glabel, conf)

    def _voxels(self, geo, label) -> dict[int, set[int]]:
        get = geo.get if isinstance(geo, Mapping) else lambda k: getattr(geo, k, None)
        depth = np.asarray(get("depth"))
        lab = np.asarray(label)
        if lab.shape != depth.shape:
            lab = resize_nearest(lab, depth.shape)
        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        points, plabels = unproject_labeled(
            depth, get("K"), get("c2w"), lab, stride=self.vote.stride, valid=valid)
        keep = plabels > 0
        points, plabels = points[keep], plabels[keep]
        if len(points) == 0:
            return {}
        codes = self._encode(np.floor(points / self.merge_voxel).astype(np.int64))
        return {int(lid): set(codes[plabels == lid].tolist()) for lid in np.unique(plabels)}

    def _match(self, local_vox: dict[int, set[int]]) -> dict[int, int]:
        remap: dict[int, int] = {}
        for lid, codes in local_vox.items():
            hits: dict[int, int] = defaultdict(int)
            for c in codes:
                gid = self._owner.get(c)
                if gid:
                    hits[gid] += 1
            best_gid, best = 0, self.merge_overlap * max(1, len(codes))
            for gid, n in hits.items():
                if n >= best:
                    best, best_gid = n, gid
            if not best_gid:
                best_gid, self._next_gid = self._next_gid, self._next_gid + 1
            remap[lid] = best_gid
        return remap

    @staticmethod
    def _relabel(label: np.ndarray, remap: dict[int, int]) -> np.ndarray:
        out = np.zeros_like(label)
        for lid, gid in remap.items():
            out[label == lid] = gid
        return out

    @staticmethod
    def _encode(keys: np.ndarray) -> np.ndarray:
        k = keys.astype(np.int64) + (1 << 20)     # offset so negatives stay non-negative
        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
