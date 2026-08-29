"""Boxer's complete detection->3D-lift->tracking pipeline, as one callable service.

Boxer (facebookresearch/boxer) is vendored under ``third_party/boxer`` and is
not pip-installable (no packaging; its top-level ``utils``/``loaders`` modules
collide with HERALD's ``utils``). We prepend its dir to ``sys.path`` so those
resolve to Boxer's copies -- safe because HERALD's ``utils`` is Phase-1 only,
never on the recon path. Needs a CUDA GPU + downloaded checkpoints
(``third_party/boxer/scripts/download_ckpts.sh``).

``BoxerPipeline`` faithfully replicates ``run_boxer.py``'s per-frame flow:
OWLv2 open-vocab detector -> BoxerNet 3D lift -> the labels
stamped onto the OBBs (``set_text``, so the tracker's semantic merge can fire)
-> confidence fused as ``(score2d + score3d) / 2`` -> Boxer's online
``BoundingBox3DTracker`` (Hungarian match + occlusion-aware aging via the frame's
semidense points + duplicate-track merge). ``online=False`` swaps the tracker for
Boxer's one-shot 3D-IoU fuser. Objects come back in the TartanGround (NED) world
frame, each carrying its aggregated detector labels + a few RGB crops.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

_BOXER = Path(__file__).resolve().parents[1] / "third_party" / "boxer"
if str(_BOXER) not in sys.path:
    sys.path.insert(0, str(_BOXER))            # Boxer's utils/boxernet/loaders resolve here

# Boxer's flat top-level `utils`/`loaders` collide with HERALD's. If HERALD's were
# already imported (e.g. Phase-1 via `herald`), evict them so Boxer's win here.
# Safe: HERALD's `utils` is Phase-1 only (unused on the recon path), and modules
# already imported keep their bound names.
for _m in [m for m in list(sys.modules)
           if m in ("utils", "loaders") or m.startswith(("utils.", "loaders."))]:
    if str(_BOXER) not in (getattr(sys.modules[_m], "__file__", "") or ""):
        del sys.modules[_m]

from boxernet.boxernet import BoxerNet                       # noqa: E402
from loaders.base_loader import BaseLoader as _BL            # noqa: E402
from utils.fuse_3d_boxes import BoundingBox3DFuser           # noqa: E402
from utils.tw.obb import ObbTW, iou_mc7                      # noqa: E402
from utils.tw.pose import PoseTW, rotmat_to_quat             # noqa: E402
from utils.tw.tensor_utils import pad_string, string2tensor  # noqa: E402

_CKPT = _BOXER / "ckpts" / "boxernet_hw960in2x6d768-c88128f8.ckpt"
# TartanGround NED (z-down) <-> Boxer z-up world (gravity [0,0,-1]); 180 deg about
# x, involutory so the same matrix maps both ways.
_R_FIX = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _quat_xyzw_to_R(q) -> np.ndarray:
    """Unit quaternion (x,y,z,w) -> 3x3 rotation. (Local copy so this file stays
    independent of HERALD's evicted `utils`.)"""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], np.float32)


class BoxerPipeline:
    """Boxer's complete model behind ``add_frame(rgb, K, c2w, depth)``.

    Boxer's own OWLv2 open-vocab detector feeds the BoxerNet 3D lift; detections
    are then merged across frames either online (``online=True``: Boxer's Hungarian
    ``BoundingBox3DTracker`` -- occlusion-aware aging + semantic-gated duplicate
    merge -- updated in ``add_frame``, read via ``objects()``) or offline
    (``online=False``: buffer + ``objects()`` runs the 3D-IoU fuser). Both keep an
    aggregated label distribution per object; online also keeps up to ``n_crops``
    reservoir-sampled **crop references** (``(frame_idx, box_xyxy)``, not pixels) so a
    renderer can pull a thumbnail on demand, and ``add_frame`` returns the per-detection
    track association so a caller can build per-track trajectories (e.g. recon's corridor
    filter)."""

    def __init__(self, vocab, *, device="cuda", ckpt=_CKPT, conf_thr2d=0.40, conf_thr3d=0.50,
                 iou_thr=0.3, min_obs=4, online=False, n_crops=3, persist=True,
                 max_range=15.0, min_visible=0.0):
        self.vocab = list(vocab)
        self.device = device
        self.conf_thr2d, self.conf_thr3d, self.iou_thr, self.min_obs = conf_thr2d, conf_thr3d, iou_thr, min_obs
        self.online, self.n_crops = online, n_crops
        # Per-frame box-quality gate before the tracker's Hungarian merge: only boxes
        # in front of the camera that project inside the image and sit within
        # `max_range` metres are trusted enough to associate/merge a track.
        self.max_range = max_range
        # Truncation gate: a box whose reprojected 3D OBB has less than `min_visible`
        # of its area inside the frame is a truncated/edge-straddling lift (unreliable
        # 3D). 0.0 disables it (only the centre-in-frame check above then applies).
        self.min_visible = min_visible

        from owl.owl_wrapper import OwlWrapper

        # precision=None -> auto (bf16 on cuda). Text embeddings for `vocab` are
        # computed once and disk-cached by prompt hash under ckpts/.
        self.owl = OwlWrapper(device, text_prompts=self.vocab, min_confidence=conf_thr2d, precision=None)

        self.model = BoxerNet.load_from_checkpoint(str(ckpt), device=device)
        self.model.eval()
        self.hw = int(getattr(self.model, "hw", 960))

        self._rows, self._labels = [], []          # offline: per-detection ObbTW + label
        if online:
            from utils.track_3d_boxes import BoundingBox3DTracker

            # Native run_boxer.py config: iou=0.25, min_hits=8, conf=conf_thr3d,
            # max_missed=90. `persist` bumps max_missed so nothing is pruned (a
            # persistent map for the rerun snapshot); the duplicate-track merge
            # (default merge_iou/merge_semantic thresholds) still fires either way.
            self.tracker = BoundingBox3DTracker(
                iou_threshold=0.25, min_hits=min_obs, conf_threshold=conf_thr3d,
                samp_per_dim=8, max_missed=10**9 if persist else 90, verbose=False)
            self._tlabels: dict[int, Counter] = defaultdict(Counter)   # track_id -> label counts
            self._trefs: dict[int, list] = defaultdict(list)           # track_id -> [(frame_idx, box_xyxy)]
            self._tseen: dict[int, int] = defaultdict(int)             # track_id -> #refs offered
            self._tframes: dict[int, set] = defaultdict(set)           # track_id -> {observing frame_idx}

            # When Boxer merges two tracks, combine our per-track attachments onto the
            # survivor too (else the absorbed id's data orphans): label votes union,
            # crop refs union then sample back to n_crops.
            _merge = self.tracker._merge_track_pair

            def _merge_attrs(absorber, absorbed, _merge=_merge):
                a, b = int(absorber.track_id), int(absorbed.track_id)
                _merge(absorber, absorbed)
                self._tlabels[a].update(self._tlabels.pop(b, Counter()))
                pool = self._trefs.pop(a, []) + self._trefs.pop(b, [])
                if pool:
                    self._trefs[a] = pool if len(pool) <= self.n_crops else random.sample(pool, self.n_crops)
                self._tseen[a] = self._tseen.pop(a, 0) + self._tseen.pop(b, 0)
                self._tframes[a] |= self._tframes.pop(b, set())         # union observing frames

            self.tracker._merge_track_pair = _merge_attrs

    def add_frame(self, rgb, K, c2w, depth=None, frame_idx=0) -> dict:
        """Detect + lift one frame; merge (online) or buffer (offline). Returns
        {n_obj, boxes (xyxy image res), labels} for the 2D detections, plus (online)
        {match_tids, match_boxes, match_centers}: the per-detection track this frame
        matched -- its track_id, image-res xyxy box, and NED centroid."""
        boxes_xyxy, bb2d_hw, labels, scores2d = self._detect(rgb)
        if len(boxes_xyxy) == 0:
            return {"n_obj": 0, "boxes": boxes_xyxy, "labels": labels}

        datum = self._datum(rgb, bb2d_hw, K, c2w, depth)
        obbs = self._forward(datum)                          # cpu ObbTW, one per detection
        scores3d = obbs.prob.squeeze(-1).clone()
        keep = (scores3d >= self.conf_thr3d)
        km = keep.numpy()
        dets = obbs[keep].clone()
        lab_k = [labels[i] for i in range(len(labels)) if km[i]]
        box_k = boxes_xyxy[km]

        # Faithful to run_boxer.py: fuse 2D+3D confidence and STAMP the label onto
        # each OBB (text slot). The tracker derives `cached_text` from this, which
        # is what its semantic-gated duplicate merge keys on -- without it the merge
        # is dead and overlapping boxes never collapse.
        if len(dets):
            mean_scores = (scores2d[keep] + scores3d[keep]) / 2.0
            dets.set_prob(mean_scores)
            text = torch.stack([string2tensor(pad_string(l, max_len=128)) for l in lab_k])
            dets.set_text(text)

        if not self.online:
            for i in range(len(dets)):
                self._rows.append(dets[i]._data.reshape(-1))
                self._labels.append(lab_k[i])
            return {"n_obj": len(dets), "boxes": boxes_xyxy, "labels": labels}

        # Quality gate: keep only clean boxes for the tracker's single-video Hungarian
        # merge (confident + in-frustum + near). Low-quality lifts never seed/associate.
        gate = self._quality_gate(dets, K, c2w, rgb.shape[:2])
        dets = dets[gate].clone()
        lab_k = [lab_k[i] for i in range(len(lab_k)) if gate[i]]
        box_k = box_k[gate]
        if len(dets) == 0:
            return {"n_obj": len(self.tracker.get_all_tracks()), "boxes": boxes_xyxy,
                    "labels": labels, **self._attribute(dets, lab_k, box_k, [], frame_idx)}

        tracks = self.tracker.update(dets, frame_idx, datum["cam0"], datum["T_world_rig0"],
                                     observed_points=datum["sdp_w"])
        matches = self._attribute(dets, lab_k, box_k, tracks, frame_idx)
        return {"n_obj": len(tracks), "boxes": boxes_xyxy, "labels": labels, **matches}

    def objects(self) -> list[dict]:
        """Current objects -> [{center, quat_xyzw, size, conf, support, label,
        labels, track_id, crop_refs}] in NED (conf = fused detection confidence;
        crop_refs = [(frame_idx, box_xyxy)] for render-time thumbnails). Online reads
        live tracks; offline fuses."""
        if not self.online:
            return self._fuse()
        out = []
        for t in self.tracker.get_all_tracks():                # post-merge, never pruned -> persistent
            if t.support_count < self.min_obs:
                continue
            tid = int(t.track_id)
            labs = self._tlabels[tid] or Counter({t.cached_text: 1})
            c, q, s = self._to_ned(t.obb)
            out.append({"center": c, "quat_xyzw": q, "size": s,
                        "conf": float(np.asarray(t.obb.prob).reshape(-1)[0]),  # fused (score2d+score3d)/2
                        "support": int(t.support_count), "track_id": tid,
                        "label": labs.most_common(1)[0][0], "labels": dict(labs),
                        "crop_refs": list(self._trefs[tid]),
                        "frames": sorted(self._tframes[tid])})   # all observing frame indices (multi-anchor)
        return out

    def _fuse(self) -> list[dict]:
        if not self._rows:
            return []
        allo = ObbTW(torch.stack(self._rows))
        fuser = BoundingBox3DFuser(iou_threshold=self.iou_thr, min_detections=self.min_obs,
                                   conf_threshold=0.0, semantic_threshold=0.0, enable_nms=False)
        out = []
        for ins in fuser.fuse(allo, semantic_embeddings=None):
            labs = Counter(self._labels[j] for j in ins.detection_indices)
            c, q, s = self._to_ned(ins.obb)
            out.append({"center": c, "quat_xyzw": q, "size": s,
                        "conf": float(np.asarray(ins.obb.prob).reshape(-1)[0]),
                        "support": int(ins.support_count), "track_id": -1,
                        "label": labs.most_common(1)[0][0], "labels": dict(labs), "crop_refs": []})
        return out

    # --- detection: return image-res xyxy (viz/crops) + hw-scale boxer-format bb2d ---

    def _detect(self, rgb):
        """OWLv2 -> (boxes_xyxy [N,4] image res, bb2d_hw [N,4] boxer fmt at self.hw,
        labels [N] str, scores2d [N] tensor). Empty arrays when nothing detected."""
        import cv2

        H, W = rgb.shape[:2]
        sx, sy = self.hw / W, self.hw / H
        img = cv2.resize(rgb, (self.hw, self.hw), interpolation=cv2.INTER_LINEAR)
        t255 = torch.from_numpy(img).permute(2, 0, 1).float()[None]   # (1,3,hw,hw) in [0,255]
        # OWL returns boxes already in Boxer format (x1,x2,y1,y2), scaled to the
        # INPUT tensor size (= self.hw), plus per-class NMS + too-big/small filter.
        bb2d_hw, scores2d, label_ints, _ = self.owl.forward(t255, resize_to_HW=(self.hw, self.hw))
        if len(bb2d_hw) == 0:
            return np.zeros((0, 4), np.float32), bb2d_hw, [], scores2d
        labels = [self.vocab[int(i)] for i in label_ints]
        xyxy_hw = bb2d_hw[:, [0, 2, 1, 3]].numpy()                    # boxer -> xyxy at hw
        boxes_xyxy = (xyxy_hw / np.array([sx, sy, sx, sy], np.float32)).astype(np.float32)
        return boxes_xyxy, bb2d_hw.float(), labels, scores2d.float()

    def _quality_gate(self, dets, K, c2w, hw) -> np.ndarray:
        """Boolean keep-mask over `dets`: a box is trusted for the tracker only if its
        3D centre is in front of the camera and projects inside the image, it lies
        within max_range metres, and (when min_visible>0) enough of its reprojected 3D
        OBB falls inside the frame. Rejects the low-quality lifts (behind camera /
        off-image / far / truncated) that otherwise seed junk tracks and block clean
        merges. (Per-detection confidence is already gated upstream by conf_thr*.)"""
        n = len(dets)
        if n == 0:
            return np.zeros(0, bool)
        centers = np.stack([self._to_ned(dets[i])[0] for i in range(n)])   # (n,3) NED world
        c2w = np.asarray(c2w, np.float32)
        cam = (centers - c2w[:3, 3]) @ c2w[:3, :3]                         # world -> camera
        z = cam[:, 2]
        H, W = hw
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z + K[1, 2]
        dist = np.linalg.norm(centers - c2w[:3, 3], axis=1)
        keep = ((z > 0) & (dist <= self.max_range)
                & (u >= 0) & (u < W) & (v >= 0) & (v < H))
        if self.min_visible > 0.0:
            keep &= self._visible_fraction(dets, K, c2w, hw) >= self.min_visible
        return keep

    def _visible_fraction(self, dets, K, c2w, hw) -> np.ndarray:
        """Fraction of each det's 3D OBB that reprojects inside the image: area of the
        frame-clipped projected AABB / area of the full projected AABB. A truncated or
        edge-straddling box scores low; any OBB corner behind the image plane forces 0
        (a straddling lift is unreliable). Vectorised over the 8 corners per det."""
        n = len(dets)
        c2w = np.asarray(c2w, np.float32)
        R_cw, t_cw = c2w[:3, :3], c2w[:3, 3]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H, W = hw
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         np.float32)                                        # (8,3)
        frac = np.zeros(n, np.float32)
        for i in range(n):
            ctr, quat, half = self._to_ned(dets[i])
            corners = ctr[None, :] + (signs * half[None, :]) @ _quat_xyzw_to_R(quat).T  # (8,3) world
            cam = (corners - t_cw) @ R_cw                                   # world -> camera
            zc = cam[:, 2]
            if np.any(zc <= 1e-3):
                continue                                                   # straddles image plane -> 0
            uu = fx * cam[:, 0] / zc + cx
            vv = fy * cam[:, 1] / zc + cy
            u0, u1, v0, v1 = uu.min(), uu.max(), vv.min(), vv.max()
            full = max(u1 - u0, 1e-6) * max(v1 - v0, 1e-6)
            clip = max(min(u1, W) - max(u0, 0.0), 0.0) * max(min(v1, H) - max(v0, 0.0), 0.0)
            frac[i] = clip / full
        return frac

    # --- online attribution: route this frame's labels + crops to tracks by IoU,
    # and report the per-detection track association back to the caller ---

    def _attribute(self, dets, labels, boxes2d, tracks, frame_idx) -> dict:
        empty = {"match_tids": np.zeros(0, np.int64), "match_boxes": np.zeros((0, 4), np.float32),
                 "match_centers": np.zeros((0, 3), np.float32)}
        if len(dets) == 0 or len(tracks) == 0:
            return empty
        track_obbs = ObbTW(torch.stack([t.obb._data.reshape(-1) for t in tracks]))
        iou = np.asarray(iou_mc7(dets, track_obbs, samp_per_dim=8, all_pairs=True).cpu())  # (nDet, nTrk)
        tids, boxes, centers = [], [], []
        for di in range(len(dets)):
            j = int(iou[di].argmax())
            if iou[di, j] < self.iou_thr:
                continue
            tid = int(tracks[j].track_id)
            box = np.asarray(boxes2d[di], np.float32)
            self._tlabels[tid][labels[di]] += 1
            self._tframes[tid].add(int(frame_idx))               # every frame that observed this track
            self._reservoir(tid, (int(frame_idx), box))          # ref this observation (frame + box)
            tids.append(tid); boxes.append(box); centers.append(self._to_ned(dets[di])[0])
        return {"match_tids": np.asarray(tids, np.int64),
                "match_boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
                "match_centers": np.asarray(centers, np.float32).reshape(-1, 3)}

    def _reservoir(self, tid, ref):
        """Reservoir-sample up to n_crops crop refs per track (uniform, no full history)."""
        self._tseen[tid] += 1
        buf = self._trefs[tid]
        if len(buf) < self.n_crops:
            buf.append(ref)
        else:
            r = random.randint(0, self._tseen[tid] - 1)
            if r < self.n_crops:
                buf[r] = ref

    # --- minimal model-call helpers ---

    def _datum(self, rgb, bb2d_hw, K, c2w, depth):
        """Build a BoxerNet datum. ``bb2d_hw`` is already in Boxer format
        (xmin,xmax,ymin,ymax) at ``self.hw`` scale -- both detectors produce that."""
        import cv2

        H, W = rgb.shape[:2]
        rW = rH = self.hw
        sx, sy = rW / W, rH / H
        img = cv2.resize(rgb, (rW, rH), interpolation=cv2.INTER_LINEAR)
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        c2w = np.asarray(c2w, np.float32)
        R, t = _R_FIX @ c2w[:3, :3], _R_FIX @ c2w[:3, 3]      # gravity-fixed pose (z-up)
        d = None
        if depth is not None:
            d = np.asarray(depth, np.float32)
            d = np.where(np.isfinite(d) & (d > 0), d, 0.0)
            if d.shape != (rH, rW):
                d = cv2.resize(d, (rW, rH), interpolation=cv2.INTER_NEAREST)
        # CPU datum (BoxerNet.forward moves to its device internally; keeping cam/
        # pose on CPU also matches the CPU obbs we hand the tracker).
        return {
            "img0": _BL.img_to_tensor(img),
            "cam0": _BL.pinhole_from_K(rW, rH, fx, fy, cx, cy, valid_radius=(rW, rH)).float(),
            "T_world_rig0": PoseTW(torch.tensor([*R.flatten(), *t], dtype=torch.float32)),
            "sdp_w": _BL.sdp_from_depth(d, fx, fy, cx, cy, R, t) if d is not None else torch.zeros(0, 3),
            "bb2d": bb2d_hw.float(),
        }

    @torch.no_grad()
    def _forward(self, datum):
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if self.device == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
        with ctx:
            out = self.model.forward(datum)
        return out["obbs_pr_w"].cpu()[0]

    @staticmethod
    def _to_ned(o):
        """ObbTW (z-up) -> (center(3), quat xyzw(4), half(3)) in NED."""
        T = o.T_world_object
        Rw = T.fit_to_SO3().R.numpy().reshape(3, 3).astype(np.float32)
        e = np.asarray(o.bb3_object, np.float32).reshape(6)
        off = np.array([(e[0] + e[1]) / 2, (e[2] + e[3]) / 2, (e[4] + e[5]) / 2], np.float32)
        center = _R_FIX @ (np.asarray(T.t, np.float32).reshape(3) + Rw @ off)
        qw, qx, qy, qz = rotmat_to_quat(_R_FIX @ Rw)
        half = np.array([(e[1] - e[0]) / 2, (e[3] - e[2]) / 2, (e[5] - e[4]) / 2], np.float32)
        return center, np.array([qx, qy, qz, qw], np.float32), half
