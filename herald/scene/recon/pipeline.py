"""Tier-1 per-session reconstruction: RGB(-D) -> static scene cloud + static OBBs.

SessionRecon streams frames through the Boxer engine (services.boxer: detect ->
3D lift -> Hungarian tracker), accumulates the whole-scene colored point cloud, and
filters in-video movers with the corridor filter -- both online (a dynamic track's
2D box is skipped when unprojecting the scene cloud, so movers never smear it) and
at finalize (movers are dropped from the object list). All dynamic-object logic
lives here; Boxer just reports the per-frame track association. Knows nothing about
the multi-session map -- that is Tier 2 (refine)."""

from __future__ import annotations

import numpy as np

from herald.scene.recon.corridor import classify_static, motion
from herald.scene.recon.types import SceneObject, SessionResult
from herald.scene.recon.utils import SceneCloud, obb_inter_vol, quat_to_R, unproject_rgb


def suppress_contained(objs: list, thresh: float = 0.7) -> list:
    """Drop object dicts whose OBB is >= `thresh` contained within a strictly larger
    object's OBB -- part-of-a-whole detections (a bench 'seat' inside the bench) that
    would otherwise start their own jumpy track and read as false movers. Containment
    = (exact intersection volume) / (small-box volume); only genuine nesting fires, so
    real objects merely near each other are kept."""
    R = [quat_to_R(o["quat_xyzw"]) for o in objs]
    half = [np.asarray(o["size"], np.float32) for o in objs]
    vol = [8.0 * float(np.prod(h)) for h in half]                    # full-box volume
    keep = [True] * len(objs)
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i == j or not keep[j] or vol[j] <= vol[i]:            # j must be the larger box
                continue
            inter = obb_inter_vol(objs[i]["center"], half[i], R[i], objs[j]["center"], half[j], R[j])
            if inter / vol[i] >= thresh:                             # >= thresh of box i sits inside box j
                keep[i] = False
                break
    return [o for o, k in zip(objs, keep) if k]


def _scene_object(o: dict, session_id: str) -> SceneObject:
    """Boxer's per-track dict -> a unified SceneObject. For a single session the object's
    identity is its track id, and its provenance is this one session's support/conf/track +
    crop references (frame_idx + 2D box, resolved to thumbnails against the source path)."""
    tid = int(o["track_id"])
    support, conf = int(o["support"]), float(o.get("conf", 0.0))
    return SceneObject(uid=tid, center=o["center"], half_size=o["size"],
                       quat_xyzw=o["quat_xyzw"], label=o["label"], labels=dict(o["labels"]),
                       conf=conf, support=support,
                       sessions={session_id: {"support": support, "conf": conf, "track_id": tid,
                                              "crops": list(o.get("crop_refs", []))}},
                       embedding=None)


class SessionRecon:
    """Per-session recon over one video. Drive it frame by frame with add_frame()
    (returns the 2D detections for live rendering) then finalize(), or headless via
    run(geometry.frames(...)). Boxer keyword args pass straight through."""

    def __init__(self, vocab, *, device="cuda", scene_voxel=0.1, scene_stride=8,
                 max_depth=60.0, corridor_step=0.2, contain_thr=0.7,
                 dyn_min_support=8, **boxer_kw):
        from services.boxer import BoxerPipeline

        self.pipe = BoxerPipeline(vocab, device=device, online=True, **boxer_kw)
        self.scene = SceneCloud(voxel=scene_voxel)
        self.scene_stride = scene_stride
        self.max_depth = max_depth
        self.corridor_step = corridor_step
        self.contain_thr = contain_thr
        self.dyn_min_support = dyn_min_support
        self._traj: dict[int, list] = {}   # track_id -> [world centroid (NED)] this session
        self._dyn: set[int] = set()        # tracks whose corridor already marks them dynamic

    def add_frame(self, rgb, K, c2w, depth, frame_idx=0) -> dict:
        info = self.pipe.add_frame(rgb, K, c2w, depth, frame_idx=frame_idx)
        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        # Corridor filter: grow each matched track's trajectory and flag it dynamic
        # once its per-step motion crosses the threshold. The flag is sticky (never
        # cleared), so a dynamic track's 2D box stays cut from `valid` and its pixels
        # never reach the scene cloud -- otherwise a mover smears a trail across it.
        for tid, box, ctr in zip(info.get("match_tids", ()), info.get("match_boxes", ()),
                                 info.get("match_centers", ())):
            tid = int(tid)
            self._traj.setdefault(tid, []).append(ctr)
            if tid not in self._dyn and motion(self._traj[tid]) > self.corridor_step:
                self._dyn.add(tid)
            if tid in self._dyn:
                x0, y0, x1, y1 = box.astype(int)
                valid[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
        pts, cols = unproject_rgb(depth, K, c2w, rgb, self.scene_stride, valid)
        self.scene.add(pts, cols)
        return info

    def finalize(self, session_id: str = "session", source: str | None = None) -> SessionResult:
        objs = self.pipe.objects()                                 # conf kept as attribute, not a filter
        objs = suppress_contained(objs, thresh=self.contain_thr)   # drop part-of-a-whole boxes
        for o in objs:
            o["traj"] = np.asarray(self._traj.get(o["track_id"], ()), np.float32)
        static, dynamic = classify_static(objs, max_step=self.corridor_step,
                                          min_support=self.dyn_min_support)
        pts, cols = self.scene.cloud()
        meta = {"sessions": [session_id]}
        if source is not None:                                     # path to resolve crop refs at render time
            meta["sources"] = {session_id: str(source)}
        return SessionResult(pts, cols, [_scene_object(o, session_id) for o in static],
                             dynamic=[_scene_object(o, session_id) for o in dynamic], meta=meta)

    def run(self, frames, session_id: str = "session", source: str | None = None) -> SessionResult:
        for fi, rgb, K, c2w, depth in frames:
            self.add_frame(rgb, K, c2w, depth, frame_idx=fi)
        return self.finalize(session_id, source)
