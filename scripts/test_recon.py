#!/usr/bin/env python3
"""Diagnostic for the Tier-1 recon pipeline (herald.scene.recon.SessionRecon).

Streams a TartanGround video through the pipeline and renders to Rerun in two panels
(+ the selection sidebar). It's an ONLINE view: every frame, the tracks matched that
frame are drawn as live 3D boxes AND projected into the 2D image from the same set, so
you watch objects spawn/move and the two panels always agree. At the end the finalize
result is drawn -- dropped tracks Cleared, confirmed movers turned red.

  LEFT  (3D world): the original-color scene point cloud, the live/settled object OBBs
        (per-object palette), the camera frustum, and the trajectory.
  RIGHT (2D camera): three stacked, individually-toggleable layers over the current
        frame -- the RGB image, the depth field, and the tracked 3D OBBs projected
        into the image. Toggle RGB off to see the depth layer beneath.
  SIDEBAR: select any box -> its id/conf/support/labels text and RGB crop. The crop is
        an Image on the box entity (shown by the selection panel, which needs no pinhole);
        a per-entity visualizer override keeps the 3D *view* from rendering it, so there
        is no 'requires a pinhole' warning.

Needs a CUDA GPU on the 3090 partition (scripts/boxer.slurm -> SCRIPT override).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from detect_video import DEFAULT_VOCAB
from herald.datasets import TartanGroundGeometry, TartanGroundTraj
from herald.scene.recon import GtGeometry, SessionRecon
from herald.viz import colorize_labels

# 8 box corners as (±1,±1,±1); bit2=x, bit1=y, bit0=z, so corners differing in one
# bit share an edge -> the 12 wireframe edges.
_SIGNS = np.array([[(-1) ** (i >> 2), (-1) ** (i >> 1), (-1) ** i] for i in range(8)], np.float32)
_EDGES = [(i, i | b) for i in range(8) for b in (1, 2, 4) if not (i & b)]


def tag_dict(labels: dict) -> str:
    """Top label votes as 'bench x224, chair x74'."""
    items = sorted(labels.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} x{v}" for k, v in items[:4])


def tag(o) -> str:
    return tag_dict(o.labels)


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


def project_obb(center, half, quat, K, c2w):
    """Project an OBB's 12 wireframe edges into image pixels -> list of (2,2)
    segments, or None if any corner is at/behind the image plane (z <= 0)."""
    corners = np.asarray(center, np.float32) + (_SIGNS * np.asarray(half, np.float32)) @ quat_to_R(quat).T
    cam = (corners - c2w[:3, 3]) @ c2w[:3, :3]          # world -> camera (R_wc^T @ (X - t))
    z = cam[:, 2]
    if np.any(z <= 1e-3):
        return None
    uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    return [uv[[a, b]] for a, b in _EDGES]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0001")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--conf-thr2d", type=float, default=0.40, help="OWLv2 2D detection score floor")
    p.add_argument("--conf-thr3d", type=float, default=0.50, help="BoxerNet 3D-lift score floor")
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--min-obs", type=int, default=4)
    p.add_argument("--max-depth", type=float, default=60.0)
    p.add_argument("--corridor-step", type=float, default=0.2,
                   help="median per-frame centroid step (m) above which an object is dynamic")
    p.add_argument("--max-range", type=float, default=15.0,
                   help="per-frame box-quality gate: max camera->box distance (m) to trust a box")
    p.add_argument("--scene-voxel", type=float, default=0.12)
    p.add_argument("--scene-stride", type=int, default=8)
    p.add_argument("--snap", type=int, default=64, help="frames between live 3D snapshots")
    p.add_argument("--n-crops", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dump", type=Path, default=None,
                   help="also save the SessionResult (objects + cloud + crop-ref provenance) .npz")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    idxs = list(range(len(traj)))[:: max(1, args.frame_stride)]
    if args.max_frames:
        idxs = idxs[: args.max_frames]
    src = GtGeometry(traj, TartanGroundGeometry(traj))

    recon = SessionRecon(DEFAULT_VOCAB, device=args.device, conf_thr2d=args.conf_thr2d,
                         conf_thr3d=args.conf_thr3d, iou_thr=args.iou_thr, min_obs=args.min_obs,
                         n_crops=args.n_crops, scene_voxel=args.scene_voxel,
                         scene_stride=args.scene_stride, max_depth=args.max_depth,
                         corridor_step=args.corridor_step, max_range=args.max_range)

    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("herald.test_recon")

    def make_bp(overrides=None):
        """2 panels + selection sidebar. `overrides` maps each box entity to a
        VisualizerOverrides(["Boxes3D"]) so the 3D view draws only the box, not the crop
        Image hung on the same entity -- the Image still shows in the selection sidebar
        (which needs no pinhole), but the 3D *view* no longer tries to render it, killing
        the 'requires a pinhole ancestor' warning."""
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="scene + objects", overrides=overrides or {}),
                rrb.Spatial2DView(origin="camera/image", name="rgb / depth / obb"),
                column_shares=[3, 2]),
            rrb.SelectionPanel(state="expanded"),
            auto_views=False)

    rr.send_blueprint(make_bp())
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)   # TartanGround NED
    print(f"{args.env}/{args.traj}: OWLv2+Boxer test-recon over {len(idxs)} frames", flush=True)

    def log_scene(fi):
        sp, sc = recon.scene.cloud()
        if len(sp):
            rr.set_time("frame", sequence=fi)
            rr.log("world/scene", rr.Points3D(sp, colors=sc, radii=0.01))

    annotated: set[str] = set()

    def draw_box(ent, o, col, dyn=False, refresh=False):
        """Log a box + its sidebar metadata (text + crop). The crop Image is hung on the
        box entity so selecting the box shows it in the right sidebar (which needs no
        pinhole); a per-entity VisualizerOverrides (see make_bp) stops the 3D *view* from
        trying to render that Image, so there's no pinhole warning. The box is re-logged
        each call so a mover updates; text/crop are static and logged once on first sight,
        or refreshed (`refresh=True`, used at finalize) to the settled conf/support/best
        crop rather than the stale first-frame values."""
        rr.log(ent, rr.Boxes3D(centers=[o["center"]], half_sizes=[o["size"]],
                               quaternions=[o["quat_xyzw"]], colors=[col], fill_mode="majorwireframe"))
        if ent not in annotated or refresh:
            annotated.add(ent)
            head = "DYNAMIC | " if dyn else ""
            rr.log(ent, rr.TextDocument(
                f"id {o['track_id']} | {head}conf {o['conf']:.2f} | support {o['support']}\n"
                f"{tag_dict(o['labels'])}"), static=True)
            if o.get("crops"):
                rr.log(ent, rr.Image(np.concatenate(o["crops"], axis=1)), static=True)

    t0 = time.perf_counter()
    cams: list[np.ndarray] = []
    for k, (fi, rgb, K, c2w, depth) in enumerate(src.frames(idxs), 1):
        info = recon.add_frame(rgb, K, c2w, depth, frame_idx=fi)
        rr.set_time("frame", sequence=fi)

        h, w = depth.shape
        rr.log("world/camera", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[w, h],
                                          camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.6))
        cams.append(c2w[:3, 3].copy())

        # 2D panel: three stacked layers (rgb on top by default, depth beneath, obb over both)
        rr.log("camera/image", rr.Image(rgb, draw_order=1.0))
        rr.log("camera/image/depth", rr.DepthImage(
            np.where(np.isfinite(depth) & (depth > 0), depth, 0.0), meter=1.0, draw_order=0.0))
        by_id = {o["track_id"]: o for o in recon.pipe.objects()}
        matched = [tid for tid in {int(t) for t in info.get("match_tids", [])} if tid in by_id]

        # 3D world + 2D obb are driven by the SAME matched set so the two panels always
        # agree -- every track matched this frame gets both its projected 2D wireframe and
        # its live 3D box, drawn EVERY frame so you watch objects spawn/move online. No
        # conf/dynamic gate here: that filtering is a finalize decision, applied at the end
        # (dropped tracks are Cleared, confirmed movers turn red).
        strips, scols = [], []
        for tid in matched:
            o = by_id[tid]
            col = colorize_labels(np.array([tid]))[0]
            draw_box(f"world/objects/{tid}", o, col)
            edges = project_obb(o["center"], o["size"], o["quat_xyzw"], K, c2w)
            if edges:
                strips += edges
                scols += [col] * len(edges)
        rr.log("camera/image/obb",
               rr.LineStrips2D(strips, colors=scols, radii=1.0, draw_order=2.0) if strips
               else rr.Clear(recursive=False))

        if len(cams) > 1:                                # grow the camera trajectory
            rr.log("world/path", rr.LineStrips3D([np.asarray(cams)], colors=[(255, 0, 255)]))

        if k % args.snap == 0:                           # scene cloud is heavy -> snapshot only
            log_scene(fi)
            dt = time.perf_counter() - t0
            print(f"  frame {fi}: {len(by_id)} live objects  [{dt:.0f}s, {dt / k:.2f}s/frame]", flush=True)

    result = recon.finalize(args.traj, source=str(traj._dir("image")))
    dt = time.perf_counter() - t0
    print(f"PERF: {len(idxs)} frames in {dt:.1f}s = {dt / len(idxs):.2f}s/frame", flush=True)

    # final 3D: scene cloud, static objects (colored) and dropped movers (red) -- same
    # draw_box path as live. Crops are refs now (render-only); the recon view skips them --
    # they're resolved in the merged view. The dumped .npz keeps the refs + source path.
    def as_dict(o):  # SceneObject -> the box dict draw_box expects (id key = persistent uid)
        return {"track_id": o.uid, "center": o.center, "size": o.half_size,
                "quat_xyzw": o.quat_xyzw, "conf": o.conf, "support": o.support,
                "labels": o.labels, "crops": []}

    log_scene(idxs[-1])
    rr.set_time("frame", sequence=idxs[-1])
    kept = {o.uid for o in result.objects}
    # the last (settled) frame shows only the finalize result: Clear the live boxes that
    # finalize dropped (low-conf / contained / reclassified as dynamic) so scrubbing to the
    # end gives the clean map; confirmed movers reappear on the red world/dynamic path.
    for ent in list(annotated):
        if ent.startswith("world/objects/") and int(ent.rsplit("/", 1)[1]) not in kept:
            rr.log(ent, rr.Clear(recursive=True))
    cols = colorize_labels(np.array([o.uid for o in result.objects])) if result.objects else []
    for o, col in zip(result.objects, cols):
        draw_box(f"world/objects/{o.uid}", as_dict(o), col, refresh=True)
    for o in result.dynamic:
        draw_box(f"world/dynamic/{o.uid}", as_dict(o), (255, 0, 0), dyn=True, refresh=True)
    if len(cams) > 1:
        rr.log("world/path", rr.LineStrips3D([np.asarray(cams)], colors=[(255, 0, 255)]))

    # now that every box entity path is known, re-send the blueprint with a per-entity
    # visualizer override so the 3D view draws boxes only (crops remain sidebar-only) -> the
    # 'requires a pinhole' warning is gone.
    rr.send_blueprint(make_bp({ent: rrb.VisualizerOverrides(["Boxes3D"]) for ent in annotated
                               if ent.startswith(("world/objects/", "world/dynamic/"))}))

    print(f"FINAL: {len(result.objects)} static objects, {len(result.dynamic)} dynamic dropped "
          f"| scene cloud: {len(result.points)} voxels", flush=True)
    confs = np.array([o.conf for o in result.objects], np.float32)
    if len(confs):
        q = np.percentile(confs, [0, 25, 50, 75, 100])
        print(f"CONF: mean={confs.mean():.3f} std={confs.std():.3f} | min={q[0]:.3f} "
              f"p25={q[1]:.3f} med={q[2]:.3f} p75={q[3]:.3f} max={q[4]:.3f}", flush=True)
        edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
        hist, _ = np.histogram(confs, bins=edges)
        print("  hist " + "  ".join(f"[{edges[i]:.1f},{edges[i + 1]:.1f}):{hist[i]}"
                                    for i in range(len(hist))), flush=True)
    has_ref = sum(1 for o in result.objects if any(s.get("crops") for s in o.sessions.values()))
    print(f"CROP-REFS: {has_ref}/{len(result.objects)} objects have crop refs "
          f"({len(result.objects) - has_ref} without)", flush=True)
    for o in sorted(result.objects, key=lambda o: -o.conf)[:15]:
        nref = sum(len(s.get("crops", [])) for s in o.sessions.values())
        print(f"  obj {o.uid:4d} conf={o.conf:.3f} support={o.support:3d} "
              f"crop_refs={nref}  {o.labels}", flush=True)

    out = args.out or (traj.base / "recon" / f"testrecon_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}\nview with: rerun {out}", flush=True)
    if args.dump:                     # full SceneMap (objects + cloud + crop-ref provenance) for Tier-2
        from herald.scene.refine import save_session
        save_session(args.dump, result)
        print(f"dumped session (objects + cloud + provenance): {args.dump}", flush=True)


if __name__ == "__main__":
    main()
