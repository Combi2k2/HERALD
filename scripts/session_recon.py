#!/usr/bin/env python3
"""Tier-1 per-session recon: a TartanGround video -> static scene cloud + static
object OBBs, rendered live to Rerun.

Streams frames through herald.scene.recon.SessionRecon (Boxer detect -> 3D lift ->
Hungarian tracker), builds the original-color scene cloud on the fly, and at the
end drops in-video movers via the corridor filter. Static objects render in a
per-object palette (labelled, with a few RGB crops); the filtered-out dynamic
objects render separately in red so you can see what was removed.

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
from herald.scene.recon.utils import object_crops
from herald.viz import colorize_labels


def tag(o) -> str:
    """Aggregated labels of an object, e.g. 'bench x224, chair x74'."""
    items = sorted(o.labels.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} x{v}" for k, v in items[:4])


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
    p.add_argument("--min-visible", type=float, default=0.0,
                   help="per-frame box-quality gate: min fraction of the reprojected 3D OBB inside "
                        "the frame to trust a box (0 disables; e.g. 0.6 drops truncated edge lifts)")
    p.add_argument("--scene-voxel", type=float, default=0.12)
    p.add_argument("--scene-stride", type=int, default=8)
    p.add_argument("--snap", type=int, default=128, help="frames between live snapshots")
    p.add_argument("--n-crops", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dump", type=Path, default=None,
                   help="also save the SessionResult (objects + cloud) to this .npz for Tier-2 refine")
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
                         corridor_step=args.corridor_step, max_range=args.max_range,
                         min_visible=args.min_visible)

    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("herald.session_recon")
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="scene"),
            rrb.Spatial2DView(origin="camera/image", name="camera"),
            column_shares=[3, 1]),
        rrb.SelectionPanel(state="expanded"),
        auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)   # TartanGround NED
    print(f"{args.env}/{args.traj}: OWLv2+Boxer session-recon over {len(idxs)} frames", flush=True)

    def log_scene(fi):
        sp, sc = recon.scene.cloud()
        if len(sp):
            rr.set_time("frame", sequence=fi)
            rr.log("world/scene", rr.Points3D(sp, colors=sc, radii=0.01))

    t0 = time.perf_counter()
    cams: list[np.ndarray] = []
    for k, (fi, rgb, K, c2w, depth) in enumerate(src.frames(idxs), 1):
        info = recon.add_frame(rgb, K, c2w, depth, frame_idx=fi)

        rr.set_time("frame", sequence=fi)
        h, w = depth.shape
        rr.log("world/camera", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[w, h],
                                          camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.6))
        rr.log("camera/image", rr.Image(rgb))
        if len(info["boxes"]):
            b = info["boxes"]
            rr.log("camera/image/bbox", rr.Boxes2D(
                mins=b[:, :2], sizes=b[:, 2:] - b[:, :2], labels=info["labels"]))
        else:
            rr.log("camera/image/bbox", rr.Clear(recursive=False))
        cams.append(c2w[:3, 3].copy())

        if k % args.snap == 0:
            log_scene(fi)
            dt = time.perf_counter() - t0
            print(f"  frame {fi}: {len(recon.pipe.objects())} live objects  "
                  f"[{dt:.0f}s, {dt / k:.2f}s/frame]", flush=True)

    result = recon.finalize(args.traj, source=str(traj._dir("image")))
    dt = time.perf_counter() - t0
    print(f"PERF: {len(idxs)} frames in {dt:.1f}s = {dt / len(idxs):.2f}s/frame", flush=True)

    # final scene + static objects (colored, labelled, crop thumbnails) + dynamic (red)
    log_scene(idxs[-1])
    rr.set_time("frame", sequence=idxs[-1])
    cols = colorize_labels(np.array([o.uid for o in result.objects])) if result.objects else []
    for o, col in zip(result.objects, cols):
        ent = f"world/objects/{o.uid}"
        rr.log(ent, rr.Boxes3D(centers=[o.center], half_sizes=[o.half_size],
                               quaternions=[o.quat_xyzw], colors=[col], fill_mode="majorwireframe"))
        rr.log(ent, rr.TextDocument(f"id {o.uid} | support {o.support} | conf {o.conf:.2f}\n{tag(o)}"),
               static=True)
        crops = object_crops(o, result.meta.get("sources"))   # resolve crop refs -> thumbnails
        if crops:
            rr.log(ent, rr.Image(np.concatenate(crops, axis=1)), static=True)
    for o in result.dynamic:
        rr.log(f"world/dynamic/{o.uid}", rr.Boxes3D(
            centers=[o.center], half_sizes=[o.half_size], quaternions=[o.quat_xyzw],
            colors=[(255, 0, 0)], fill_mode="majorwireframe", labels=[o.label]))
    if len(cams) > 1:
        rr.log("world/path", rr.LineStrips3D([np.asarray(cams)], colors=[(255, 0, 255)]))

    print(f"FINAL: {len(result.objects)} static objects, {len(result.dynamic)} dynamic dropped "
          f"| scene cloud: {len(result.points)} voxels", flush=True)
    for o in sorted(result.objects, key=lambda o: -o.support)[:15]:
        print(f"  obj {o.uid:4d} support={o.support:3d} conf={o.conf:.2f}  {o.labels}", flush=True)

    out = args.out or (traj.base / "recon" / f"session_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}\nview with: rerun {out}", flush=True)

    if args.dump:
        from herald.scene.refine import save_session
        save_session(args.dump, result)
        print(f"dumped session (objects + cloud): {args.dump}", flush=True)


if __name__ == "__main__":
    main()
