#!/usr/bin/env python3
"""Fuse TartanGround GT geometry + GT segmentation into a labeled cloud and
render the reconstruction *evolving* in rerun: the camera walks the trajectory
while the Fuser is flushed every few frames, so scrubbing the timeline shows
the semantic point cloud build up."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundGeometry, TartanGroundSeg, TartanGroundStoredSeg, TartanGroundTraj
from herald.scene.recon import Fuser
from herald.viz import colorize_labels

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--seg", default="gt", help="gt or a stored modality dir, e.g. seg_sam2")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0=all frames")
    p.add_argument("--pixel-stride", type=int, default=2, help="Fuser pixel subsampling")
    p.add_argument("--voxel", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=60.0)
    p.add_argument("--erode", type=int, default=1)
    p.add_argument("--min-cluster", type=int, default=10)
    p.add_argument("--min-hits", type=int, default=1)
    p.add_argument("--ignore-background", action="store_true", help="Drop label 0")
    p.add_argument("--snapshot-every", type=int, default=8, help="Flush + log the cloud every N frames")
    p.add_argument("--radius", type=float, default=0.03)
    p.add_argument("--out", type=Path, default=None, help="Output .rrd")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    indices = list(range(len(traj)))[:: max(1, args.frame_stride)]
    if args.max_frames:
        indices = indices[: args.max_frames]
    if not indices:
        raise SystemExit(f"no frames under {traj.base}")

    geo = TartanGroundGeometry(traj)
    seg = TartanGroundSeg(traj) if args.seg == "gt" else TartanGroundStoredSeg(traj, args.seg)

    fuser = Fuser(voxel=args.voxel, stride=args.pixel_stride, max_depth=args.max_depth,
                  erode=args.erode, ignore_ids=(0,) if args.ignore_background else ())

    import rerun as rr

    rr.init("herald.fuse_gt")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # TartanGround GT poses are NED (Z down)

    print(f"{args.env}/{args.traj}: fusing {len(indices)} frames (seg={seg.name})", flush=True)
    centers: list[np.ndarray] = []
    for step, i in enumerate(indices):
        g = geo.geometry(i)
        fuser.push(g, seg.labels(i))

        rr.set_time("frame", sequence=i)
        h, w = g.depth.shape
        rr.log("world/camera", rr.Transform3D(translation=g.c2w[:3, 3], mat3x3=g.c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=g.K, resolution=[w, h],
                                                camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/image/rgb", rr.Image(traj.load_rgb(i)))
        centers.append(g.c2w[:3, 3].copy())
        if len(centers) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers)], colors=[(255, 0, 255)]))

        last = step == len(indices) - 1
        if last or (args.snapshot_every and (step + 1) % args.snapshot_every == 0):
            cloud = fuser.flush(frame=i, min_cluster=args.min_cluster, min_hits=args.min_hits)
            pts = cloud["points"]
            rr.log("world/cloud", rr.Points3D(pts, colors=colorize_labels(cloud["labels"]),
                                              radii=args.radius))
            print(f"  frame {i}: {len(cloud['obj_labels'])} objects, {len(pts)} points", flush=True)

    out = args.out or (traj.base / "recon" / f"fuse_{args.env}_{args.traj}_{seg.name}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)

if __name__ == "__main__":
    main()
