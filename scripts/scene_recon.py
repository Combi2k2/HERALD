#!/usr/bin/env python3
"""scene_recon: reconstruct one TartanGround video into a static scene cloud + object OBBs.

Streams frames through herald.scene.recon.SessionRecon (Boxer detect -> 3D lift -> tracker),
builds the coloured cloud, drops in-video movers via the corridor filter, and writes the
SessionResult (objects + cloud) to --out. No rendering -- view with the render script.

Needs a CUDA GPU (scripts/scene_recon.slurm).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

DEFAULT_VOCAB = [
    "person", "chair", "wheelchair", "bed", "hospital bed", "stretcher", "table",
    "desk", "cabinet", "shelf", "counter", "sink", "toilet", "door", "window",
    "curtain", "monitor", "screen", "tv", "computer", "laptop", "keyboard",
    "telephone", "medical equipment", "iv stand", "ventilator", "oxygen tank",
    "trash can", "box", "cart", "trolley", "stool", "sofa", "couch", "bench",
    "lamp", "light", "sign", "fire extinguisher", "clock", "plant", "bottle",
    "cup", "bag", "pillow", "blanket", "towel", "mirror", "whiteboard", "book",
    "basket", "bucket", "ladder", "backpack", "fan", "picture frame",
    "stair", "entrance",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--out", type=Path, required=True, help="SessionResult .npz (objects + cloud)")
    p.add_argument("--version", default="omni")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--conf-thr2d", type=float, default=0.40)
    p.add_argument("--conf-thr3d", type=float, default=0.50)
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--min-obs", type=int, default=4)
    p.add_argument("--max-depth", type=float, default=60.0)
    p.add_argument("--corridor-step", type=float, default=0.2)
    p.add_argument("--max-range", type=float, default=15.0)
    p.add_argument("--min-visible", type=float, default=0.0)
    p.add_argument("--scene-voxel", type=float, default=0.12)
    p.add_argument("--scene-stride", type=int, default=8)
    p.add_argument("--n-crops", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from herald.datasets import TartanGroundGeometry, TartanGroundTraj
    from herald.scene.recon import GtGeometry, SessionRecon
    from herald.scene.refine import save_session

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

    print(f"{args.env}/{args.traj}: recon over {len(idxs)} frames", flush=True)
    t0 = time.perf_counter()
    for k, (fi, rgb, K, c2w, depth) in enumerate(src.frames(idxs), 1):
        recon.add_frame(rgb, K, c2w, depth, frame_idx=fi)
        if k % 256 == 0:
            dt = time.perf_counter() - t0
            print(f"  frame {fi}: {len(recon.pipe.objects())} live  [{dt:.0f}s, {dt / k:.2f}s/frame]", flush=True)

    result = recon.finalize(args.traj, source=str(traj._dir("image")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_session(args.out, result)
    print(f"FINAL: {len(result.objects)} static objects, {len(result.dynamic)} dynamic dropped, "
          f"{len(result.points)} voxels -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
