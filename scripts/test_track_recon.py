#!/usr/bin/env python3
"""SAM2-tracking + 3D-merge reconstruction on TartanGround -- the flicker-
controlled, neither-over-nor-under-fusing path.

SAM2 *video* tracking (chunked, re-seeded per chunk, FRESH local ids each chunk)
gives within-chunk identity; TrackReconStream matches those local objects across
chunks by 3D voxel co-occupancy (overlap, not adjacency) and remaps them to
persistent global ids; VoteCloud accumulates the global labels into a
voxel-majority-vote cloud with the flicker-prone pixels dropped (frame-border
margin, depth discontinuities, eroded edges, confidence floor). No appearance
gate, no union-find -- overlap-merge keeps distinct objects apart (no floor
bridge) while stitching a surface's per-chunk fragments into one id. GT
depth/pose from TartanGround supplies geometry. Renders the cloud evolving in
rerun (colored by global id), so scrubbing shows it build up.

Needs a real CUDA GPU on the 3090 partition (scripts/test_track_recon.slurm);
the P100 crashes cu130 torch.
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundGeometry, TartanGroundTraj
from herald.scene.recon.track import Sam2Tracker, TrackReconStream, VoteCloud
from herald.viz import colorize_labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0=all frames")
    p.add_argument("--pixel-stride", type=int, default=2, help="cloud pixel subsampling")
    p.add_argument("--voxel", type=float, default=0.05, help="cloud voxel size (smaller=more detail)")
    p.add_argument("--max-depth", type=float, default=60.0, help="drop pixels farther than this")
    # --- SAM2 video tracking (chunked) ---
    p.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--chunk", type=int, default=16, help="frames per SAM2 session (bounds memory)")
    p.add_argument("--points-per-side", type=int, default=16,
                   help="AMG seed grid (per chunk); higher=finer masks")
    p.add_argument("--min-area", type=int, default=400, help="drop AMG masks smaller than this")
    p.add_argument("--no-offload", action="store_true", help="keep SAM2 state on GPU (uses more VRAM)")
    # --- cross-chunk 3D voxel-overlap merge ---
    p.add_argument("--merge-voxel", type=float, default=0.1,
                   help="world voxel size for cross-chunk co-occupancy matching")
    p.add_argument("--merge-overlap", type=float, default=0.2,
                   help="min fraction of a chunk object's voxels on a global object to merge")
    # --- flicker control / voting ---
    p.add_argument("--border", type=int, default=12, help="drop pixels within N px of the frame edge")
    p.add_argument("--edge-rtol", type=float, default=0.03, help="depth-discontinuity filter (0=off)")
    p.add_argument("--erode", type=int, default=1, help="erode mask labels by N px before voting")
    p.add_argument("--sem-conf", type=float, default=0.0, help="min SAM2 mask confidence to vote (0=off)")
    p.add_argument("--min-cluster", type=int, default=10, help="drop DBSCAN clusters smaller than this")
    p.add_argument("--min-frames", type=int, default=1, help="drop tracks seen in fewer frames")
    p.add_argument("--device", default="cuda")
    p.add_argument("--snapshot-every", type=int, default=32, help="flush + log the cloud every N frames")
    p.add_argument("--point-radius", type=float, default=0.02, help="rerun point display radius")
    p.add_argument("--out", type=Path, default=None, help="output .rrd")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    indices = list(range(len(traj)))[:: max(1, args.frame_stride)]
    if args.max_frames:
        indices = indices[: args.max_frames]
    if not indices:
        raise SystemExit(f"no frames under {traj.base}")

    geo = TartanGroundGeometry(traj)
    tracker = Sam2Tracker(args.sam2_model, device=args.device, chunk=args.chunk,
                          points_per_side=args.points_per_side, min_area=args.min_area,
                          offload=not args.no_offload)
    vote = VoteCloud(voxel=args.voxel, stride=args.pixel_stride, max_depth=args.max_depth,
                     sem_conf_thresh=args.sem_conf, edge_rtol=args.edge_rtol,
                     erode=args.erode, border=args.border)
    stream = TrackReconStream(tracker, vote, merge_voxel=args.merge_voxel,
                              merge_overlap=args.merge_overlap, max_depth=args.max_depth)

    import rerun as rr

    rr.init("herald.test_track_recon")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # TartanGround GT poses are NED (Z down)

    print(f"{args.env}/{args.traj}: tracking {len(indices)} frames "
          f"(chunk={args.chunk}, pps={args.points_per_side}, voxel={args.voxel})", flush=True)
    flush_kw = dict(min_cluster=args.min_cluster, min_frames=args.min_frames)
    t_start = time.perf_counter()
    centers: list[np.ndarray] = []
    state = {"done": 0}

    def on_frame(fi: int, g, rgb: np.ndarray, label: np.ndarray, conf: np.ndarray) -> None:
        rr.set_time("frame", sequence=fi)
        h, w = g.depth.shape
        rr.log("world/camera", rr.Transform3D(translation=g.c2w[:3, 3], mat3x3=g.c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=g.K, resolution=[w, h],
                                                camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/image/rgb", rr.Image(rgb))
        rr.log("world/camera/image/seg", rr.Image(colorize_labels(label.astype(np.int64))))
        centers.append(g.c2w[:3, 3].copy())
        if len(centers) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers)], colors=[(255, 0, 255)]))
        state["done"] += 1
        if args.snapshot_every and state["done"] % args.snapshot_every == 0:
            cloud = vote.flush(frame=fi, **flush_kw)
            rr.log("world/cloud", rr.Points3D(cloud["points"],
                                              colors=colorize_labels(cloud["labels"]),
                                              radii=args.point_radius))
            dt = time.perf_counter() - t_start
            print(f"  frame {fi}: {len(cloud['obj_labels'])} objects, {len(cloud['points'])} points"
                  f"  [{dt:.0f}s, {dt / state['done']:.2f}s/frame]", flush=True)

    for i in indices:
        stream.push(i, geo.geometry(i), traj.load_rgb(i), on_frame)
    stream.finish(on_frame)

    cloud = vote.flush(frame=indices[-1], **flush_kw)
    rr.set_time("frame", sequence=indices[-1])
    rr.log("world/cloud", rr.Points3D(cloud["points"], colors=colorize_labels(cloud["labels"]),
                                      radii=args.point_radius))
    print(f"  final: {len(cloud['obj_labels'])} objects, {len(cloud['points'])} points", flush=True)

    elapsed = time.perf_counter() - t_start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # KiB -> GiB on Linux
    n = len(indices)
    print(f"PERF: {n} frames in {elapsed:.1f}s = {elapsed / n:.2f}s/frame "
          f"({n / elapsed:.2f} fps) | peak host RSS {peak_gb:.1f} GiB", flush=True)

    out = args.out or (traj.base / "recon" / f"track_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)


if __name__ == "__main__":
    main()
