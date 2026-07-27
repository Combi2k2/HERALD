#!/usr/bin/env python3
"""End-to-end RGB(D)-stream reconstruction on TartanGround.

Unlike fuse_gt.py (which feeds the Fuser GT geometry *and* GT segmentation),
this runs the real streaming pipeline: every RGB frame is segmented into
instance masks by Sam2Segmenter, each mask is embedded with SigLIP and
associated into a class-agnostic object by appearance + occupancy (the Fuser),
GT depth/pose from TartanGround supplies the geometry, and ReconStream drains
it all into the Fuser. It renders the reconstruction *evolving* in rerun, so
scrubbing the timeline shows the semantic cloud build up.

This is the "rgbd" mode: depth is provided by the dataset. Swap the geometry
source for a VggtStream (pass it to ReconStream(vggt=...) and push RGB paths
with geo=None) to run fully monocular from RGB alone.
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundGeometry, TartanGroundTraj
from herald.scene.recon import Fuser, Sam2Segmenter, SiglipEmbedder
from herald.scene.recon.pipeline import ReconStream
from herald.scene.recon.utils import masks_to_labels
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
    p.add_argument("--pixel-stride", type=int, default=2, help="Fuser pixel subsampling")
    p.add_argument("--voxel", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=60.0, help="Drop pixels farther than this")
    p.add_argument("--erode", type=int, default=1)
    # --- SAM2 (per-frame automatic mask generation) ---
    p.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus",
                   help="HF SAM2 checkpoint (e.g. sam2.1-hiera-tiny / -base-plus / -large)")
    p.add_argument("--points-per-side", type=int, default=8,
                   help="SAM2 auto-mask seed grid; NxN prompts. Fewer => fewer masks, much faster")
    p.add_argument("--min-area", type=int, default=800, help="Drop masks smaller than this many px")
    # --- SigLIP appearance embedder ---
    p.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    # --- Fuser association ---
    p.add_argument("--radius", type=float, default=1.0, help="AABB prefilter slack (m)")
    p.add_argument("--min-overlap", type=float, default=0.2,
                   help="Min fraction of a mask's voxels overlapping an object to associate")
    p.add_argument("--dilate", type=int, default=1, help="Voxel dilation (cells) for the overlap test")
    p.add_argument("--bg-fade", type=float, default=0.5,
                   help="Fade non-mask crop pixels toward white before embedding (0 = raw crop)")
    p.add_argument("--min-csim", type=float, default=0.5, help="Min embedding cosine to associate")
    p.add_argument("--ema", type=float, default=0.2, help="Object embedding update rate on a match")
    p.add_argument("--merge-dilate", type=int, default=2, help="Voxel dilation for the object-merge pass")
    p.add_argument("--merge-csim", type=float, default=0.5, help="Min cosine to merge two existing objects")
    p.add_argument("--merge-overlap", type=float, default=0.25,
                   help="Min contact (frac of smaller object's voxels) to merge; low collapses scenes")
    # --- flush filtering ---
    p.add_argument("--min-obs", type=int, default=1, help="Drop objects seen in fewer frames")
    p.add_argument("--min-points", type=int, default=10, help="Drop objects with fewer voxels")
    p.add_argument("--device", default="cuda", help="SAM2 / SigLIP device (cuda/cpu)")
    p.add_argument("--snapshot-every", type=int, default=8, help="Flush + log the cloud every N frames")
    p.add_argument("--point-radius", type=float, default=0.03, help="rerun point display radius")
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
    sam2 = Sam2Segmenter(args.sam2_model, device=args.device,
                         points_per_side=args.points_per_side, min_area=args.min_area)
    embedder = SiglipEmbedder(args.siglip_model, device=args.device)
    fuser = Fuser(embedder, voxel=args.voxel, stride=args.pixel_stride, max_depth=args.max_depth,
                  erode=args.erode, radius=args.radius, min_overlap=args.min_overlap,
                  dilate=args.dilate, min_area=args.min_area, bg_fade=args.bg_fade,
                  min_csim=args.min_csim, ema=args.ema,
                  merge_dilate=args.merge_dilate, merge_csim=args.merge_csim,
                  merge_overlap=args.merge_overlap)
    recon = ReconStream(sam2, fuser)

    import rerun as rr

    rr.init("herald.test_recon")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # TartanGround GT poses are NED (Z down)

    print(f"{args.env}/{args.traj}: streaming {len(indices)} frames "
          f"(SAM2 {args.sam2_model} + SigLIP on {args.device})", flush=True)
    flush_kw = dict(min_obs=args.min_obs, min_points=args.min_points)
    t_start = time.perf_counter()
    centers: list[np.ndarray] = []
    for step, i in enumerate(indices):
        g = geo.geometry(i)
        rgb = traj.load_rgb(i)
        # segment once, reuse the masks both to render and to fuse (no double work)
        masks = sam2.segment(rgb)
        recon.push(rgb, g, masks=masks)

        rr.set_time("frame", sequence=i)
        h, w = g.depth.shape
        rr.log("world/camera", rr.Transform3D(translation=g.c2w[:3, 3], mat3x3=g.c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=g.K, resolution=[w, h],
                                                camera_xyz=rr.ViewCoordinates.RDF))
        # rgb + depth + semantic-mask streams, all under the pinhole so they
        # overlay in the 2D view and back-project in 3D; toggle each in the
        # entity tree / blueprint.
        rr.log("world/camera/image/rgb", rr.Image(rgb))
        # Viridis: perceptually monotonic (near=dark blue -> far=yellow), unlike
        # rerun's default Turbo rainbow which reads warm->cold->warm with depth.
        rr.log("world/camera/image/depth",
               rr.DepthImage(g.depth, meter=1.0, colormap=rr.components.Colormap.Viridis))
        # seg colored by the mask's STABLE object color id (same palette as the
        # cloud), so each object keeps one color across frames — no flicker.
        seg = (masks_to_labels(recon.fuser.last_masks, values=recon.fuser.last_cids)
               if recon.fuser.last_masks else np.zeros(rgb.shape[:2], np.int32))
        rr.log("world/camera/image/seg", rr.Image(colorize_labels(seg)))
        centers.append(g.c2w[:3, 3].copy())
        if len(centers) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers)], colors=[(255, 0, 255)]))

        last = step == len(indices) - 1
        if last or (args.snapshot_every and (step + 1) % args.snapshot_every == 0):
            cloud = recon.flush(frame=i, **flush_kw)
            pts = cloud["points"]
            rr.log("world/cloud", rr.Points3D(pts, colors=colorize_labels(cloud["labels"]),
                                              radii=args.point_radius))
            dt = time.perf_counter() - t_start
            print(f"  frame {i}: {len(cloud['obj_labels'])} objects, {len(pts)} points"
                  f"  [{dt:.0f}s, {dt / (step + 1):.2f}s/frame]", flush=True)

    cloud = recon.finish(frame=indices[-1], **flush_kw)
    rr.set_time("frame", sequence=indices[-1])
    rr.log("world/cloud", rr.Points3D(cloud["points"], colors=colorize_labels(cloud["labels"]),
                                      radii=args.point_radius))
    print(f"  final: {len(cloud['obj_labels'])} objects, {len(cloud['points'])} points", flush=True)

    elapsed = time.perf_counter() - t_start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # KiB -> GiB on Linux
    n = len(indices)
    print(f"PERF: {n} frames in {elapsed:.1f}s = {elapsed / n:.2f}s/frame "
          f"({n / elapsed:.2f} fps) | peak host RSS {peak_gb:.1f} GiB", flush=True)

    out = args.out or (traj.base / "recon" / f"recon_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)

if __name__ == "__main__":
    main()
