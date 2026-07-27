#!/usr/bin/env python3
"""Sanity-check SAM2 on a TartanGround trajectory: segment the RGB stream into
per-frame instance masks and render them (as a segmentation image) next to the
RGB in rerun."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundTraj
from herald.scene.recon.sam2 import Sam2Segmenter

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--device", default="cuda")
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--min-area", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=32, help="0=all frames")
    p.add_argument("--out", type=Path, default=None, help="Output .rrd")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    indices = list(range(len(traj)))[:: max(1, args.stride)]
    if args.max_frames:
        indices = indices[: args.max_frames]
    if not indices:
        raise SystemExit(f"no frames under {traj.base}")

    import rerun as rr

    seg = Sam2Segmenter(args.model, device=args.device,
                        points_per_side=args.points_per_side, min_area=args.min_area)

    rr.init("herald.test_sam2")
    print(f"sam2: segmenting {len(indices)} frames", flush=True)
    total = 0
    for k, i in enumerate(indices):
        masks = seg.segment(traj.load_rgb(i))
        total += len(masks)
        # paint a per-frame label image: mask index+1, larger masks first
        lab = np.zeros(masks[0].shape, np.int32) if masks else np.zeros((1, 1), np.int32)
        for j, m in enumerate(masks):
            lab[m] = j + 1
        rr.set_time("frame", sequence=i)
        rr.log("camera/rgb", rr.Image(traj.load_rgb(i)))
        rr.log("camera/seg", rr.SegmentationImage(lab))
        print(f"  {k + 1}/{len(indices)} frames, {len(masks)} masks this frame", flush=True)

    out = args.out or (traj.base / "recon" / f"sam2_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"total masks over {len(indices)} frames: {total}", flush=True)
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)

if __name__ == "__main__":
    main()
