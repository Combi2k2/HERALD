#!/usr/bin/env python3
"""Sanity-check SAM2 on a TartanGround trajectory: segment the RGB stream and
render the per-frame instance masks next to the RGB in rerun."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundTraj
from herald.scene.recon.sam2 import Sam2Stream

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=32, help="Frames per SAM2 chunk")
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--min-area", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=32, help="0=all frames (processed chunk by chunk)")
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

    stream = Sam2Stream(args.model, device=args.device, chunk=args.chunk,
                        points_per_side=args.points_per_side, min_area=args.min_area)

    rr.init("herald.test_sam2")
    print(f"sam2: segmenting {len(indices)} frames (chunk={args.chunk})", flush=True)
    ids: set[int] = set()
    rgbs = (traj.load_rgb(i) for i in indices)
    for k, lab in enumerate(stream.run_stream(rgbs)):
        ids.update(int(i) for i in np.unique(lab) if i)
        rr.set_time("frame", sequence=indices[k])
        rr.log("camera/rgb", rr.Image(traj.load_rgb(indices[k])))
        rr.log("camera/seg", rr.SegmentationImage(lab))
        if (k + 1) % args.chunk == 0 or k == len(indices) - 1:
            print(f"  {k + 1}/{len(indices)} frames, {len(ids)} objects", flush=True)

    out = args.out or (traj.base / "recon" / f"sam2_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"objects: {len(ids)}", flush=True)
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)

if __name__ == "__main__":
    main()
