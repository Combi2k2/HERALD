#!/usr/bin/env python3
"""Build a session scene cloud straight from RGB-D (no Boxer) and save PLY + npz.

Tier-2 refine input builder: unprojects every (strided) frame's valid depth into the
NED world frame and voxel-accumulates a colored cloud (herald.scene.recon.utils.SceneCloud).
No detector/tracker -- we only need geometry to align two sessions.

  uv run python scripts/build_cloud.py --traj P0000 --out data/tartanground/Hospital/recon/P0000.ply
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundGeometry, TartanGroundTraj
from herald.scene.recon import GtGeometry
from herald.scene.recon.utils import SceneCloud, unproject_rgb, write_ply


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", default="Hospital")
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", required=True)
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--stride", type=int, default=2, help="frame stride")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-depth", type=float, default=40.0)
    p.add_argument("--pixel-stride", type=int, default=4, help="unprojection pixel subsample")
    p.add_argument("--voxel", type=float, default=0.1)
    p.add_argument("--out", type=Path, required=True, help="output .ply (npz written alongside)")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    idxs = list(range(len(traj)))[:: max(1, args.stride)]
    if args.max_frames:
        idxs = idxs[: args.max_frames]
    src = GtGeometry(traj, TartanGroundGeometry(traj))
    scene = SceneCloud(voxel=args.voxel)

    t0 = time.perf_counter()
    for k, (fi, rgb, K, c2w, depth) in enumerate(src.frames(idxs), 1):
        valid = np.isfinite(depth) & (depth > 0) & (depth < args.max_depth)
        pts, cols = unproject_rgb(depth, K, c2w, rgb, args.pixel_stride, valid)
        scene.add(pts, cols)
        if k % 100 == 0:
            print(f"  {k}/{len(idxs)} frames  [{time.perf_counter()-t0:.0f}s]", flush=True)

    pts, cols = scene.cloud()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.out, pts, cols)
    np.savez_compressed(args.out.with_suffix(".npz"),
                        points=pts.astype(np.float32), colors=cols.astype(np.uint8))
    print(f"{args.env}/{args.traj}: {len(pts)} voxels  ->  {args.out}  (+ .npz)  "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
