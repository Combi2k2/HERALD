#!/usr/bin/env python3
"""Sanity-check VGGT-Omega on a TartanGround trajectory: reconstruct a colored
point cloud from predicted depth + poses and render it in rerun."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundTraj

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--checkpoint", required=True, type=Path, help="VGGT-Omega checkpoint")
    p.add_argument("--image-resolution", type=int, default=512, help="Must be divisible by 16 (VGGT-Omega patch size)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=32, help="VGGT runs one batch over all frames")
    p.add_argument("--pixel-stride", type=int, default=4)
    p.add_argument("--max-depth", type=float, default=0.0, help="Clip depth (0=off; VGGT depth is up-to-scale)")
    p.add_argument("--conf-thresh", type=float, default=0.0, help="Min depth confidence (0=off)")
    p.add_argument("--radius", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=None, help="Output .rrd")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    indices = list(range(len(traj)))[:: max(1, args.stride)]
    if args.max_frames:
        indices = indices[: args.max_frames]
    paths = [traj.rgb_path(i) for i in indices]
    if not paths:
        raise SystemExit(f"no frames under {traj.base}")

    from herald.scene.recon import vggt

    print(f"vggt: inferring geometry for {len(paths)} frames", flush=True)
    model = vggt.load_model(str(args.checkpoint), args.device)
    geo = vggt.vggt_infer(paths, args.image_resolution, model)
    s, h, w = geo["depth"].shape
    print(f"  depth {s}x{h}x{w}", flush=True)

    import rerun as rr
    from PIL import Image

    max_depth = args.max_depth if args.max_depth > 0 else np.inf
    st = max(1, args.pixel_stride)

    rr.init("herald.test_vggt")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    centers: list[np.ndarray] = []
    for k in range(s):
        K, c2w, depth = geo["K"][k], geo["c2w"][k], geo["depth"][k]
        rgb = np.asarray(Image.open(paths[k]).convert("RGB").resize((w, h)))

        d = depth[::st, ::st]
        vs, us = np.mgrid[0:h:st, 0:w:st]
        valid = np.isfinite(d) & (d > 0) & (d < max_depth)
        if args.conf_thresh > 0:
            valid &= geo["conf"][k][::st, ::st] >= args.conf_thresh
        z, u, v = d[valid], us[valid], vs[valid]
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
        pts = cam @ c2w[:3, :3].T + c2w[:3, 3]
        cols = rgb[::st, ::st][valid]

        rr.set_time("frame", sequence=indices[k])
        rr.log("world/camera", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(
            image_from_camera=K, resolution=[w, h], camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/image/rgb", rr.Image(rgb))
        rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1.0))
        rr.log(f"world/cloud/{k:04d}", rr.Points3D(pts, colors=cols, radii=args.radius), static=True)
        centers.append(c2w[:3, 3])
        if len(centers) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers)], colors=[(255, 0, 255)]))

    out = args.out or (traj.base / "recon" / f"vggt_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)

if __name__ == "__main__":
    main()
