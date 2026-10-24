#!/usr/bin/env python
"""scene_align: Sim3 that integrates a new recon session into the persistent scene map.

Aligns the new session's cloud onto the persistent map's cloud (energy-based Sim3). Identity
when there is no persistent map yet (first session) or --no-align (pre-registered poses, e.g.
TartanGround global frame). Writes the transform for scene_track_obj to consume.

--mode anchored (experimental, opt-in, default stays 'cloud' = today's unchanged behavior) ALSO
anchors against a building-footprint contour (scene_extract_contour.py's output) for sessions
with low real overlap against the persistent map, where cloud-to-cloud correlation alone has
little true signal -- see herald.scene.refine.align.anchored_align()'s docstring for exactly
what was validated and its narrow-optimum caveat before relying on it for a new case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True, help="new recon dump (.npz)")
    p.add_argument("--persistent", type=Path, required=True, help="persistent scene map (.npz)")
    p.add_argument("--out", type=Path, required=True, help="output Sim3 (.json)")
    p.add_argument("--no-align", action="store_true", help="identity transform (pre-registered poses)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["cloud", "anchored"], default="cloud",
                    help="'cloud' (default, unchanged): point-cloud-vs-point-cloud only via "
                         "align(). 'anchored' (experimental, opt-in): also anchors against "
                         "--contour -- see this script's module docstring")
    p.add_argument("--contour", type=Path, default=None,
                    help="required for --mode anchored (scene_extract_contour.py's output .json)")
    args = p.parse_args()

    from herald.scene.refine import load_session

    identity = {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "inliers": 1.0}
    if args.no_align or not args.persistent.exists():
        sim3 = identity
        why = "no persistent map yet" if not args.persistent.exists() else "--no-align"
        print(f"identity transform ({why})", flush=True)
    elif args.mode == "anchored":
        if args.contour is None:
            raise SystemExit("--mode anchored requires --contour (run scene_extract_contour.py first)")
        from herald.scene.refine.align import anchored_align
        from herald.scene.refine.contour import resample_contour
        _, spts, _ = load_session(args.session)
        _, ppts, _ = load_session(args.persistent)
        # anchored_align assumes a planar (yaw + xy-translation) registration, same as every
        # Tier-2 cloud (gravity-aligned NED, "up" already a known common axis) -- flatten for
        # the alignment math only, the recovered Sim3 is still a full rigid transform applied to
        # the real (unflattened) points downstream by scene_track_obj.
        spts_flat, ppts_flat = spts.copy(), ppts.copy()
        spts_flat[:, 2] = 0.0
        ppts_flat[:, 2] = 0.0
        contour = resample_contour(args.contour)
        res = anchored_align(spts_flat, contour, ppts_flat)
        T = res.transform
        sim3 = {"scale": float(T.s), "R": np.asarray(T.R).tolist(),
                "t": np.asarray(T.t).tolist(), "inliers": -1.0}  # not tracked on this path
        print(f"anchored align session->persistent: yaw={np.degrees(res.yaw):.2f}deg "
              f"energy={res.energy:.4f}", flush=True)
    else:
        from herald.scene.refine import align
        _, spts, _ = load_session(args.session)
        _, ppts, _ = load_session(args.persistent)
        A = align(spts, ppts, yaw_seeds=24, iters=80, record=False, device=args.device, verbose=True)
        T = A.transform
        sim3 = {"scale": float(T.s), "R": np.asarray(T.R).tolist(),
                "t": np.asarray(T.t).tolist(), "inliers": float(A.inlier_frac)}
        print(f"align session->persistent: scale={T.s:.4f} inliers={A.inlier_frac:.2%}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sim3))
    print(f"sim3 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
