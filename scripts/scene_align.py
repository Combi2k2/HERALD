#!/usr/bin/env python
"""scene_align: Sim3 that integrates a new recon session into the persistent scene map.

Aligns the new session's cloud onto the persistent map's cloud (energy-based Sim3). Identity
when there is no persistent map yet (first session) or --no-align (pre-registered poses, e.g.
TartanGround global frame). Writes the transform for scene_track_obj to consume.
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
    args = p.parse_args()

    from herald.scene.refine import load_session

    identity = {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "inliers": 1.0}
    if args.no_align or not args.persistent.exists():
        sim3 = identity
        why = "no persistent map yet" if not args.persistent.exists() else "--no-align"
        print(f"identity transform ({why})", flush=True)
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
