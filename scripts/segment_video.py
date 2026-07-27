#!/usr/bin/env python3
"""Run ONLY the SAM2 segmentation stage (services.Sam2) over a whole TartanGround
video and render the masks to an .rrd for inspection -- no geometry, no
reconstruction. Verifies the unified services.Sam2 class end to end and lets you
scrub the raw masks (tiling / flicker / reflection artifacts) frame by frame.

Default: per-frame automatic masks (track=False) -- stateless, no video-model
memory, so no OOM; one mask list per frame, painted largest-first into a label
image. --track switches to the video tracker (track=True): chunked, heavier, and
can OOM on dense chunks (that's the tracker, not this script).

Needs a real CUDA GPU on the 3090 partition (scripts/segment_video.slurm).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundTraj
from herald.viz import colorize_labels
from services.sam2 import Sam2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0=all frames")
    p.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--points-per-batch", type=int, default=32)
    p.add_argument("--min-area", type=int, default=200)
    p.add_argument("--iou-thresh", type=float, default=0.7, help="AMG pred_iou_thresh (lib default 0.88)")
    p.add_argument("--stability-thresh", type=float, default=0.9,
                   help="AMG stability_score_thresh (lib default 0.95)")
    p.add_argument("--track", action="store_true",
                   help="use the video tracker (track=True) instead of per-frame AMG")
    p.add_argument("--chunk", type=int, default=16, help="track only: frames per SAM2 session")
    p.add_argument("--no-offload", action="store_true", help="track only: keep SAM2 state on GPU")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None, help="output .rrd")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    indices = list(range(len(traj)))[:: max(1, args.frame_stride)]
    if args.max_frames:
        indices = indices[: args.max_frames]
    if not indices:
        raise SystemExit(f"no frames under {traj.base}")

    sam2 = Sam2(args.sam2_model, device=args.device,
                points_per_side=args.points_per_side, points_per_batch=args.points_per_batch,
                min_area=args.min_area, iou_thresh=args.iou_thresh,
                stability_thresh=args.stability_thresh,
                track=args.track, chunk=args.chunk, offload=not args.no_offload)

    import rerun as rr

    rr.init("herald.segment_video")
    print(f"{args.env}/{args.traj}: segmenting {len(indices)} frames "
          f"(track={args.track}, pps={args.points_per_side}, min_area={args.min_area})", flush=True)

    t0 = time.perf_counter()
    state = {"done": 0}

    def log(fi: int, rgb: np.ndarray, lab: np.ndarray, n: int) -> None:
        rr.set_time("frame", sequence=fi)
        rr.log("rgb", rr.Image(rgb))
        rr.log("seg", rr.Image(colorize_labels(lab.astype(np.int64))))
        state["done"] += 1
        if state["done"] % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"  frame {fi}: {n} masks  [{dt:.0f}s, {dt / state['done']:.2f}s/frame]", flush=True)

    if args.track:
        pending: list[tuple[int, np.ndarray]] = []

        def drain(results) -> None:
            for (fi, rgb), (lab, _conf) in zip(pending, results):
                log(fi, rgb, lab, int(lab.max()))
            pending.clear()

        for i in indices:
            rgb = traj.load_rgb(i)
            pending.append((i, rgb))
            res = sam2.push(rgb)
            if res:
                drain(res)
        res = sam2.finish()
        if res:
            drain(res)
    else:
        for i in indices:
            rgb = traj.load_rgb(i)
            lab = sam2.segment(rgb)                 # collapsed label image (module-side)
            log(i, rgb, lab, int(lab.max()))

    dt = time.perf_counter() - t0
    n = state["done"]
    print(f"PERF: {n} frames in {dt:.1f}s = {dt / max(1, n):.2f}s/frame", flush=True)

    out = args.out or (traj.base / "recon" / f"seg_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)


if __name__ == "__main__":
    main()
