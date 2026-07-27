#!/usr/bin/env python3
"""Run ONLY open-vocab object detection (YOLO-World / ultralytics) over a whole
TartanGround video and render boxes to an .rrd -- no geometry, no segmentation,
no reconstruction. A recall/count sanity check for the detection-first idea:
how many *object* boxes per frame vs the ~84-123 SAM masks, and whether it finds
hospital-specific things.

Needs a real CUDA GPU on the 3090 partition (scripts/detect_video.slurm).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from herald.datasets import TartanGroundTraj

# Broad indoor + hospital vocabulary for the open-vocab prompt (edit / --vocab-file).
DEFAULT_VOCAB = [
    "person", "chair", "wheelchair", "bed", "hospital bed", "stretcher", "table",
    "desk", "cabinet", "shelf", "counter", "sink", "toilet", "door", "window",
    "curtain", "monitor", "screen", "tv", "computer", "laptop", "keyboard",
    "telephone", "medical equipment", "iv stand", "ventilator", "oxygen tank",
    "trash can", "box", "cart", "trolley", "stool", "sofa", "couch", "bench",
    "lamp", "light", "sign", "fire extinguisher", "clock", "plant", "bottle",
    "cup", "bag", "pillow", "blanket", "towel", "mirror", "whiteboard", "book",
    "basket", "bucket", "ladder", "backpack", "fan", "picture frame",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0=all frames")
    p.add_argument("--model", default="yolov8x-worldv2.pt", help="YOLO-World weights")
    p.add_argument("--conf", type=float, default=0.1, help="detection confidence floor")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--vocab-file", type=Path, default=None, help="one class name per line")
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

    vocab = (args.vocab_file.read_text().split() if args.vocab_file else None) or DEFAULT_VOCAB

    from ultralytics import YOLOWorld

    model = YOLOWorld(args.model)
    model.set_classes(vocab)

    import rerun as rr

    rr.init("herald.detect_video")
    print(f"{args.env}/{args.traj}: detecting {len(indices)} frames "
          f"({args.model}, {len(vocab)} classes, conf={args.conf})", flush=True)

    t0 = time.perf_counter()
    counts: list[int] = []
    for k, i in enumerate(indices, 1):
        rgb = traj.load_rgb(i).astype(np.uint8)
        r = model.predict(Image.fromarray(rgb), conf=args.conf, max_det=args.max_det,
                          device=args.device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names
        labels = [f"{names[c]} {s:.2f}" for c, s in zip(cls, conf)]
        counts.append(len(xyxy))

        rr.set_time("frame", sequence=i)
        rr.log("image", rr.Image(rgb))
        if len(xyxy):
            rr.log("image/boxes", rr.Boxes2D(mins=xyxy[:, :2], sizes=xyxy[:, 2:] - xyxy[:, :2],
                                             labels=labels))
        else:
            rr.log("image/boxes", rr.Clear(recursive=False))
        if k % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"  frame {i}: {len(xyxy)} detections  [{dt:.0f}s, {dt / k:.2f}s/frame]", flush=True)

    dt = time.perf_counter() - t0
    c = np.array(counts)
    print(f"PERF: {len(indices)} frames in {dt:.1f}s = {dt / len(indices):.2f}s/frame", flush=True)
    print(f"DETECTIONS/frame: mean {c.mean():.1f}, median {int(np.median(c))}, "
          f"min {c.min()}, max {c.max()}", flush=True)

    out = args.out or (traj.base / "recon" / f"detect_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}", flush=True)
    print(f"view with: rerun {out}", flush=True)


if __name__ == "__main__":
    main()
