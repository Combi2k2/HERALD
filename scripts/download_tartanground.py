#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tartanair as ta

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/tartanground", help="Download root")
    p.add_argument("--env", nargs="+", required=True, help="Environment name(s), e.g. DowntownWest")
    p.add_argument("--version", nargs="+", default=["omni"], choices=["omni", "diff", "anymal"])
    p.add_argument("--traj", nargs="+", default=["P0000"], help="Trajectory ids, e.g. P0000 P0001")
    p.add_argument(
        "--modality",
        nargs="+",
        default=["image", "depth", "seg", "meta"],
        help="image/depth/seg/meta/lidar/imu/... (meta carries poses+intrinsics)",
    )
    p.add_argument("--camera", nargs="+", default=["lcam_front"], help="Camera name(s)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-unzip", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    ta.init(str(root))
    ta.download_ground(
        env=args.env,
        version=args.version,
        traj=args.traj,
        modality=args.modality,
        camera_name=args.camera,
        unzip=not args.no_unzip,
        num_workers=args.workers,
    )
    print(f"Done. Data under {root}", flush=True)

if __name__ == "__main__":
    main()
