#!/usr/bin/env python3
"""Tier-2 refine end-to-end: align two session clouds, animate the Sim(3) optimisation,
and merge them into one density-normalised cloud.

  1. load two clouds (target P1, source P2) from .npz/.ply,
  2. `align` P2 -> P1 with the energy-field optimiser (records the Sim(3) trajectory),
  3. Rerun: P1 shown static; P2 re-transformed by the Sim(3) at *every* optimisation step
     over a timeline, so you watch it converge onto P1,
  4. merge = P1 + T(P2), voxel-downsampled so points from the two clouds that land in the
     same voxel are snapped to one (normalised density); saved as PLY.

  uv run python scripts/merge_sessions.py --p1 P0000.npz --p2 P0001.npz --out merged.ply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.scene.common.geometry import Sim3
from herald.scene.recon.utils import SceneCloud, write_ply
from herald.scene.refine import align


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (points (N,3) float32, colors (N,3) uint8) from .npz/.npy/.ply."""
    path = Path(path)
    if path.suffix in (".npz", ".npy"):
        d = np.load(path)
        if path.suffix == ".npy":
            return d.astype(np.float32).reshape(-1, 3), np.full((len(d), 3), 160, np.uint8)
        pts = d["points"].astype(np.float32)
        cols = d["colors"].astype(np.uint8) if "colors" in d.files else np.full((len(pts), 3), 160, np.uint8)
        return pts, cols
    if path.suffix == ".ply":
        return _read_ply(path)
    raise ValueError(f"unsupported cloud format: {path.suffix}")


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        n, ascii_fmt, has_rgb = 0, False, False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("format"):
                ascii_fmt = "ascii" in line
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property") and ("red" in line or "green" in line or "blue" in line):
                has_rgb = True
            elif line == "end_header":
                break
        if ascii_fmt:
            rows = np.array([f.readline().split() for _ in range(n)], np.float32)
            xyz = rows[:, :3]
            cols = rows[:, 3:6].astype(np.uint8) if has_rgb else np.full((n, 3), 160, np.uint8)
            return xyz, cols
        raw = np.frombuffer(f.read(), np.uint8)
    stride = len(raw) // n
    view = raw.reshape(n, stride)
    xyz = view[:, :12].copy().view(np.float32).reshape(n, 3)
    cols = view[:, 12:15].copy() if has_rgb and stride >= 15 else np.full((n, 3), 160, np.uint8)
    return xyz, cols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p1", type=Path, required=True, help="target cloud (.npz/.ply)")
    p.add_argument("--p2", type=Path, required=True, help="source cloud aligned onto p1")
    p.add_argument("--voxel", type=float, default=0.1, help="merge voxel (density normalisation)")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--rec-stride", type=int, default=6, help="record every Nth optimiser step")
    p.add_argument("--optimize-scale", action="store_true")
    p.add_argument("--anim-points", type=int, default=40000, help="pts shown for the moving cloud")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-yaw", type=float, default=0.0,
                   help="rotate source by this known yaw (deg) before aligning -> ground-truth "
                        "recovery test (TartanGround trajectories share a world frame, so the true "
                        "transform is otherwise identity)")
    p.add_argument("--perturb-t", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="known translation applied with --perturb-yaw")
    p.add_argument("--out", type=Path, required=True, help="merged .ply")
    p.add_argument("--rrd", type=Path, default=None)
    args = p.parse_args()

    pts1, cols1 = load_cloud(args.p1)
    pts2, cols2 = load_cloud(args.p2)
    print(f"target {args.p1.name}: {len(pts1)} pts | source {args.p2.name}: {len(pts2)} pts", flush=True)

    # optional known perturbation of the source -> recovery test with ground truth
    G, pts2_ref = None, pts2.copy()
    if args.perturb_yaw or any(args.perturb_t):
        a = np.radians(args.perturb_yaw); cz, sz = np.cos(a), np.sin(a)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
        G = Sim3(1.0, Rz, np.asarray(args.perturb_t, np.float64))
        pts2 = G.apply(pts2).astype(np.float32)
        print(f"PERTURB source by yaw={args.perturb_yaw}deg t={args.perturb_t} "
              f"-> aligner should recover the inverse", flush=True)

    A = align(pts2, pts1, yaw_seeds=24, iters=args.iters, rec_stride=args.rec_stride,
              optimize_scale=args.optimize_scale, record=True, device=args.device,
              seed=args.seed, verbose=True)
    T = A.transform
    print(f"\nTRANSFORM: scale={T.s:.4f} yaw={np.degrees(A.yaw):.2f}deg t={T.t.round(3)}\n"
          f"  energy={A.energy:.4f} inliers={A.inlier_frac:.2%} history_steps={len(A.history)}", flush=True)
    if G is not None:                                     # ground-truth recovery: T should invert G
        err = np.linalg.norm(A.apply(pts2) - pts2_ref, axis=1)
        print(f"RECOVERY vs known perturbation: RMSE={np.sqrt((err**2).mean()):.4f} m  "
              f"median={np.median(err):.4f} m  p95={np.percentile(err,95):.4f} m", flush=True)

    # merge + density-normalise: both clouds into one voxel grid (shared voxel -> one point)
    scene = SceneCloud(voxel=args.voxel)
    scene.add(pts1, cols1)
    scene.add(A.apply(pts2), cols2)
    mpts, mcols = scene.cloud()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.out, mpts, mcols)
    np.savez_compressed(args.out.with_suffix(".npz"), points=mpts.astype(np.float32), colors=mcols.astype(np.uint8))
    print(f"MERGED: {len(pts1)}+{len(pts2)} -> {len(mpts)} voxels  ->  {args.out}", flush=True)

    # ---- Rerun: static target + source re-transformed at every recorded optimiser step
    import rerun as rr
    import rerun.blueprint as rrb
    rng = np.random.default_rng(0)
    sub2 = rng.choice(len(pts2), min(args.anim_points, len(pts2)), replace=False)
    sub1 = rng.choice(len(pts1), min(150000, len(pts1)), replace=False)
    rr.init("herald.merge_sessions")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/target", rr.Points3D(pts1[sub1], colors=cols1[sub1], radii=0.02), static=True)
    for k, Sk in enumerate(A.history):
        rr.set_time("iter", sequence=k)
        rr.log("world/source", rr.Points3D(Sk.apply(pts2[sub2]).astype(np.float32),
                                           colors=cols2[sub2], radii=0.02))
    rr.set_time("iter", sequence=len(A.history))           # final: the full merged cloud
    rr.log("world/source", rr.Clear(recursive=True))
    rr.log("world/merged", rr.Points3D(mpts, colors=mcols, radii=0.02))
    out_rrd = args.rrd or args.out.with_suffix(".rrd")
    rr.save(str(out_rrd))
    print(f"rrd: {out_rrd}  (scrub the 'iter' timeline to watch convergence)", flush=True)


if __name__ == "__main__":
    main()
