#!/usr/bin/env python3
"""Diagnostic for Tier-2 refine alignment (herald.scene.refine.align) on real clouds.

TartanGround trajectories share a world frame, so the true P_i<->P_j transform is ~identity
-- to get GROUND TRUTH we perturb the source by a KNOWN Sim3 and check the aligner inverts
it. The perturbation spans the full config: yaw about +Z, xy-tilt (rotation about X and Y,
i.e. a non-gravity-aligned ground), translation, and scale. xy-tilt turns on the RANSAC
levelling path (level=True); scale!=1 turns on scale optimisation.

Reports the recovered Sim3, rotation/scale error vs the true inverse, and the point RMSE,
and renders the convergence: target static (blue-ish real colour), source re-transformed by
the Sim3 at every optimiser step over the 'iter' timeline.

  uv run python scripts/test_align.py --yaw 35 --tilt-x 6 --tilt-y -4 --trans 5 3 0.5 --scale 1.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.scene.common.geometry import Sim3
from herald.scene.recon.utils import SceneCloud, write_ply
from herald.scene.refine import align
from merge_sessions import load_cloud


def _R(axis: int, deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    m = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    m[i, i], m[j, j] = c, c
    m[i, j], m[j, i] = (-s, s) if axis != 1 else (s, -s)
    return m


def rot_xyz(yaw: float, tilt_x: float, tilt_y: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(tilt_y) @ Rx(tilt_x)  (degrees)."""
    return (_R(2, yaw) @ _R(1, tilt_y) @ _R(0, tilt_x)).astype(np.float64)


def _ang(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _axisangle(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def slerp_sim3(A: Sim3, B: Sim3, m: int) -> list:
    """`m` interpolated Sim3 from A (exclusive) to B (inclusive): SLERP rotation, lerp
    scale/translation. Used only to make the coarse-search jump watchable in the viz."""
    Rr = A.R.T @ B.R
    ang = np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))
    axis = (np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
            / (2 * np.sin(ang)) if ang > 1e-6 else np.array([0, 0, 1.0]))
    return [Sim3(A.s + (B.s - A.s) * f, A.R @ _axisangle(axis, f * ang), A.t + (B.t - A.t) * f)
            for f in np.linspace(0, 1, m + 1)[1:]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p1", type=Path, default=Path("data/tartanground/Hospital/recon/P0000.npz"),
                   help="target cloud")
    p.add_argument("--p2", type=Path, default=Path("data/tartanground/Hospital/recon/P0001.npz"),
                   help="source cloud (perturbed, then aligned back onto target)")
    # ---- initial config: the known perturbation applied to the source
    p.add_argument("--yaw", type=float, default=35.0, help="yaw about +Z (deg)")
    p.add_argument("--tilt-x", type=float, default=6.0, help="tilt about X / roll (deg)")
    p.add_argument("--tilt-y", type=float, default=-4.0, help="tilt about Y / pitch (deg)")
    p.add_argument("--trans", type=float, nargs=3, default=[5.0, 3.0, 0.5])
    p.add_argument("--scale", type=float, default=1.1)
    # ---- optimiser config
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rec-stride", type=int, default=5)
    p.add_argument("--anim-points", type=int, default=40000)
    p.add_argument("--voxel", type=float, default=0.1, help="merge voxel (density normalisation)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/refine/test_align.rrd"))
    args = p.parse_args()

    pts1, cols1 = load_cloud(args.p1)
    pts2, cols2 = load_cloud(args.p2)
    G = Sim3(args.scale, rot_xyz(args.yaw, args.tilt_x, args.tilt_y), np.asarray(args.trans, np.float64))
    src = G.apply(pts2).astype(np.float32)                       # perturbed source
    has_tilt = abs(args.tilt_x) > 1e-6 or abs(args.tilt_y) > 1e-6
    has_scale = abs(args.scale - 1.0) > 1e-6
    print(f"target {args.p1.name}: {len(pts1)} | source {args.p2.name}: {len(pts2)}", flush=True)
    print(f"PERTURB (known): yaw={args.yaw} tilt_x={args.tilt_x} tilt_y={args.tilt_y} "
          f"t={args.trans} scale={args.scale}  -> level={has_tilt} optimize_scale={has_scale}", flush=True)

    A = align(src, pts1, yaw_seeds=24, iters=args.iters, rec_stride=args.rec_stride,
              level=has_tilt, optimize_scale=has_scale, record=True, device=args.device,
              seed=args.seed, verbose=True)
    T = A.transform

    # ground truth: T should equal G^{-1}  (inverse similarity)
    Ginv_R = G.R.T
    rot_err = _ang(T.R @ G.R)                                    # ~0 if T.R == G.R^T
    err = np.linalg.norm(A.apply(src) - pts2, axis=1)
    print(f"\nRECOVERED: scale={T.s:.4f} (true {1/args.scale:.4f})  rot_err={rot_err:.2f}deg  "
          f"t={T.t.round(3)}", flush=True)
    print(f"RECOVERY: RMSE={np.sqrt((err**2).mean()):.4f} m  median={np.median(err):.4f} m  "
          f"p95={np.percentile(err,95):.4f} m  | energy={A.energy:.4f} inliers={A.inlier_frac:.2%}", flush=True)
    ok = np.sqrt((err ** 2).mean()) < 0.3 and rot_err < 3.0
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}  (RMSE<0.3m and rot_err<3deg)", flush=True)

    # ---- convergence animation: target static, source re-transformed at every step
    import rerun as rr
    import rerun.blueprint as rrb
    rng = np.random.default_rng(0)
    s2 = rng.choice(len(src), min(args.anim_points, len(src)), replace=False)
    s1 = rng.choice(len(pts1), min(150000, len(pts1)), replace=False)
    rr.init("herald.test_align")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/target", rr.Points3D(pts1[s1], colors=cols1[s1], radii=0.02), static=True)
    # frame 0 = the raw perturbed source (identity), then a smooth ramp across the coarse-
    # search jump into history[0], then the recorded refinement -- so the full perturbation
    # (e.g. a 160deg yaw) is actually visible converging, not hidden by the coarse init.
    frames = [Sim3()] + slerp_sim3(Sim3(), A.history[0], 20) + A.history[1:]
    for k, Sk in enumerate(frames):
        rr.set_time("iter", sequence=k)
        rr.log("world/source", rr.Points3D(Sk.apply(src[s2]).astype(np.float32),
                                           colors=cols2[s2], radii=0.02))

    # final step: the voxel-snapped MERGED cloud (target + aligned source -> one point per
    # shared voxel), rendered in place of the moving source
    scene = SceneCloud(voxel=args.voxel)
    scene.add(pts1, cols1)
    scene.add(A.apply(src), cols2)
    mpts, mcols = scene.cloud()
    rr.set_time("iter", sequence=len(frames))
    rr.log("world/source", rr.Clear(recursive=True))
    rr.log("world/merged", rr.Points3D(mpts, colors=mcols, radii=0.02))
    ply = args.out.with_suffix(".ply")
    write_ply(ply, mpts, mcols)
    print(f"MERGED: {len(pts1)}+{len(src)} -> {len(mpts)} voxels  ->  {ply}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.out))
    print(f"rrd: {args.out}  (scrub 'iter'; last step shows the merged cloud)", flush=True)


if __name__ == "__main__":
    main()
