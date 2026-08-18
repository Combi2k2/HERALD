#!/usr/bin/env python3
"""Diagnostic for Tier-2 refine cloud alignment (herald.scene.refine.align).

Two modes:

  SYNTHETIC (default): build one asymmetric room cloud, cut it into two partially
    overlapping halves, put each in its own arbitrary frame (independent ground tilt +
    yaw + translation, optional scale + noise), then recover the source->target transform
    with `align` and report the residual RMSE on the shared region. This is a closed-loop
    correctness check -- no data or GPU needed (runs on CPU).

  REAL (--p1 A --p2 B): align two saved clouds (.npz with a `points` key, or .ply) and
    report energy / inlier fraction.

Both render to Rerun: target (blue), source before (red), source after alignment (green).
Good alignment = the green cloud lands on the blue one.

  uv run python scripts/test_refine.py                    # synthetic self-test (CPU)
  uv run python scripts/test_refine.py --p1 a.npz --p2 b.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.scene.refine import align, fit_ground_plane


# ------------------------------------------------------------------- cloud IO / synth

def load_points(path: Path) -> np.ndarray:
    """Load (N,3) points from .npz/.npy (key `points` if present) or binary/ascii .ply."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32).reshape(-1, 3)
    if path.suffix == ".npz":
        d = np.load(path)
        return d[("points" if "points" in d.files else d.files[0])].astype(np.float32).reshape(-1, 3)
    if path.suffix == ".ply":
        return _read_ply(path)
    raise ValueError(f"unsupported cloud format: {path.suffix}")


def _read_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n, ascii_fmt = 0, False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("format"):
                ascii_fmt = "ascii" in line
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line == "end_header":
                break
        if ascii_fmt:
            rows = [f.readline().split()[:3] for _ in range(n)]
            return np.asarray(rows, np.float32)
        # binary: assume x,y,z float32 first (write_ply layout), skip any trailing bytes/row
        rest = np.frombuffer(f.read(), np.uint8)
    stride = len(rest) // n
    xyz = np.empty((n, 3), np.float32)
    view = rest.reshape(n, stride)
    xyz[:] = view[:, :12].copy().view(np.float32)
    return xyz


def synth_room(seed: int = 0) -> np.ndarray:
    """A *cluttered* asymmetric room (representative of a real dense reconstruction): a
    12x8 floor, walls of different heights with a doorway, and ~40 distinct furniture
    boxes scattered throughout. The clutter is what gives the alignment energy a unique
    minimum at the true pose -- a bare room (walls only) is translation-ambiguous along
    the walls under partial overlap, which no global registration can resolve."""
    rng = np.random.default_rng(seed)
    parts = []
    fx, fy = np.meshgrid(np.linspace(0, 12, 90), np.linspace(0, 8, 60))
    parts.append(np.stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)], 1))

    def wall(x0, y0, x1, y1, h, gap=None):
        t = np.linspace(0, 1, 140)
        z = np.linspace(0, h, 28)
        tt, zz = np.meshgrid(t, z)
        keep = np.ones(tt.size, bool) if gap is None else \
            ((tt.ravel() < gap[0]) | (tt.ravel() > gap[1]))
        x = x0 + (x1 - x0) * tt.ravel()
        y = y0 + (y1 - y0) * tt.ravel()
        return np.stack([x[keep], y[keep], zz.ravel()[keep]], 1)

    parts.append(wall(0, 0, 12, 0, 3.0))                    # south wall, tall
    parts.append(wall(0, 8, 12, 8, 2.0, gap=(0.4, 0.6)))    # north wall, doorway
    parts.append(wall(0, 0, 0, 8, 2.5))                     # west wall
    parts.append(wall(12, 0, 12, 8, 1.5))                   # east wall, short
    # ~40 distinct furniture boxes on a jittered grid over the whole floor
    fixed = np.random.default_rng(seed + 99)
    for cx in np.linspace(1.0, 11.0, 8):
        for cy in np.linspace(1.0, 7.0, 5):
            if fixed.random() < 0.15:                        # a few gaps -> asymmetry
                continue
            cx2, cy2 = cx + fixed.uniform(-0.4, 0.4), cy + fixed.uniform(-0.4, 0.4)
            sx, sy, h = fixed.uniform(0.3, 0.9, 3) * [1, 1, 1.3]
            b = fixed.uniform(-0.5, 0.5, (150, 3)) * [sx, sy, h] + [cx2, cy2, h / 2]
            parts.append(b)
    return np.concatenate(parts, 0).astype(np.float32)


def rand_frame(pts: np.ndarray, rng, tilt_deg: float, scale: float, noise: float):
    """Put `pts` into an arbitrary session frame: small ground tilt + yaw + translation
    (+ scale + noise). Returns (transformed_points, R, t, s) with pt' = s R pt + t."""
    yaw = rng.uniform(0, 2 * np.pi)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    # small tilt about a random horizontal axis
    ax = np.array([rng.normal(), rng.normal(), 0.0]); ax /= np.linalg.norm(ax) + 1e-9
    ang = np.radians(rng.uniform(-tilt_deg, tilt_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], np.float32)
    Rt = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    R = (Rt @ Rz).astype(np.float32)
    t = rng.uniform(-5, 5, 3).astype(np.float32)
    out = scale * pts @ R.T + t
    if noise:
        out = out + rng.normal(0, noise, out.shape)
    return out.astype(np.float32), R, t, float(scale)


# ------------------------------------------------------------------------------- viz

def show(target, src_raw, src_aligned, out: Path | None):
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init("herald.test_refine")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="align"), auto_views=False))
    rr.log("align/target", rr.Points3D(target, colors=[(80, 130, 255)], radii=0.02))
    rr.log("align/source_raw", rr.Points3D(src_raw, colors=[(230, 70, 70)], radii=0.02))
    rr.log("align/source_aligned", rr.Points3D(src_aligned, colors=[(70, 210, 90)], radii=0.02))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(out))
        print(f"rrd: {out}", flush=True)


# ------------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p1", type=Path, help="target cloud (.npz/.npy/.ply); omit for synthetic")
    p.add_argument("--p2", type=Path, help="source cloud to align onto p1")
    p.add_argument("--tilt", type=float, default=8.0, help="synthetic: max per-cloud ground tilt (deg)")
    p.add_argument("--gt-scale", type=float, default=1.0, help="synthetic: scale applied to source frame")
    p.add_argument("--noise", type=float, default=0.02, help="synthetic: per-point noise (m)")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--max-points", type=int, default=12000)
    p.add_argument("--optimize-scale", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/refine/test_refine.rrd"))
    args = p.parse_args()

    # synthetic clouds get an independent ground tilt -> need RANSAC levelling (level=True);
    # real NED session clouds are already gravity-aligned so merge_sessions uses level=False.
    kw = dict(yaw_seeds=24, iters=args.iters, src_points=args.max_points,
              record=False, level=True, optimize_scale=args.optimize_scale, device=args.device,
              seed=args.seed, verbose=True)

    if args.p1 and args.p2:
        target, source = load_points(args.p1), load_points(args.p2)
        print(f"REAL: target {len(target)} pts, source {len(source)} pts", flush=True)
        A = align(source, target, **kw)
        print(f"\nRESULT: scale={A.transform.s:.4f} yaw={np.degrees(A.yaw):.2f}deg "
              f"t={A.transform.t.round(3)} energy={A.energy:.4f} inliers={A.inlier_frac:.2%}", flush=True)
        show(target, source, A.apply(source), args.out)
        return

    # ---- synthetic closed-loop test
    rng = np.random.default_rng(args.seed)
    C = synth_room(args.seed)
    xm = 4.0, 8.0                                            # overlap band in x: [4, 8]
    base1 = C[C[:, 0] <= xm[1]]                              # target half  (x <= 8)
    base2 = C[C[:, 0] >= xm[0]]                              # source half  (x >= 4)
    over = C[(C[:, 0] >= xm[0]) & (C[:, 0] <= xm[1])]        # shared region (ground truth)

    target, R1, t1, s1 = rand_frame(base1, rng, args.tilt, 1.0, args.noise)
    source, R2, t2, s2 = rand_frame(base2, rng, args.tilt, args.gt_scale, args.noise)
    print(f"SYNTH: room {len(C)} pts | target {len(target)} | source {len(source)} | "
          f"overlap {len(over)} | tilt<= {args.tilt}deg gt_scale={args.gt_scale}", flush=True)

    A = align(source, target, **kw)

    # recovery error: map the shared region through both frames and compare
    #   target frame: p1 = s1 R1 O + t1 ;  source frame: p2 = s2 R2 O + t2
    #   a perfect T satisfies T(p2) = p1 on the overlap.
    p1 = (s1 * over @ R1.T + t1).astype(np.float32)
    p2 = (s2 * over @ R2.T + t2).astype(np.float32)
    err = np.linalg.norm(A.apply(p2) - p1, axis=1)
    diag = float(np.linalg.norm(target.max(0) - target.min(0)))
    print(f"\nRESULT: scale={A.transform.s:.4f} (gt {args.gt_scale:.3f})  yaw={np.degrees(A.yaw):.2f}deg  "
          f"energy={A.energy:.4f}  inliers={A.inlier_frac:.2%}", flush=True)
    print(f"RECOVERY on overlap: RMSE={np.sqrt((err**2).mean()):.4f} m  median={np.median(err):.4f} m  "
          f"p95={np.percentile(err,95):.4f} m   (scene diag {diag:.1f} m)", flush=True)
    ok = np.sqrt((err ** 2).mean()) < max(0.15, 3 * args.noise)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} (RMSE < {max(0.15, 3*args.noise):.3f} m)", flush=True)

    show(target, source, A.apply(source), args.out)


if __name__ == "__main__":
    main()
