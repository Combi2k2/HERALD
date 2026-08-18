#!/usr/bin/env python3
"""Detection boxes -> per-object 3D oriented bounding boxes (OBBs), on GT geometry.

Validation of the detection-first, no-mask idea: per frame, YOLO-World gives
object boxes; for each box we unproject its GT depth into the world and remove
the background *geometrically* -- cut the global floor plane (up + floor height
estimated once from the point cloud, `estimate_ground`), then keep the nearest
substantial DBSCAN cluster (drops the disconnected wall behind). We fit a
gravity-aligned, yaw-only oriented box (vertical sides along up, footprint
rotated in the horizontal plane) -- well-posed even for thin/planar objects.
The up direction comes from the cloud (trajectory only seeds it), so it
transfers to VGGT geometry. Renders rgb+boxes and the per-frame 3D object
points + OBBs to an .rrd.

Needs a real CUDA GPU on the 3090 partition (scripts/box_obb.slurm).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from detect_video import DEFAULT_VOCAB
from herald.datasets import TartanGroundGeometry, TartanGroundTraj
from herald.scene.recon.utils import unproject_labeled
from herald.viz import colorize_labels


def box_label_image(boxes: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Paint each box into a uint16 id image, largest-area first so smaller
    (usually nearer/foreground) boxes overwrite. Returns (label_img, rank->row)."""
    lab = np.zeros(shape, np.uint16)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)                       # largest first
    for rank, i in enumerate(order, 1):
        x0, y0, x1, y1 = boxes[i].astype(int)
        lab[max(0, y0):y1, max(0, x0):x1] = rank
    return lab, order


def _ransac_plane(P: np.ndarray, thresh: float, iters: int, rng) -> tuple[np.ndarray, float]:
    """Largest-support plane by RANSAC -> (unit normal, offset d), plane = {x: n.x == d}."""
    best_n, best_d, best_cnt = np.array([0.0, 0.0, 1.0]), 0.0, -1
    for _ in range(iters):
        s = P[rng.integers(0, len(P), 3)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = float(n @ s[0])
        cnt = int((np.abs(P @ n - d) < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best_n, best_d = cnt, n, d
    return best_n, best_d


def estimate_ground(geo, indices, stride: int = 6, n_samples: int = 80,
                    thresh: float = 0.05, iters: int = 400, max_depth: float = 40.0):
    """Up direction + camera eye-height from the *point cloud*, via RANSAC of the
    dominant floor plane (its largest flat support). Robust where a global
    least-variance axis tilts -- multi-level scenes, big walls -- and
    gauge-independent, so it transfers to VGGT geometry. Cameras disambiguate the
    sign (they ride above the floor). Returns (up unit(3), eye), where `eye` is
    the typical camera height above the LOCAL floor: ground removal then uses a
    per-frame cut floor_i = cam_i . up - eye, which handles multiple levels
    (a single global floor plane would only clear the lowest one)."""
    sel = indices[:: max(1, len(indices) // n_samples)]
    rng = np.random.default_rng(0)
    cams, frames = [], []
    for i in sel:
        g = geo.geometry(i)
        cams.append(g.c2w[:3, 3])
        valid = np.isfinite(g.depth) & (g.depth > 0) & (g.depth < max_depth)
        p, _ = unproject_labeled(g.depth, g.K, g.c2w, np.ones(g.depth.shape, np.uint16),
                                 stride=stride, valid=valid)
        if len(p):
            frames.append(p[rng.integers(0, len(p), min(len(p), 4000))])
    cams = np.asarray(cams)
    pts = np.concatenate(frames)
    n, d = _ransac_plane(pts, thresh, iters, rng)
    if np.median(cams @ n) < d:                       # orient up so cameras sit above the floor
        n, d = -n, -d
    inl = pts[np.abs(pts @ n - d) < thresh]           # refine on floor inliers
    q = inl - inl.mean(0)
    _, V = np.linalg.eigh(q.T @ q)
    up = V[:, 0]
    up = (up if up @ n > 0 else -up)
    up = up / np.linalg.norm(up)
    eyes = [c @ up - np.percentile(f @ up, 3) for c, f in zip(cams, frames) if len(f) > 500]
    return up, float(np.median(eyes))


def foreground(pts: np.ndarray, cam: np.ndarray, up: np.ndarray, floor_h: float,
               eps: float, min_samples: int, min_pts: int,
               floor_margin: float = 0.08) -> np.ndarray | None:
    """Geometric background removal: (1) cut everything within `floor_margin` of
    the local floor (height `floor_h` along `up`, = cam.up - eye) -- DBSCAN can't
    split the ground because it is physically connected to the object base --
    then (2) keep the nearest substantial DBSCAN cluster (drops the wall behind)."""
    if len(pts) < min_pts:
        return None
    pts = pts[(pts @ up) > floor_h + floor_margin]      # per-frame local floor cut
    if len(pts) < min_pts:
        return None
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    best, best_d = None, np.inf
    for c in np.unique(lab):
        if c < 0:
            continue
        sel = lab == c
        if int(sel.sum()) < min_pts:
            continue
        d = float(np.median(np.linalg.norm(pts[sel] - cam, axis=1)))
        if d < best_d:
            best_d, best = d, sel
    return pts[best] if best is not None else None


def fit_obb(pts: np.ndarray, up: np.ndarray):
    """Gravity-aligned (yaw-only) oriented box: sides vertical along `up`, the
    footprint rotated in the horizontal plane. Well-posed even for thin/planar
    objects (doors, mirrors) where full-3D PCA rolls to a random tilt. Returns
    (center(3), quat xyzw(4), half_sizes(3)); half order = [footprint1, footprint2, height]."""
    up = up / np.linalg.norm(up)
    a = np.eye(3)[int(np.argmin(np.abs(up)))]
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)                             # (e1, e2, up) right-handed
    B = np.stack([e1, e2], axis=1)                    # (3, 2) plane basis
    P = pts @ B                                       # (N, 2) footprint
    c2 = P.mean(0); Q = P - c2
    _, ev = np.linalg.eigh(Q.T @ Q)                   # 2x2 in-plane PCA
    if np.linalg.det(ev) < 0:
        ev[:, -1] *= -1                               # proper 2D rotation
    proj = Q @ ev
    lo, hi = proj.min(0), proj.max(0)
    h = pts @ up; hlo, hhi = float(h.min()), float(h.max())
    ax = B @ ev                                       # (3, 2) world footprint axes
    R = np.column_stack([ax[:, 0], ax[:, 1], up])     # (e1,e2,up) RH * proper 2D rot -> det +1
    mid = c2 + ev @ ((lo + hi) / 2)                   # footprint centre in (e1, e2)
    center = mid[0] * e1 + mid[1] * e2 + ((hlo + hhi) / 2) * up
    half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hhi - hlo) / 2])
    return center, Rotation.from_matrix(R).as_quat(), half


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0=all frames")
    p.add_argument("--model", default="yolov8x-worldv2.pt")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--pixel-stride", type=int, default=3, help="depth subsampling per box")
    p.add_argument("--max-depth", type=float, default=60.0)
    p.add_argument("--dbscan-eps", type=float, default=0.15, help="cluster radius (m)")
    p.add_argument("--min-samples", type=int, default=4)
    p.add_argument("--min-pts", type=int, default=30, help="drop objects with fewer fg points")
    p.add_argument("--floor-margin", type=float, default=0.08, help="ground slab thickness (m) sliced along up")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    indices = list(range(len(traj)))[:: max(1, args.frame_stride)]
    if args.max_frames:
        indices = indices[: args.max_frames]
    if not indices:
        raise SystemExit(f"no frames under {traj.base}")

    geo = TartanGroundGeometry(traj)
    up, eye = estimate_ground(geo, indices, max_depth=args.max_depth)
    print(f"ground: up={np.round(up, 3).tolist()}  eye_height={eye:.3f}m", flush=True)

    from ultralytics import YOLOWorld

    model = YOLOWorld(args.model)
    model.set_classes(DEFAULT_VOCAB)

    import rerun as rr

    rr.init("herald.box_obb")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # TartanGround GT = NED
    print(f"{args.env}/{args.traj}: box->OBB over {len(indices)} frames "
          f"(conf={args.conf}, eps={args.dbscan_eps})", flush=True)

    t0 = time.perf_counter()
    n_obj: list[int] = []
    centers_path: list[np.ndarray] = []
    for k, i in enumerate(indices, 1):
        rgb = traj.load_rgb(i).astype(np.uint8)
        g = geo.geometry(i)
        r = model.predict(Image.fromarray(rgb), conf=args.conf, device=args.device, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        rr.set_time("frame", sequence=i)
        h, w = g.depth.shape
        rr.log("world/camera", rr.Transform3D(translation=g.c2w[:3, 3], mat3x3=g.c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=g.K, resolution=[w, h],
                                                camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/image/rgb", rr.Image(rgb))
        centers_path.append(g.c2w[:3, 3].copy())
        if len(centers_path) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers_path)], colors=[(255, 0, 255)]))

        if len(boxes) == 0:
            rr.log("world/objects", rr.Clear(recursive=True))
            n_obj.append(0)
            continue
        rr.log("world/camera/image/boxes", rr.Boxes2D(
            mins=boxes[:, :2], sizes=boxes[:, 2:] - boxes[:, :2],
            labels=[names[c] for c in cls]))

        lab, order = box_label_image(boxes, (h, w))
        valid = np.isfinite(g.depth) & (g.depth > 0) & (g.depth < args.max_depth)
        pts, plab = unproject_labeled(g.depth, g.K, g.c2w, lab, stride=args.pixel_stride, valid=valid)
        cam = g.c2w[:3, 3]

        centers, halves, quats, labels, opts, oplab = [], [], [], [], [], []
        for rank in np.unique(plab):
            if rank <= 0:
                continue
            fg = foreground(pts[plab == rank], cam, up, cam @ up - eye, args.dbscan_eps,
                            args.min_samples, args.min_pts, args.floor_margin)
            if fg is None:
                continue
            ctr, quat, half = fit_obb(fg, up)
            centers.append(ctr); halves.append(half); quats.append(quat)
            labels.append(names[int(cls[order[rank - 1]])])
            opts.append(fg); oplab.append(np.full(len(fg), rank))
        n_obj.append(len(centers))

        if centers:
            P = np.concatenate(opts)
            rr.log("world/objects/points", rr.Points3D(P, colors=colorize_labels(np.concatenate(oplab)), radii=0.02))
            rr.log("world/objects/obb", rr.Boxes3D(
                centers=centers, half_sizes=halves,
                quaternions=np.asarray(quats, np.float32),   # Nx4 xyzw; rr.Quaternion path breaks on numpy<2 (ultralytics pin)
                labels=labels, fill_mode="majorwireframe"))
        else:
            rr.log("world/objects", rr.Clear(recursive=True))

        if k % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"  frame {i}: {len(boxes)} boxes -> {len(centers)} OBBs  "
                  f"[{dt:.0f}s, {dt / k:.2f}s/frame]", flush=True)

    dt = time.perf_counter() - t0
    a = np.array(n_obj)
    print(f"PERF: {len(indices)} frames in {dt:.1f}s = {dt / len(indices):.2f}s/frame", flush=True)
    print(f"OBBs/frame: mean {a.mean():.1f}, median {int(np.median(a))}, max {a.max()}", flush=True)

    out = args.out or (traj.base / "recon" / f"obb_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}\nview with: rerun {out}", flush=True)


if __name__ == "__main__":
    main()
