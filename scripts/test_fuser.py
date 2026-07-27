#!/usr/bin/env python3
"""Validate the Fuser's object association on synthetic frames, and render a
small synthetic reconstruction stream so the behaviour is visible in rerun.

The Fuser identifies objects by appearance (SigLIP embedding of each mask crop)
AND occupancy (voxel overlap in world space), gated by an AABB prefilter. This
exercises the behaviours that matter without needing SAM2 or a GPU.

Two parts:
  1. assertion cases on hand-drawn frames (headless), printed PASS/FAIL;
  2. a synthetic multi-view stream of a few colored boxes, rendered to rerun
     with the rgb / depth / semantic-mask streams (toggleable overlays under
     the camera) alongside the evolving reconstruction — the same view layout
     as test_recon, but tiny and CPU-only.

    CUDA_VISIBLE_DEVICES="" python scripts/test_fuser.py        # CPU; writes runs/test_fuser.rrd

Cases:
  A  same two objects over 3 frames        -> 2  (cross-frame merge, no fragmenting)
  B  two identical-looking objects, apart  -> 2  (occupancy splits despite cos=1)
  C  same object, camera moves 5 cm        -> 1  (small motion still overlaps)
  D  same-looking object 5 m away          -> 2  (AABB prefilter rejects)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.scene.recon import Fuser, SiglipEmbedder
from herald.scene.recon.utils import masks_to_labels

H = W = 96
K = np.array([[60.0, 0, 48], [0, 60, 48], [0, 0, 1]])


def _mask(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _geo(depth_val=5.0, t=(0.0, 0.0, 0.0), boxes=()):
    depth = np.full((H, W), depth_val, np.float32)
    for (y0, y1, x0, x1, d) in boxes:
        depth[y0:y1, x0:x1] = d
    c2w = np.eye(4)
    c2w[:3, 3] = t
    return {"K": K, "c2w": c2w, "depth": depth}


def _fuser(embedder, **kw):
    return Fuser(embedder, voxel=0.1, stride=1, erode=0, min_area=50,
                 radius=1.0, min_overlap=0.2, dilate=1, min_csim=0.5, **kw)


def _assertions(embedder) -> bool:
    results = []

    # A: two distinct objects (red near, blue far), 3 identical frames -> 2 objects
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[10:30, 10:30] = [200, 40, 40]
    rgb[60:80, 60:80] = [40, 40, 200]
    mR, mB = _mask(10, 30, 10, 30), _mask(60, 80, 60, 80)
    g = _geo(boxes=[(60, 80, 60, 80, 8.0)])
    f = _fuser(embedder)
    for _ in range(3):
        f.push(g, [mR, mB], rgb)
    results.append(("A two objs x3 frames", len(f._objs), 2))

    # B: two identical red blobs, far apart in one frame -> 2 (occupancy splits)
    rgb2 = np.zeros((H, W, 3), np.uint8)
    rgb2[10:30, 10:30] = [200, 40, 40]
    rgb2[10:30, 60:80] = [200, 40, 40]
    f2 = _fuser(embedder)
    f2.push(_geo(), [_mask(10, 30, 10, 30), _mask(10, 30, 60, 80)], rgb2)
    cos = float(f2._emb[0] @ f2._emb[1]) if len(f2._objs) == 2 else float("nan")
    results.append((f"B identical-look apart (cos={cos:.2f})", len(f2._objs), 2))

    # C: same object, camera translates 5 cm (< voxel) -> 1 (still overlaps)
    rgbC = np.zeros((H, W, 3), np.uint8)
    rgbC[10:30, 10:30] = [200, 40, 40]
    mC = _mask(10, 30, 10, 30)
    fC = _fuser(embedder)
    fC.push(_geo(t=(0.0, 0, 0)), [mC], rgbC)
    fC.push(_geo(t=(0.05, 0, 0)), [mC], rgbC)
    results.append(("C same obj, 5cm motion", len(fC._objs), 1))

    # D: same-looking object 5 m away -> 2 (AABB prefilter rejects)
    fD = _fuser(embedder)
    fD.push(_geo(t=(0.0, 0, 0)), [mC], rgbC)
    fD.push(_geo(t=(5.0, 0, 0)), [mC], rgbC)
    results.append(("D same look, 5m apart", len(fD._objs), 2))

    print()
    ok = True
    for name, got, want in results:
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<34} objects={got} (expect {want})")

    out = f.flush(min_obs=1, min_points=1)
    assert set(out) == {"points", "labels", "hits", "obj_labels", "obj_frames", "frame"}
    assert out["points"].shape[0] == out["labels"].shape[0] == out["hits"].shape[0]
    print(f"  flush keys + shapes OK ({len(out['obj_labels'])} objects, {out['points'].shape[0]} points)")
    return ok


# --------------------------------------------------------------------------- #
#  synthetic multi-view stream: a few colored boxes rendered by point splatting
# --------------------------------------------------------------------------- #
SH = SW = 140
SK = np.array([[100.0, 0, SW / 2], [0, 100, SH / 2], [0, 0, 1]])
# (center, color, half-extent) — spaced so their voxel clouds don't touch
BOXES = [
    ((-1.5, 0.0, 5.0), (220, 50, 50)),   # red
    ((0.0, 0.0, 5.0), (50, 200, 70)),    # green
    ((1.5, 0.0, 5.0), (60, 90, 230)),    # blue
]
HALF = 0.5


def _box_points(center, n=2500, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-HALF, HALF, (n, 3))


def _render(objs, c2w):
    """Splat world-space colored points into (rgb, depth, inst) for one pose."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    us, vs, zs, oid, col = [], [], [], [], []
    for k, (pts, color) in enumerate(objs, start=1):
        cam = (pts - t) @ R                       # world -> camera
        z = cam[:, 2]
        m = z > 1e-3
        cam, z = cam[m], z[m]
        us.append(cam[:, 0] / z * SK[0, 0] + SK[0, 2])
        vs.append(cam[:, 1] / z * SK[1, 1] + SK[1, 2])
        zs.append(z)
        oid.append(np.full(len(z), k))
        col.append(np.broadcast_to(color, (len(z), 3)))
    u, v, z = np.concatenate(us), np.concatenate(vs), np.concatenate(zs)
    oid, col = np.concatenate(oid), np.concatenate(col)
    order = np.argsort(-z)                          # far first, so nearer overwrite
    u, v, z, oid, col = u[order], v[order], z[order], oid[order], col[order]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    rgb = np.zeros((SH, SW, 3), np.uint8)
    depth = np.zeros((SH, SW), np.float32)
    inst = np.zeros((SH, SW), np.int32)
    for du in (-1, 0, 1):                           # 3x3 splat to fill holes
        for dv in (-1, 0, 1):
            uu, vv = ui + du, vi + dv
            ok = (uu >= 0) & (uu < SW) & (vv >= 0) & (vv < SH)
            rgb[vv[ok], uu[ok]] = col[ok]
            depth[vv[ok], uu[ok]] = z[ok]
            inst[vv[ok], uu[ok]] = oid[ok]
    return rgb, depth, inst


def _render_stream(embedder, out_path: Path, n_frames: int = 10) -> int:
    import rerun as rr
    from herald.viz import colorize_labels

    objs = [(_box_points(c, seed=k), col) for k, (c, col) in enumerate(BOXES)]
    fuser = Fuser(embedder, voxel=0.1, stride=1, erode=0, min_area=50,
                  radius=1.0, min_overlap=0.2, dilate=1, min_csim=0.5)

    rr.init("herald.test_fuser")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    centers = []
    for f, x in enumerate(np.linspace(-0.4, 0.4, n_frames)):
        c2w = np.eye(4)
        c2w[:3, 3] = [x, 0.0, 0.0]
        rgb, depth, inst = _render(objs, c2w)
        masks = [inst == k for k in range(1, len(objs) + 1) if (inst == k).any()]
        fuser.push({"K": SK, "c2w": c2w, "depth": depth}, masks, rgb)

        rr.set_time("frame", sequence=f)
        rr.log("world/camera", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=SK, resolution=[SW, SH],
                                                camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/image/rgb", rr.Image(rgb))
        rr.log("world/camera/image/depth",
               rr.DepthImage(depth, meter=1.0, colormap=rr.components.Colormap.Viridis))
        # seg colored by stable object color id (same palette as the cloud)
        seg = masks_to_labels(fuser.last_masks, values=fuser.last_cids)
        rr.log("world/camera/image/seg", rr.Image(colorize_labels(seg)))
        centers.append(c2w[:3, 3].copy())
        if len(centers) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(centers)], colors=[(255, 0, 255)]))

        cloud = fuser.flush(frame=f, min_obs=1, min_points=1)
        rr.log("world/cloud", rr.Points3D(cloud["points"], colors=colorize_labels(cloud["labels"]),
                                          radii=0.02))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_path))
    final = fuser.flush(min_obs=1, min_points=1)
    return len(final["obj_labels"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rrd", type=Path, default=Path("runs/test_fuser.rrd"),
                    help="synthetic-stream reconstruction output ('' to skip rendering)")
    ap.add_argument("--frames", type=int, default=10, help="synthetic-stream frame count")
    args = ap.parse_args()

    print(f"loading SigLIP {args.model} on {args.device} ...", flush=True)
    embedder = SiglipEmbedder(args.model, device=args.device, batch_size=16)

    ok = _assertions(embedder)

    if str(args.rrd):
        n = _render_stream(embedder, args.rrd, n_frames=args.frames)
        good = n == len(BOXES)
        ok &= good
        print(f"\n  [{'PASS' if good else 'FAIL'}] synthetic stream: {n} objects "
              f"(expect {len(BOXES)} boxes merged across {args.frames} views)")
        print(f"  rrd: {args.rrd}   (view: rerun {args.rrd})")

    print("\n" + ("OK — Fuser association behaves as designed." if ok else "FAILURES above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
