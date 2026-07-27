#!/usr/bin/env python3
"""Object-level reconstruction: detection boxes -> background-removed clouds ->
merge across frames by 3D voxel-overlap gating -> per-object OBB, on the fly.

Per frame: YOLO-World boxes -> per box, unproject GT depth + geometric
background removal (box_obb.foreground: global floor-plane cut then nearest
DBSCAN cluster) -> a small object cloud. Each cloud is associated to a
persistent GLOBAL object by voxel co-occupancy (fraction of its voxels landing
on an existing object >= merge_overlap) gated by class label, else it starts a
new object. Points accumulate per object (voxel-downsampled), and the
gravity-aligned yaw-only oriented box is refit whenever the object grows.
Up + floor height come from the point cloud (box_obb.estimate_ground). Renders
the evolving object clouds + OBBs to an .rrd. No masks, no SAM2.

Needs a real CUDA GPU on the 3090 partition (scripts/box_recon.slurm).
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from box_obb import box_label_image, estimate_ground, fit_obb, foreground
from detect_video import DEFAULT_VOCAB
from herald.datasets import TartanGroundGeometry, TartanGroundTraj
from herald.scene.recon.utils import unproject_labeled
from herald.viz import colorize_labels

_OFF = 1 << 20


class ObjectMap:
    """Accumulate per-detection background-removed clouds into persistent global
    objects by 3D voxel-overlap gating (+ class gate). Points are stored
    voxel-downsampled (one running centroid per occupied voxel), so memory is
    bounded by explored volume, not frame count."""

    def __init__(self, voxel: float = 0.1, merge_overlap: float = 0.2, class_gate: bool = True):
        self.voxel = voxel
        self.merge_overlap = merge_overlap
        self.class_gate = class_gate
        self._objs: dict[int, dict] = {}   # gid -> {vox: {code:[sx,sy,sz,n]}, cls, n_obs}
        self._owner: dict[int, int] = {}   # voxel code -> gid
        self._next = 1

    def _encode(self, pts: np.ndarray) -> np.ndarray:
        k = np.floor(pts / self.voxel).astype(np.int64) + _OFF
        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]

    def update(self, pts: np.ndarray, cls: str) -> int | None:
        if len(pts) == 0:
            return None
        codes = self._encode(pts)
        uniq, inv = np.unique(codes, return_inverse=True)
        sums = np.zeros((len(uniq), 3)); np.add.at(sums, inv, pts)
        cnts = np.bincount(inv)

        hits: dict[int, int] = defaultdict(int)      # match against pre-update state
        for c in uniq.tolist():
            g = self._owner.get(c)
            if g is not None and (not self.class_gate or self._objs[g]["cls"] == cls):
                hits[g] += 1
        gid, best = None, self.merge_overlap * len(uniq)
        for g, n in hits.items():
            if n >= best:
                best, gid = n, g
        if gid is None:
            gid, self._next = self._next, self._next + 1
            self._objs[gid] = {"vox": {}, "cls": cls, "n_obs": 0}

        obj = self._objs[gid]; obj["n_obs"] += 1
        vox = obj["vox"]
        for i, c in enumerate(uniq.tolist()):
            e = vox.get(c)
            if e is None:
                vox[c] = [sums[i, 0], sums[i, 1], sums[i, 2], float(cnts[i])]
                self._owner[c] = gid
            else:
                e[0] += sums[i, 0]; e[1] += sums[i, 1]; e[2] += sums[i, 2]; e[3] += cnts[i]
        return gid

    def objects(self, min_vox: int = 3, min_obs: int = 1):
        """Yield (gid, points Nx3 voxel-centroids, cls, n_obs) for solid objects."""
        for g, o in self._objs.items():
            if len(o["vox"]) < min_vox or o["n_obs"] < min_obs:
                continue
            v = np.asarray(list(o["vox"].values()))          # (M,4)
            yield g, v[:, :3] / v[:, 3:4], o["cls"], o["n_obs"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--model", default="yolov8x-worldv2.pt")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--pixel-stride", type=int, default=3)
    p.add_argument("--max-depth", type=float, default=60.0)
    p.add_argument("--dbscan-eps", type=float, default=0.15)
    p.add_argument("--min-samples", type=int, default=4)
    p.add_argument("--min-pts", type=int, default=30, help="min fg points to accept a detection")
    p.add_argument("--floor-margin", type=float, default=0.08, help="ground slab thickness (m) sliced along up")
    # --- merge ---
    p.add_argument("--merge-voxel", type=float, default=0.1)
    p.add_argument("--merge-overlap", type=float, default=0.2)
    p.add_argument("--no-class-gate", action="store_true", help="merge regardless of class label")
    p.add_argument("--min-vox", type=int, default=5, help="drop objects with fewer voxels")
    p.add_argument("--min-obs", type=int, default=2, help="drop objects seen in fewer frames")
    p.add_argument("--snapshot-every", type=int, default=64)
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
    up, floor_d = estimate_ground(geo, indices, max_depth=args.max_depth)
    omap = ObjectMap(voxel=args.merge_voxel, merge_overlap=args.merge_overlap,
                     class_gate=not args.no_class_gate)

    from ultralytics import YOLOWorld

    model = YOLOWorld(args.model)
    model.set_classes(DEFAULT_VOCAB)

    import rerun as rr

    rr.init("herald.box_recon")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    print(f"{args.env}/{args.traj}: object-recon over {len(indices)} frames "
          f"(merge_overlap={args.merge_overlap}, class_gate={not args.no_class_gate}, "
          f"up={np.round(up, 3).tolist()}, floor_d={floor_d:.3f})", flush=True)

    def snapshot(fi: int) -> int:
        pts, gids, centers, halves, quats, labels = [], [], [], [], [], []
        for g, cloud, cls, _ in omap.objects(min_vox=args.min_vox, min_obs=args.min_obs):
            pts.append(cloud); gids.append(np.full(len(cloud), g))
            ctr, quat, half = fit_obb(cloud, up)
            centers.append(ctr); halves.append(half); quats.append(quat); labels.append(cls)
        rr.set_time("frame", sequence=fi)
        if pts:
            rr.log("world/objects/points", rr.Points3D(np.concatenate(pts),
                   colors=colorize_labels(np.concatenate(gids)), radii=0.02))
            rr.log("world/objects/obb", rr.Boxes3D(centers=centers, half_sizes=halves,
                   quaternions=np.asarray(quats, np.float32), labels=labels, fill_mode="majorwireframe"))
        return len(centers)

    t0 = time.perf_counter()
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

        if len(boxes):
            rr.log("world/camera/image/boxes", rr.Boxes2D(
                mins=boxes[:, :2], sizes=boxes[:, 2:] - boxes[:, :2], labels=[names[c] for c in cls]))
            lab, order = box_label_image(boxes, (h, w))
            valid = np.isfinite(g.depth) & (g.depth > 0) & (g.depth < args.max_depth)
            pts, plab = unproject_labeled(g.depth, g.K, g.c2w, lab, stride=args.pixel_stride, valid=valid)
            cam = g.c2w[:3, 3]
            for rank in np.unique(plab):
                if rank <= 0:
                    continue
                fg = foreground(pts[plab == rank], cam, up, floor_d, args.dbscan_eps,
                                args.min_samples, args.min_pts, args.floor_margin)
                if fg is not None:
                    omap.update(fg, names[int(cls[order[rank - 1]])])

        if args.snapshot_every and k % args.snapshot_every == 0:
            n = snapshot(i)
            dt = time.perf_counter() - t0
            print(f"  frame {i}: {n} objects  [{dt:.0f}s, {dt / k:.2f}s/frame]", flush=True)

    n = snapshot(indices[-1])
    dt = time.perf_counter() - t0
    print(f"PERF: {len(indices)} frames in {dt:.1f}s = {dt / len(indices):.2f}s/frame", flush=True)
    print(f"FINAL: {n} objects", flush=True)

    out = args.out or (traj.base / "recon" / f"objrecon_{args.env}_{args.traj}.rrd")
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    print(f"rrd: {out}\nview with: rerun {out}", flush=True)


if __name__ == "__main__":
    main()
