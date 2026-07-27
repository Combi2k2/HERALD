#!/usr/bin/env python3
"""Measure the SigLIP cosine-similarity distributions the Fuser gates on.

Runs SAM2 + SigLIP on a subsample of TartanGround frames and reports two
populations of pairwise cosines:

  NEG  different masks within the same frame        -> gate should REJECT
  POS  same physical object across adjacent frames  -> gate should ACCEPT
       (correspondence by mask IoU between consecutive sampled frames,
        so use a small --stride to keep inter-frame motion low)

If the NEG distribution sits mostly above min_csim/merge_csim, the semantic
gate never rejects anything and association/merging is driven by geometry
alone — which is how a whole room collapses into one object.

    sbatch/srun (3090):  .venv/bin/python scripts/csim_stats.py --env Hospital --traj P0001
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.datasets import TartanGroundTraj
from herald.scene.recon import Sam2Segmenter, SiglipEmbedder
from herald.scene.recon.utils import bbox_crop, iou


def _pct(x: np.ndarray) -> str:
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return (f"n={len(x):4d}  mean={x.mean():.3f}  "
            f"p5={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} p75={q[3]:.3f} p95={q[4]:.3f}")


def _hist(x: np.ndarray, lo=0.0, hi=1.0, bins=20, width=50) -> str:
    h, edges = np.histogram(x, bins=bins, range=(lo, hi))
    top = max(1, h.max())
    return "\n".join(f"    {edges[b]:.2f}-{edges[b+1]:.2f} |{'#' * int(width * h[b] / top):<{width}}| {h[b]}"
                     for b in range(bins) if h[b])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", required=True)
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0001")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--n-frames", type=int, default=24, help="sampled frame pairs")
    p.add_argument("--stride", type=int, default=4, help="gap between the two frames of a pair")
    p.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    p.add_argument("--points-per-side", type=int, default=8)
    p.add_argument("--min-area", type=int, default=800)
    p.add_argument("--pos-iou", type=float, default=0.6, help="mask IoU to count as same object")
    p.add_argument("--bg-fade", type=float, default=0.5,
                   help="fade non-mask crop pixels toward white before embedding (0 = raw crop)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    starts = np.linspace(0, len(traj) - 1 - args.stride, args.n_frames).astype(int)
    sam2 = Sam2Segmenter(args.sam2_model, device=args.device,
                         points_per_side=args.points_per_side, min_area=args.min_area)
    embedder = SiglipEmbedder(args.siglip_model, device=args.device)

    neg, pos = [], []
    for s in starts:
        frames = []
        for i in (int(s), int(s) + args.stride):
            rgb = traj.load_rgb(i)
            masks = sam2.segment(rgb)
            if not masks:
                frames.append(([], np.empty((0, embedder.dim))))
                continue
            embs = embedder([bbox_crop(rgb, m, bg_fade=args.bg_fade) for m in masks])
            frames.append((masks, embs))
        (mA, eA), (mB, eB) = frames
        # NEG: all distinct-mask pairs within each frame
        for masks_, embs_ in frames:
            sim = embs_ @ embs_.T
            iu = np.triu_indices(len(masks_), k=1)
            neg.extend(sim[iu].tolist())
        # POS: cross-frame pairs whose masks overlap heavily (same object)
        for a in range(len(mA)):
            for b in range(len(mB)):
                if iou(mA[a], mB[b]) >= args.pos_iou:
                    pos.append(float(eA[a] @ eB[b]))
        print(f"  frame {s}: {len(mA)}+{len(mB)} masks | neg={len(neg)} pos={len(pos)}", flush=True)

    neg_a, pos_a = np.asarray(neg), np.asarray(pos)
    print(f"\n=== {args.env}/{args.traj} ({args.n_frames} pairs, stride {args.stride}) ===")
    print(f"NEG (different objects, same frame):\n  {_pct(neg_a)}")
    print(_hist(neg_a))
    print(f"POS (same object, {args.stride} frames apart, IoU>={args.pos_iou}):\n  {_pct(pos_a)}")
    print(_hist(pos_a))

    print("\nthreshold sweep: fraction of pairs the gate would ACCEPT (cos >= t)")
    print(f"  {'t':>6} {'NEG acc (want ~0)':>18} {'POS acc (want ~1)':>18}")
    for t in np.arange(0.40, 0.96, 0.05):
        print(f"  {t:6.2f} {float((neg_a >= t).mean()):18.3f} {float((pos_a >= t).mean()):18.3f}")


if __name__ == "__main__":
    main()
