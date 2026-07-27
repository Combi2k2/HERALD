#!/usr/bin/env python3
"""Dump SAM2 mask crops from the first few recon frames as PNGs, for eyeballing
SigLIP similarity later.

Same front of the recon pipeline as test_recon.py / csim_stats.py -- load RGB,
run SAM2 automatic masks, crop each to its bbox with the background faded exactly
as the Fuser does (utils.bbox_crop, bg_fade) -- but instead of embedding and
fusing, it just writes the crops to disk. Filenames encode frame and mask index
so you can pick any two and compare.

Optionally (--embed) also runs SigLIP and saves an embeddings .npz + a cosine
matrix, so you can check similarity numerically without re-running the models.

    .venv/bin/python scripts/dump_crops.py --env Hospital --traj P0001 \
        --n-frames 4 --out runs/crops
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from herald.datasets import TartanGroundTraj
from herald.scene.recon import Sam2Segmenter
from herald.scene.recon.utils import bbox_crop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", default="Hospital")
    p.add_argument("--version", default="omni")
    p.add_argument("--traj", default="P0001")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--n-frames", type=int, default=4, help="number of frames to process")
    p.add_argument("--stride", type=int, default=4, help="gap between processed frames")
    p.add_argument("--out", type=Path, default=Path("runs/crops"))
    p.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--points-per-side", type=int, default=8)
    p.add_argument("--min-area", type=int, default=800)
    p.add_argument("--pad", type=int, default=3, help="bbox pad (px); Fuser uses 3")
    p.add_argument("--pad-frac", type=float, default=0.0,
                   help="extra pad as a fraction of bbox size per side (0.5 = +H/2, +W/2 = more context)")
    p.add_argument("--bg-fade", type=float, default=0.5,
                   help="fade non-mask crop pixels toward black before saving (0=raw, 1=solid black)")
    p.add_argument("--max-per-frame", type=int, default=12, help="cap crops saved per frame")
    p.add_argument("--embed", action="store_true",
                   help="also run SigLIP and save embeddings + cosine matrix (needs GPU)")
    p.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    traj = TartanGroundTraj(root=args.root, env=args.env, version=args.version,
                            traj=args.traj, camera=args.camera)
    sam2 = Sam2Segmenter(args.sam2_model, device=args.device,
                         points_per_side=args.points_per_side, min_area=args.min_area)

    args.out.mkdir(parents=True, exist_ok=True)
    names, crops_all = [], []
    idxs = list(range(0, args.n_frames * args.stride, args.stride))
    for i in idxs:
        rgb = traj.load_rgb(i)
        masks = sam2.segment(rgb)[: args.max_per_frame]   # already largest-first
        for k, m in enumerate(masks):
            crop = bbox_crop(rgb, m, pad=args.pad, pad_frac=args.pad_frac, bg_fade=args.bg_fade)
            name = f"f{i:04d}_m{k:02d}_a{int(m.sum())}"
            Image.fromarray(np.asarray(crop, np.uint8)).save(args.out / f"{name}.png")
            names.append(name)
            crops_all.append(crop)
        print(f"frame {i:4d}: saved {len(masks)} crops", flush=True)
    print(f"wrote {len(names)} crops to {args.out}", flush=True)

    if args.embed:
        from herald.scene.recon import SiglipEmbedder
        embedder = SiglipEmbedder(args.siglip_model, device=args.device)
        embs = embedder(crops_all)                        # (N, D) unit-norm
        cos = embs @ embs.T                               # cosine matrix (unit vectors)
        np.savez(args.out / "embeddings.npz", names=np.array(names), embs=embs, cos=cos)
        print(f"wrote embeddings.npz ({embs.shape}) + cosine matrix to {args.out}", flush=True)
        # quick peek: the most-similar distinct-crop pairs
        iu = np.triu_indices(len(names), k=1)
        order = np.argsort(cos[iu])[::-1][:10]
        print("\ntop-10 most similar crop pairs:")
        for o in order:
            a, b = iu[0][o], iu[1][o]
            print(f"  {cos[a, b]:.3f}  {names[a]}  <->  {names[b]}")


if __name__ == "__main__":
    main()
