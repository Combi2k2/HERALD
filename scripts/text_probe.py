#!/usr/bin/env python3
"""Probe SigLIP image<->text similarity of saved crop embeddings against a set
of words (ground/floor/wall/...), to see if text scoring can flag ground masks.

Reuses the image embeddings already in a dump_crops run's embeddings.npz (they
live in SigLIP's joint image-text space), and only encodes the text side here.
Reports, per word, the cosine distribution over all crops and the top crops.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from herald.scene.recon import SiglipEmbedder

WORDS = ["ground", "floor", "wall", "ceiling", "chair", "table", "sofa", "door", "window"]
TEMPLATE = "This is a photo of a {}."


def pct(x):
    q = np.percentile(x, [5, 50, 95])
    return f"mean={x.mean():+.3f}  p5={q[0]:+.3f} p50={q[1]:+.3f} p95={q[2]:+.3f}  max={x.max():+.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/910109/crops", "runs/910116/crops"])
    p.add_argument("--words", nargs="+", default=WORDS)
    p.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    p.add_argument("--device", default="cpu")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    emb = SiglipEmbedder(args.siglip_model, device=args.device)
    prompts = [TEMPLATE.format(w) for w in args.words]
    with torch.no_grad():
        inp = emb.processor(text=prompts, return_tensors="pt", padding="max_length").to(args.device)
        tfeat = emb.model.get_text_features(**inp).pooler_output
        tfeat = torch.nn.functional.normalize(tfeat, dim=-1).cpu().numpy()  # (W, D)

    for run in args.runs:
        npz = np.load(Path(run) / "embeddings.npz", allow_pickle=True)
        names = [str(x) for x in npz["names"]]
        img = npz["embs"]                        # (N, D) unit-norm, joint space
        cos = img @ tfeat.T                      # (N, W)
        print(f"\n================  {run}  (N={len(names)} crops)  ================")
        for wi, w in enumerate(args.words):
            col = cos[:, wi]
            print(f"\n[{w}]  {pct(col)}")
            order = np.argsort(col)[::-1][:args.topk]
            for o in order:
                print(f"    {col[o]:+.3f}  {names[o]}")


if __name__ == "__main__":
    main()
