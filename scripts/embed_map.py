#!/usr/bin/env python3
"""Backfill (or refresh) text-label embeddings on a saved SceneMap, in place.

Loads a merged/session `.npz`, fills every object's `embedding` with the vote-weighted mean of
its label vectors (`herald.scene.semantic.embed_objects`) using the text-only `TextEncoder`, and
re-saves -- cloud, meta, and all other object fields are round-tripped untouched. Only the small
distinct-label vocabulary is encoded, so this is fast.

    uv run python scripts/embed_map.py --map runs/merge_multi3/merged.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.scene.recon.types import SessionResult
from herald.scene.refine import load_meta, load_session, save_session
from herald.scene.semantic import embed_objects
from services.embeddings import TextEncoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=Path, required=True, help="SceneMap .npz to embed in place")
    p.add_argument("--out", type=Path, default=None, help="write here instead of overwriting --map")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    objs, pts, cols = load_session(args.map)
    meta = load_meta(args.map)
    labelled = [o for o in objs if o.labels]
    print(f"loaded {args.map.name}: {len(objs)} objects ({len(labelled)} with labels), "
          f"{len(pts)} pts", flush=True)

    enc = TextEncoder(model_id=args.embed_model, device=args.device)
    embed_objects(objs, enc)
    dim = next((o.embedding.shape[0] for o in objs if o.embedding is not None), 0)
    done = sum(o.embedding is not None for o in objs)
    print(f"embedded {done}/{len(objs)} objects  dim={dim}  model={args.embed_model}", flush=True)

    # sanity: nearest-neighbour by cosine for a couple of objects
    E = np.array([o.embedding for o in objs if o.embedding is not None], np.float32)
    lab = [o.label for o in objs if o.embedding is not None]
    if len(E) > 1:
        S = E @ E.T
        np.fill_diagonal(S, -1.0)
        for i in range(min(5, len(E))):
            j = int(S[i].argmax())
            print(f"  '{lab[i]}' nearest -> '{lab[j]}'  cos={S[i, j]:.3f}", flush=True)

    out = args.out or args.map
    save_session(out, SessionResult(pts, cols, objs, meta=meta))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
