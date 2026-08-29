#!/usr/bin/env python
"""N-way multi-session merge (Tier-2).

Aligns each session's cloud onto the first (canonical) session, then reconciles objects into
one persistent map by chaining `reconcile` (union-find two-pass gate). Objects are rendered
colour-coded by their contributing source(s): each session gets a base colour and an object's
colour is the mean of its sources -- so a box seen in only one session shows that pure colour,
and a fused box shows the blend. A few crop thumbnails per object are resolved from each
session's source path for inspection.

    uv run python scripts/merge_multi.py --sessions A.npz B.npz C.npz --out merged.npz
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from herald.scene.recon.types import SessionResult
from herald.scene.recon.utils import SceneCloud, object_crops, write_ply
from herald.scene.refine import (
    align, load_meta, load_session, overlap_merge, reconcile, save_session, transform_object)

# per-source base colours (RGB); blended by averaging for multi-source (fused) objects
BASE = [(230, 60, 60), (60, 200, 60), (70, 110, 255),
        (235, 195, 40), (200, 70, 200), (60, 200, 200)]


def obj_color(session_ids, order: dict) -> tuple:
    cols = [BASE[order[s]] for s in session_ids if s in order]
    if not cols:
        return (150, 150, 150)
    return tuple(int(v) for v in np.mean(cols, axis=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", required=True, type=Path, help="session .npz dumps")
    p.add_argument("--ios-trust", type=float, default=0.9, help="pass 1: IoS to fuse regardless of label")
    p.add_argument("--cdiag-gate", type=float, default=0.5, help="pass 2: centre-dist/diag co-location gate")
    p.add_argument("--overlap-ios", type=float, default=0.5,
                   help="final dedup pass: fuse ANY remaining pair with OBB IoS >= this "
                        "(label/session-agnostic; catches intra-session + label-disagreeing overlaps; 0 disables)")
    p.add_argument("--no-align", action="store_true",
                   help="skip Sim3 align, use identity (TartanGround global poses are pre-registered)")
    p.add_argument("--voxel", type=float, default=0.1, help="merged-cloud voxel (m)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True, help="merged map .npz")
    p.add_argument("--rrd", type=Path, default=None)
    p.add_argument("--no-embed", dest="embed", action="store_false",
                   help="skip text-label embeddings (default: embed via text-only model)")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    maps = [load_session(s) for s in args.sessions]          # each: (objects, points, colors)
    metas = [load_meta(s) for s in args.sessions]
    sids, sources = [], {}
    for m in metas:
        roster = m.get("sessions") or list(m.get("sources", {}).keys())
        sids.append(roster[0] if roster else f"s{len(sids)}")
        sources.update(m.get("sources", {}))
    order = {sid: i for i, sid in enumerate(sids)}
    for s, (objs, pts, _) in zip(sids, maps):
        print(f"session {s}: {len(objs)} objects, {len(pts)} pts", flush=True)

    objs0, pts0, cols0 = maps[0]
    scene = SceneCloud(voxel=args.voxel)
    scene.add(pts0, cols0)
    M = objs0                                                # canonical (session-0 frame)
    for i in range(1, len(maps)):
        objs_i, pts_i, cols_i = maps[i]
        if args.no_align:                                    # global poses already co-registered
            objs_i_t, pts_i_t = objs_i, pts_i
        else:
            A = align(pts_i, pts0, yaw_seeds=24, iters=80, record=False, device=args.device, verbose=True)
            T = A.transform
            print(f"align {sids[i]}->{sids[0]}: scale={T.s:.4f} yaw={np.degrees(A.yaw):.2f}deg "
                  f"inliers={A.inlier_frac:.2%}", flush=True)
            objs_i_t = [transform_object(o, T) for o in objs_i]
            pts_i_t = T.apply(pts_i)
        rec = reconcile(M, objs_i_t, ios_trust=args.ios_trust, cdiag_gate=args.cdiag_gate)
        M = rec["merged"]
        scene.add(pts_i_t, cols_i)
        print(f"  + {sids[i]}: {len(rec['matched'])} fused this step -> {len(M)} persistent objects", flush=True)

    if args.overlap_ios > 0:                                 # final label-agnostic dedup (in-merge)
        n_before = len(M)
        M = overlap_merge(M, ios_thr=args.overlap_ios)
        print(f"overlap-dedup (ios>={args.overlap_ios}): {n_before} -> {len(M)} objects", flush=True)

    if args.embed:                                           # text-label embedding (vote-weighted mean)
        try:
            from herald.scene.semantic import embed_objects
            from services.embeddings import TextEncoder
            embed_objects(M, TextEncoder(model_id=args.embed_model, device=args.device))
            print(f"embedded {len(M)} objects via {args.embed_model}", flush=True)
        except Exception as e:
            print(f"WARN: embedding skipped ({type(e).__name__}: {e})", flush=True)

    mpts, mcols = scene.cloud()
    save_session(args.out, SessionResult(mpts, mcols, M, meta={"sessions": sids, "sources": sources}))
    write_ply(args.out.with_suffix(".ply"), mpts, mcols)

    combo = Counter(tuple(sorted(o.sessions.keys())) for o in M)
    print(f"\nMERGED {len(M)} objects, {len(mpts)} voxels  ->  {args.out}", flush=True)
    for k, v in sorted(combo.items(), key=lambda x: -x[1]):
        print(f"  {'+'.join(k) or '?'}: {v}", flush=True)

    # ---- Rerun: voxel cloud + source-coloured boxes + a few resolved crops ----
    import rerun as rr
    import rerun.blueprint as rrb
    rng = np.random.default_rng(0)
    sub = rng.choice(len(mpts), min(200000, len(mpts)), replace=False)
    rr.init("herald.merge_multi")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(mpts[sub], colors=mcols[sub], radii=0.02), static=True)
    for o in M:
        ent = f"world/obj/{o.uid}"
        srcs = "+".join(sorted(o.sessions.keys()))
        rr.log(ent, rr.Boxes3D(centers=[o.center], half_sizes=[o.half_size], quaternions=[o.quat_xyzw],
                               colors=[obj_color(o.sessions.keys(), order)],
                               fill_mode="majorwireframe"), static=True)
        rr.log(ent, rr.TextDocument(f"uid {o.uid} | {o.label} | support {o.support} | sources {srcs}"),
               static=True)
        crops = object_crops(o, sources, max_n=4)
        if crops:
            rr.log(ent, rr.Image(np.concatenate(crops, axis=1)), static=True)
    out_rrd = args.rrd or args.out.with_suffix(".rrd")
    rr.save(str(out_rrd))
    legend = ", ".join(f"{s}={BASE[i]}" for i, s in enumerate(sids))
    print(f"rrd: {out_rrd}  (source colours: {legend}; blended = fused)", flush=True)


if __name__ == "__main__":
    main()
