#!/usr/bin/env python3
"""Tier-2 refine end-to-end at the OBJECT level: align two recon'd sessions, carry one
session's OBBs through the Sim3, and reconcile the boxes into one persistent set.

  1. load two saved sessions (objects + cloud) written by session_recon.py --dump,
  2. `align` session B's cloud onto A -> Sim3,
  3. transform B's OBBs by that Sim3, then `reconcile` against A's OBBs (Hungarian match on
     centre distance, label-gated) -> fused boxes + singletons,
  4. Rerun: the merged cloud + the reconciled OBBs colour-coded by provenance
     (green = fused/seen-in-both, blue = only A, orange = only B),
  5. save the merged object set + merged cloud.

  uv run python scripts/merge_objects.py --a session_P0000.npz --b session_P0001.npz --out merged.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from herald.scene.recon.types import SessionResult
from herald.scene.recon.utils import SceneCloud, write_ply
from herald.scene.refine import (
    align, load_meta, load_session, reconcile, save_session, transform_object)


def _boxes(objs):
    """(centers, half_sizes, quats) arrays for a list of SceneObjects."""
    if not objs:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    return (np.array([o.center for o in objs], np.float32),
            np.array([o.half_size for o in objs], np.float32),
            np.array([o.quat_xyzw for o in objs], np.float32))


def _log_boxes(rr, ent, objs, color):
    c, h, q = _boxes(objs)
    rr.log(ent, rr.Boxes3D(centers=c, half_sizes=h, quaternions=q, colors=[color],
                           fill_mode="majorwireframe",
                           labels=[f"{o.label} (sup {o.support})" for o in objs]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, default=Path("data/tartanground/Hospital/recon/session_P0000.npz"),
                   help="target/canonical session")
    p.add_argument("--b", type=Path, default=Path("data/tartanground/Hospital/recon/session_P0001.npz"),
                   help="session aligned + reconciled onto A")
    p.add_argument("--ios-trust", type=float, default=0.5, help="pass 1: IoS above which boxes fuse regardless of label")
    p.add_argument("--cdiag-gate", type=float, default=0.5, help="pass 2: centre-dist / mean-diagonal below which co-located boxes with agreeing labels fuse")
    p.add_argument("--voxel", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True, help="merged session .npz")
    p.add_argument("--rrd", type=Path, default=None)
    p.add_argument("--no-embed", dest="embed", action="store_false",
                   help="skip text-label embeddings (default: embed via text-only model)")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    objs_a, pts_a, cols_a = load_session(args.a)
    objs_b, pts_b, cols_b = load_session(args.b)
    print(f"A {args.a.name}: {len(objs_a)} objects, {len(pts_a)} pts | "
          f"B {args.b.name}: {len(objs_b)} objects, {len(pts_b)} pts", flush=True)

    # 1-2) align B's cloud onto A
    A = align(pts_b, pts_a, yaw_seeds=24, iters=args.iters, record=False,
              device=args.device, verbose=True)
    T = A.transform
    print(f"\nSim3: scale={T.s:.4f} yaw={np.degrees(A.yaw):.2f}deg t={T.t.round(3)} "
          f"inliers={A.inlier_frac:.2%}", flush=True)

    # 3) carry B's OBBs through the Sim3, then reconcile against A
    objs_b_t = [transform_object(o, T) for o in objs_b]
    rec = reconcile(objs_a, objs_b_t, ios_trust=args.ios_trust, cdiag_gate=args.cdiag_gate)
    print(f"RECONCILE (pass1 IoS>={args.ios_trust} / pass2 cdiag<{args.cdiag_gate}): {len(rec['matched'])} fused "
          f"(seen in both) | {len(rec['only_a'])} only-A | {len(rec['only_b'])} only-B  ->  "
          f"{len(rec['merged'])} persistent objects", flush=True)
    for oa, ob, ios in rec["matched"][:12]:
        print(f"  fuse '{oa.label}'~'{ob.label}'  IoS={ios:.2f}  sup {oa.support}+{ob.support}", flush=True)

    # 4) merged cloud (voxel-snapped) + save merged session
    scene = SceneCloud(voxel=args.voxel)
    scene.add(pts_a, cols_a)
    scene.add(T.apply(pts_b), cols_b)
    if args.embed:                                              # text-label embedding (vote-weighted mean)
        try:
            from herald.scene.semantic import embed_objects
            from services.embeddings import TextEncoder
            embed_objects(rec["merged"], TextEncoder(model_id=args.embed_model, device=args.device))
            print(f"embedded {len(rec['merged'])} objects via {args.embed_model}", flush=True)
        except Exception as e:
            print(f"WARN: embedding skipped ({type(e).__name__}: {e})", flush=True)

    mpts, mcols = scene.cloud()
    sess = sorted({s for o in rec["merged"] for s in o.sessions})    # contributing sessions
    srcs = {}
    for pth in (args.a, args.b):
        srcs.update(load_meta(pth).get("sources", {}))
    save_session(args.out, SessionResult(mpts, mcols, rec["merged"],
                                         meta={"sessions": sess, "sources": srcs}))
    write_ply(args.out.with_suffix(".ply"), mpts, mcols)
    print(f"MERGED cloud {len(mpts)} voxels + {len(rec['merged'])} objects  ->  {args.out}", flush=True)

    # 5) Rerun: cloud backdrop + reconciled boxes colour-coded by provenance
    import rerun as rr
    import rerun.blueprint as rrb
    rng = np.random.default_rng(0)
    sub = rng.choice(len(mpts), min(200000, len(mpts)), replace=False)
    rr.init("herald.merge_objects")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(mpts[sub], colors=mcols[sub], radii=0.02), static=True)
    fused = rec["merged"][: len(rec["matched"])]          # merged = fused ++ only_a ++ only_b
    _log_boxes(rr, "world/fused", fused, (70, 210, 90))
    _log_boxes(rr, "world/only_a", rec["only_a"], (80, 130, 255))
    _log_boxes(rr, "world/only_b", rec["only_b"], (240, 150, 40))
    out_rrd = args.rrd or args.out.with_suffix(".rrd")
    rr.save(str(out_rrd))
    print(f"rrd: {out_rrd}  (green=fused, blue=only-A, orange=only-B)", flush=True)


if __name__ == "__main__":
    main()
