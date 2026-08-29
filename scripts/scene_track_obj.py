#!/usr/bin/env python
"""scene_track_obj: integrate a new recon session's objects into the persistent object set.

Applies the scene_align Sim3, reconciles the new objects against the persistent set (two-pass
gate + union-find), runs the in-merge overlap dedup, accumulates the cloud, and writes the
persistent map back in place. Persistent object uids stay stable across integrations; new
objects get fresh uids. Seeds the map from the session if none exists yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True, help="new recon dump (.npz)")
    p.add_argument("--persistent", type=Path, required=True, help="persistent scene map (.npz), updated in place")
    p.add_argument("--sim3", type=Path, required=True, help="Sim3 from scene_align (.json)")
    p.add_argument("--ios-trust", type=float, default=0.9, help="pass-1 IoS to fuse regardless of label")
    p.add_argument("--cdiag-gate", type=float, default=0.5, help="pass-2 centre-dist/diag co-location gate")
    p.add_argument("--overlap-ios", type=float, default=0.5, help="final label-agnostic dedup IoS (0 disables)")
    p.add_argument("--voxel", type=float, default=0.1, help="merged-cloud voxel (m)")
    args = p.parse_args()

    from herald.scene.common.geometry import Sim3
    from herald.scene.recon.types import SceneMap
    from herald.scene.recon.utils import SceneCloud
    from herald.scene.refine import (
        load_meta, load_session, overlap_merge, reconcile, save_session, transform_object)

    sobjs, spts, scols = load_session(args.session)
    smeta = load_meta(args.session)
    d = json.loads(args.sim3.read_text())
    T = Sim3(s=d["scale"], R=np.asarray(d["R"]), t=np.asarray(d["t"]))

    scene = SceneCloud(voxel=args.voxel)
    if args.persistent.exists():
        pobjs, ppts, pcols = load_session(args.persistent)
        pmeta = load_meta(args.persistent)
        scene.add(ppts, pcols)
        sobjs_t = [transform_object(o, T) for o in sobjs]
        rec = reconcile(pobjs, sobjs_t, ios_trust=args.ios_trust, cdiag_gate=args.cdiag_gate)
        M = overlap_merge(rec["merged"], ios_thr=args.overlap_ios)
        scene.add(T.apply(spts), scols)
        sessions = list(dict.fromkeys(list(pmeta.get("sessions", [])) + list(smeta.get("sessions", []))))
        sources = {**pmeta.get("sources", {}), **smeta.get("sources", {})}
        print(f"integrate {len(sobjs)} new objs into {len(pobjs)} persistent -> "
              f"{len(rec['merged'])} ({len(rec['matched'])} fused) -> {len(M)} after dedup", flush=True)
    else:
        M = sobjs
        scene.add(spts, scols)
        sessions = list(smeta.get("sessions", []))
        sources = dict(smeta.get("sources", {}))
        print(f"seed persistent map with {len(M)} objects", flush=True)

    mpts, mcols = scene.cloud()
    save_session(args.persistent, SceneMap(mpts, mcols, M, meta={"sessions": sessions, "sources": sources}))
    print(f"persistent map -> {args.persistent} ({len(M)} objects, {len(mpts)} voxels)", flush=True)


if __name__ == "__main__":
    main()
