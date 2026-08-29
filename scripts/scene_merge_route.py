#!/usr/bin/env python
"""scene_merge_route: absorb a new session's camera trajectory into the persistent route graph,
and add that session's observing nav-nodes to each object it saw (multi-anchor attachment).

The route graph is the geodesic backbone for area clustering. Absorb is append-only (nodes keep
their ids), so object attachments never need remapping. Pose file is derived from the persistent
map's stored source path, so no dataset args are needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objects", type=Path, required=True, help="persistent scene map (.npz)")
    p.add_argument("--route", type=Path, required=True, help="persistent route graph (.json), updated in place")
    p.add_argument("--attach", type=Path, required=True, help="object->route-node attachment (.json), updated in place")
    p.add_argument("--session", required=True, help="session id being integrated")
    p.add_argument("--sim3", type=Path, required=True, help="Sim3 from scene_align (.json)")
    p.add_argument("--spacing", type=float, default=1.0, help="nav-node spacing (m)")
    p.add_argument("--r-connect", type=float, default=2.0, help="cross-session proximity radius (m)")
    args = p.parse_args()

    from herald.scene.common.geometry import Sim3
    from herald.scene.common.route import RouteGraph
    from herald.scene.refine import load_meta, load_session
    from herald.scene.refine.route import build_route

    objs, _, _ = load_session(args.objects)
    meta = load_meta(args.objects)
    src = meta.get("sources", {}).get(args.session)
    if not src:
        raise SystemExit(f"no source path for session {args.session!r} in {args.objects}")
    img_dir = Path(src)
    camera = img_dir.name.replace("image_", "")
    pos = np.loadtxt(img_dir.parent / f"pose_{camera}.txt", dtype=np.float64)[:, :3].astype(np.float32)

    d = json.loads(args.sim3.read_text())
    T = Sim3(s=d["scale"], R=np.asarray(d["R"]), t=np.asarray(d["t"]))
    pos = T.apply(pos).astype(np.float32)

    new = build_route(pos, session=args.session, spacing=args.spacing)
    route = RouteGraph.from_json(args.route) if args.route.exists() else RouteGraph()
    route.absorb(new)
    min_gap = int(round(args.r_connect / max(args.spacing, 1e-6))) + 1
    n_prox = route.add_proximity_edges(args.r_connect, min_seq_gap=min_gap)
    route.to_json(args.route)

    # multi-anchor attach: nearest node of each observing-frame pose, added to the object's set
    node_pos = np.array([n.pos for n in route.nodes], np.float32)
    node_ids = [n.id for n in route.nodes]
    attach = json.loads(args.attach.read_text()) if args.attach.exists() else {}
    n_attached = 0
    for o in objs:
        ref = next((r for r in o.refs if r.assigned_by == args.session), None)
        if ref is None:
            continue
        cur = set(attach.get(str(o.uid), []))
        for f in ref.metadata.get("frames", []):
            if 0 <= int(f) < len(pos):
                cur.add(node_ids[int(np.linalg.norm(node_pos - pos[int(f)], axis=1).argmin())])
        if cur:
            attach[str(o.uid)] = sorted(cur)
            n_attached += 1
    args.attach.write_text(json.dumps(attach))
    print(f"absorbed {args.session}: route now {len(route.nodes)} nodes, {len(route.edges)} edges "
          f"(+{n_prox} proximity); attached {n_attached} objects", flush=True)


if __name__ == "__main__":
    main()
