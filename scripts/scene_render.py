#!/usr/bin/env python3
"""scene_render: render the persistent scene representation to Rerun.

Reads the persistent map (cloud + object OBBs), the area/object SceneGraph, and (optionally) the
persistent route graph, and logs: the cloud, each object OBB coloured by its area, portals as
magenta gates, and the route graph as grey lines (portal-cut edges in red). Objects show their
label + crops on selection. This is the only place rendering happens -- the pipeline scripts just
write data to disk.
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np


def _acol(uid: int):
    r, g, b = colorsys.hsv_to_rgb((uid * 0.61803) % 1.0, 0.6, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


PORTAL_RADII = 0.04                                           # thicker edge lines for portal bboxes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objects", type=Path, required=True, help="persistent map (.npz)")
    p.add_argument("--graph", type=Path, required=True, help="area/object SceneGraph (.json)")
    p.add_argument("--route", type=Path, default=None, help="persistent route graph (.json)")
    p.add_argument("--out", type=Path, required=True, help="output .rrd")
    p.add_argument("--max-points", type=int, default=200000)
    args = p.parse_args()

    import rerun as rr
    import rerun.blueprint as rrb

    from herald.scene.common.graph import SceneGraph
    from herald.scene.recon.utils import object_crops, quat_to_R
    from herald.scene.refine import load_meta, load_session
    from herald.scene.refine.portals import edge_crosses_gate

    objs, pts, cols = load_session(args.objects)
    sources = load_meta(args.objects).get("sources", {})
    graph = SceneGraph.from_json(args.graph)
    parent = {n.uid: n.parent for n in graph.nodes if n.level == "object"}
    gattrs = {n.uid: n.attrs for n in graph.nodes if n.level == "object"}
    portal_uids = {uid for uid, a in gattrs.items() if a.get("is_portal")}

    rr.init("herald.scene", spawn=False)
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    sub = np.random.default_rng(0).choice(len(pts), min(args.max_points, len(pts)), replace=False)
    rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)

    def label_crops(ent, o, label):                              # label + crops on selection
        rr.log(ent, rr.TextDocument(label), static=True)
        crops = object_crops(o, sources, max_n=3) if sources else []
        if crops:
            rr.log(ent, rr.Image(np.concatenate(crops, axis=1)), static=True)
            return 1
        return 0

    # portal gates: (center, R, half, margin) -- also used to red-flag cut route edges below
    gates = []
    ncrops = ndoor = 0
    for o in objs:
        if o.uid in portal_uids:                                  # portal -> magenta thick-lined bbox
            ent = f"world/portal/{o.uid}"
            rr.log(ent, rr.Boxes3D(centers=[o.center], half_sizes=[o.half_size], quaternions=[o.quat_xyzw],
                   colors=[(255, 0, 255)], fill_mode="majorwireframe", radii=PORTAL_RADII), static=True)
            ncrops += label_crops(ent, o, f"portal | {o.name}")
            gates.append((o.center.astype(np.float64), quat_to_R(o.quat_xyzw), o.half_size.astype(np.float64),
                          float(gattrs.get(o.uid, {}).get("gate_margin", 0.3))))
            continue
        if o.name == "door":                                      # door -> magenta solid panel
            ndoor += 1
            ent = f"world/door/{o.uid}"
            rr.log(ent, rr.Boxes3D(centers=[o.center], half_sizes=[o.half_size],
                   quaternions=[o.quat_xyzw], colors=[(255, 0, 255)], fill_mode="solid"), static=True)
            ncrops += label_crops(ent, o, "door")
            continue
        a = parent.get(o.uid)
        col = _acol(a) if a is not None else (140, 140, 140)      # free objects grey
        ent = f"world/obj/{o.uid}"
        rr.log(ent, rr.Boxes3D(centers=[o.center], half_sizes=[o.half_size], quaternions=[o.quat_xyzw],
               colors=[col], fill_mode="majorwireframe"), static=True)
        ncrops += label_crops(ent, o, f"{o.name} | area {a}")

    nkept = ncut = 0
    if args.route is not None and args.route.exists():
        from herald.scene.common.route import RouteGraph
        route = RouteGraph.from_json(args.route)
        pos = {n.id: np.asarray(n.pos, np.float64) for n in route.nodes}
        kept, cut = [], []
        for e in route.edges:
            p0, p1 = pos[e.source_id], pos[e.target_id]
            (cut if any(edge_crosses_gate(p0, p1, c, R, h, m) for c, R, h, m in gates) else kept).append([p0, p1])
        nkept, ncut = len(kept), len(cut)
        if kept:
            rr.log("world/route", rr.LineStrips3D(kept, colors=[(90, 90, 90)]), static=True)
        if cut:
            rr.log("world/route_cut", rr.LineStrips3D(cut, colors=[(230, 60, 60)]), static=True)

    n_area = sum(1 for n in graph.nodes if n.level == "area")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.out))
    print(f"rendered {len(objs)} objects in {n_area} areas ({ncrops} with crops), "
          f"{len(portal_uids)} portals, {ndoor} doors, route {nkept} kept / {ncut} cut -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
