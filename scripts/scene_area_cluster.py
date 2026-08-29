#!/usr/bin/env python
"""scene_area_cluster: group objects into portal-bounded area nodes on the geodesic proxy.

Consumes the persistent map (objects), the persistent route graph, and the object->route-node
attachment. Reclassifies mirrors the camera walked through (or beside a door) as portals, cuts
route edges that cross a portal gate, computes the object<->object distance as the min over the
two node-sets of the (cut) route geodesic floored by Euclidean, and agglomerates within d_max.
Emits a SceneGraph: one SceneNode(level="area") per cluster, parenting its member objects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PORTAL = "entrance"
BIG = 1.0e4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=Path, required=True, help="persistent scene map (.npz)")
    ap.add_argument("--route", type=Path, required=True, help="persistent route graph (.json)")
    ap.add_argument("--attach", type=Path, required=True, help="object->route-node attachment (.json)")
    ap.add_argument("--out", type=Path, required=True, help="output area/object SceneGraph (.json)")
    ap.add_argument("--d-max", type=float, default=10.0, help="area diameter cap (m)")
    ap.add_argument("--linkage", choices=["complete", "average", "single"], default="average")
    ap.add_argument("--drop-labels", default="door,entrance,mirror",
                    help="structural labels excluded from areas (still emitted as free objects)")
    ap.add_argument("--gate-margin", type=float, default=0.3)
    ap.add_argument("--mirror-adj", type=float, default=2.0,
                    help="a mirror within this many m of a door is a portal (doorway, not a wall mirror)")
    args = ap.parse_args()

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from sklearn.cluster import AgglomerativeClustering

    from herald.scene.common.geometry import Geometry
    from herald.scene.common.graph import SceneEdge, SceneGraph, SceneNode
    from herald.scene.common.route import RouteGraph
    from herald.scene.recon.utils import quat_to_R
    from herald.scene.refine import load_session
    from herald.scene.refine.portals import edge_crosses_gate

    objs, _, _ = load_session(args.objects)
    route = RouteGraph.from_json(args.route)
    attach = json.loads(args.attach.read_text())

    labels = [o.name for o in objs]
    C = np.array([o.center for o in objs], np.float64).reshape(-1, 3)
    H = np.array([o.half_size for o in objs], np.float64).reshape(-1, 3)
    R = [quat_to_R(o.quat_xyzw) for o in objs]

    npos = np.array([n.pos for n in route.nodes], np.float32)
    nidx = {n.id: i for i, n in enumerate(route.nodes)}
    edges = [(nidx[e.source_id], nidx[e.target_id]) for e in route.edges]

    # 1. mirror -> portal: camera walked through its gate, or it is adjacent to a door
    door_c = C[[i for i, l in enumerate(labels) if l == "door"]]
    portal = [l == PORTAL for l in labels]
    n_through = n_near = 0
    for i, l in enumerate(labels):
        if l != "mirror":
            continue
        through = any(edge_crosses_gate(npos[a], npos[b], C[i], R[i], H[i], args.gate_margin) for a, b in edges)
        near = bool(len(door_c) and np.linalg.norm(door_c - C[i], axis=1).min() <= args.mirror_adj)
        if through:
            portal[i] = True; n_through += 1
        elif near:
            portal[i] = True; n_near += 1
    portal_ix = [i for i, p in enumerate(portal) if p]
    for i in portal_ix:                                        # mark for the renderer
        objs[i].attrs["is_portal"] = True
        objs[i].attrs["gate_margin"] = args.gate_margin
    print(f"portals: {len(portal_ix)} (mirror->portal: {n_through} walk-through, {n_near} near-door)", flush=True)

    # 2. cut route edges crossing a portal gate
    gates = [(C[i], R[i], H[i]) for i in portal_ix]
    crosses = lambda p0, p1: any(edge_crosses_gate(p0, p1, c, r, h, args.gate_margin) for c, r, h in gates)
    kept = [(a, b) for (a, b) in edges if not crosses(npos[a], npos[b])]
    print(f"route edges: {len(edges)} -> {len(kept)} kept ({len(edges) - len(kept)} cut)", flush=True)

    # 3. object<->object distance: observing-node->object offset + route geodesic + offset again,
    # minimised over the two anchor sets. An offset hop whose centre->node segment crosses a portal
    # is infinite (an object seen through a doorway can't anchor across it).
    drop = {s.strip() for s in args.drop_labels.split(",") if s.strip()}
    keep = [i for i, l in enumerate(labels) if l not in drop]
    node_sets = [[nidx[nid] for nid in attach.get(str(objs[i].uid), []) if nid in nidx] for i in keep]
    node_sets = [ns if ns else [int(np.linalg.norm(npos - C[i], axis=1).argmin())]
                 for i, ns in zip(keep, node_sets)]           # no attachment -> nearest node
    offw = [np.array([np.inf if crosses(C[i], npos[a]) else float(np.linalg.norm(C[i] - npos[a]))
                      for a in ns]) for i, ns in zip(keep, node_sets)]      # anchor-offset weights
    M = len(route.nodes)
    if kept:
        ij = np.array(kept, np.int64)
        w = np.linalg.norm(npos[ij[:, 0]] - npos[ij[:, 1]], axis=1)
        rows = np.concatenate([ij[:, 0], ij[:, 1]]); cols = np.concatenate([ij[:, 1], ij[:, 0]])
        csr = csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(M, M))
    else:
        csr = csr_matrix((M, M))
    all_src = sorted({a for ns in node_sets for a in ns})
    src_row = {a: r for r, a in enumerate(all_src)}
    grows = dijkstra(csr, directed=False, indices=all_src)                  # (S, M)
    N = len(keep)
    # cost to reach every route node from object i = min_a (offset_ia + geodesic(a, node))
    reach = np.stack([(grows[[src_row[a] for a in ns]] + offw[k][:, None]).min(axis=0)
                      for k, ns in enumerate(node_sets)])                   # (N, M)
    geo = np.empty((N, N))
    for j, ns in enumerate(node_sets):
        geo[:, j] = (reach[:, ns] + offw[j][None, :]).min(axis=1)           # + far-end offset
    geo = np.minimum(geo, geo.T)
    Dm = np.where(np.isinf(geo), BIG, geo)                                  # inf across portals -> BIG
    np.fill_diagonal(Dm, 0.0)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=args.d_max,
                                  metric="precomputed", linkage=args.linkage).fit_predict(Dm)
    n_area = int(lab.max()) + 1
    sizes = np.bincount(lab, minlength=n_area)
    print(f"areas {n_area} | singletons {int((sizes == 1).sum())} | largest {sizes.max()}", flush=True)

    # 4. build the area/object SceneGraph
    next_uid = max((o.uid for o in objs), default=0) + 1
    graph = SceneGraph(emb_model_id="")
    for o in objs:                                            # every object is a node
        o.parent = None
        graph.add_node(o)
    for c in range(n_area):
        members = [keep[i] for i in range(N) if lab[i] == c]
        cen = C[members]
        lo, hi = cen.min(0), cen.max(0)
        diam = float(Dm[np.ix_([i for i in range(N) if lab[i] == c],
                               [i for i in range(N) if lab[i] == c])].max()) if len(members) > 1 else 0.0
        area = SceneNode(uid=next_uid, level="area", children=[objs[m].uid for m in members],
                         geom=Geometry(type="obb", frame="NED", offset=(lo + hi) / 2,
                                       half_size=(hi - lo) / 2 + 1e-3, quat_xyzw=[0, 0, 0, 1]),
                         attrs={"diameter": round(diam, 2)})
        graph.add_node(area)
        for m in members:
            objs[m].parent = area.uid
            graph.edges.append(SceneEdge(area.uid, objs[m].uid, "contains"))
        next_uid += 1

    graph.to_json(args.out)
    print(f"scene graph -> {args.out} ({n_area} areas + {len(objs)} objects)", flush=True)


if __name__ == "__main__":
    main()
