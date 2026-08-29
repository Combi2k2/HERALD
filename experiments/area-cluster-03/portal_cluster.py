"""Portal-bounded area clustering  (experiment area-cluster-03, CPU only).

Extends area-cluster-02: areas are separated by PORTALS, not just geodesic distance.
  1. mirror -> portal disambiguation: a `mirror` OBB adjacent to a `door` becomes `portal`.
  2. portal barrier: cut every route edge that passes through a portal's gate, so the node-set
     geodesic returns infinity across portal-bounded components -> cross-portal objects can never
     share an area.

Consumes the merged map + fused route graph + per-object node-sets that merge_routes.py wrote.

    uv run python experiments/area-cluster-03/portal_cluster.py \
        --merged data/tartanground/Hospital/recon/merged_portal5.npz \
        --art-dir experiments/area-cluster-02/out/portal5 --d-max 5
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "route-graph-01"))
from route_types import RouteGraph                          # noqa: E402

from herald.scene.recon.utils import quat_to_R              # noqa: E402

BIG = 1.0e4                                                 # stand-in for "infinite" (cut) distance
PORTAL = "entrance"                                         # doorway label OWL detects; the area separator


@lru_cache(maxsize=8)
def _frames(folder: str):
    p = Path(folder)
    return sorted(p.glob("*.png")) or sorted(p.glob("*.jpg"))


def _crop(folder: str, fid: int, box, size: int = 112):
    from PIL import Image as PILImage
    files = _frames(folder)
    if not (0 <= fid < len(files)):
        return None
    try:
        im = PILImage.open(files[fid]).convert("RGB")
        x1, y1, x2, y2 = (float(v) for v in box)
        return np.asarray(im.crop((x1, y1, x2, y2)).resize((size, size)))
    except Exception:
        return None


def object_crops(prov, sources, *, max_n: int = 3, size: int = 112):
    out = []
    for sid, p in prov.items():
        folder = sources.get(sid)
        if not folder:
            continue
        for (fid, box) in p.get("crops", []):
            c = _crop(folder, int(fid), box, size)
            if c is not None:
                out.append(c)
            if len(out) >= max_n:
                return out
    return out


def edge_crosses_gate(p0, p1, center, R, half, margin) -> bool:
    """True if segment p0->p1 passes through the portal's gate: it crosses the panel plane
    (normal = OBB thin axis) within the two in-plane half-extents (+margin)."""
    thin = int(np.argmin(half))
    n = R[:, thin]
    d = p1 - p0
    denom = float(n @ d)
    if abs(denom) < 1e-9:
        return False                                        # parallel to the panel plane
    t = float(n @ (center - p0)) / denom
    if not (0.0 <= t <= 1.0):
        return False                                        # plane crossing not within this edge
    rel = (p0 + t * d) - center
    for ax in range(3):
        if ax == thin:
            continue
        if abs(float(rel @ R[:, ax])) > half[ax] + margin:
            return False                                    # outside the doorway opening
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--art-dir", type=Path, required=True, help="dir with merged_route.json + object_attach.json")
    ap.add_argument("--d-max", type=float, default=5.0)
    ap.add_argument("--linkage", choices=["complete", "average", "single"], default="average")
    ap.add_argument("--drop-labels", default="door,entrance,mirror",
                    help="labels excluded from the clustered object set (structural, not area members)")
    ap.add_argument("--mirror-adj", type=float, default=1.5,
                    help="a mirror within this many m of a door is re-classified portal")
    ap.add_argument("--gate-margin", type=float, default=0.3,
                    help="expand each portal gate by this many m (opening usually wider than the panel)")
    ap.add_argument("--rrd", type=Path, default=HERE / "out" / "portal_cluster.rrd")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    d = np.load(args.merged, allow_pickle=True)
    labels = np.array([str(x) for x in d["label"]])
    C_all = d["center"].astype(np.float64)
    H_all = d["half_size"].astype(np.float64)
    Q_all = d["quat_xyzw"].astype(np.float64)
    att = json.loads((args.art_dir / "object_attach.json").read_text())

    G = RouteGraph.from_json(args.art_dir / "merged_route.json")
    P = G.positions()
    edge_idx = [(G._idx[e.source_id], G._idx[e.target_id]) for e in G.edges]

    # ---- 1. mirror -> portal disambiguation ----
    # A flat 'mirror' panel is really a doorway (portal) if EITHER (a) the camera walked THROUGH it
    # -- a route edge crosses its gate (you cannot walk through a mirror; direct evidence), OR
    # (b) it sits adjacent to a 'door'. Walk-through is the stronger signal, so it's counted first.
    door_c = C_all[labels == "door"]
    n_through = n_near = 0
    for i in np.where(labels == "mirror")[0]:
        Ri, hi, ci = quat_to_R(Q_all[i]), H_all[i], C_all[i]
        through = any(edge_crosses_gate(P[a], P[b], ci, Ri, hi, args.gate_margin) for a, b in edge_idx)
        near = bool(len(door_c) and np.linalg.norm(door_c - ci, axis=1).min() <= args.mirror_adj)
        if through:
            labels[i] = PORTAL; n_through += 1
        elif near:
            labels[i] = PORTAL; n_near += 1
    portal_ix = np.where(labels == PORTAL)[0]
    print(f"portals: {len(portal_ix)} (mirror->portal: {n_through} walk-through, {n_near} near-door)", flush=True)

    # ---- 2. cut route edges that pass through any portal gate ----
    portals = [(C_all[i], quat_to_R(Q_all[i]), H_all[i]) for i in portal_ix]
    kept, cut = [], 0
    for (a, b), e in zip(edge_idx, G.edges):
        if any(edge_crosses_gate(P[a], P[b], c, R, h, args.gate_margin) for (c, R, h) in portals):
            cut += 1
            continue
        w = e.weight if e.weight is not None else e.length
        kept.append((a, b, float(w)))
    print(f"route edges: {len(G.edges)} -> {len(kept)} kept ({cut} cut by portals)", flush=True)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    M = len(G.nodes)
    if kept:
        ij = np.array([(a, b) for a, b, _ in kept])
        w = np.array([wt for _, _, wt in kept])
        rows = np.concatenate([ij[:, 0], ij[:, 1]])
        cols = np.concatenate([ij[:, 1], ij[:, 0]])
        data = np.concatenate([w, w])
    else:
        rows = cols = np.zeros(0, int); data = np.zeros(0)
    csr = csr_matrix((data, (rows, cols)), shape=(M, M))
    ncomp = connected_components(csr, directed=False, return_labels=False)
    print(f"portal-bounded route components: {ncomp}", flush=True)

    # ---- 3. cluster the real objects (drop structural labels) ----
    drop = {s.strip() for s in args.drop_labels.split(",") if s.strip()}
    keep = np.array([lab not in drop for lab in labels])
    C = C_all[keep]
    half = H_all[keep].astype(np.float32)
    quat = Q_all[keep].astype(np.float32)
    uids = [int(x) for x, k in zip(d["uid"], keep) if k]
    sessions = [s for s, k in zip(d["sessions"], keep) if k]
    lbl = labels[keep]
    if "nodes" in att:
        node_sets_all = [ns for ns, k in zip(att["nodes"], keep) if k]
    else:
        node_sets_all = [[nid] for nid, k in zip(att["node"], keep) if k]
    node_sets = [[G._idx[nid] for nid in ns] for ns in node_sets_all]
    N = len(C)
    print(f"dropped {int((~keep).sum())} structural objs {sorted(drop)} -> {N} objects to cluster", flush=True)

    # node-set geodesic on the CUT graph: min over the two node-sets, ∞ across portal components
    all_src = sorted({a for ns in node_sets for a in ns})
    src_row = {a: r for r, a in enumerate(all_src)}
    grows = dijkstra(csr, directed=False, indices=all_src)              # (S, M); inf across components
    obj2node = np.stack([grows[[src_row[a] for a in ns]].min(axis=0) for ns in node_sets])  # (N, M)
    geo = np.empty((N, N), float)
    for j, ns in enumerate(node_sets):
        geo[:, j] = obj2node[:, ns].min(axis=1)
    geo = np.minimum(geo, geo.T)
    eu = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    Dm = np.maximum(geo, eu)                                            # Euclidean floor
    Dm = np.where(np.isinf(Dm), BIG, Dm)                               # ∞ -> large finite for sklearn
    np.fill_diagonal(Dm, 0.0)

    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=args.d_max,
                                  metric="precomputed", linkage=args.linkage).fit_predict(Dm)
    n = int(lab.max()) + 1
    sizes = np.bincount(lab, minlength=n)
    n_cross = int((geo >= BIG).sum() // 2 + (np.isinf(geo)).sum() // 2)  # informational
    print(f"areas {n} | singletons {int((sizes==1).sum())} | largest {sizes.max()}", flush=True)
    for thr in (5, 10, 20):
        print(f"  areas with >= {thr:2d} objs: {int((sizes>=thr).sum())}", flush=True)

    if args.no_viz:
        return
    import colorsys

    import rerun as rr
    import rerun.blueprint as rrb
    pts, cols = d["points"], d["colors"]
    sources = d["meta"].item().get("sources", {})
    sub = np.random.default_rng(0).choice(len(pts), min(200000, len(pts)), replace=False)
    rr.init("portal_cluster")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)
    # kept route edges (grey) vs portal-cut edges (red)
    kept_pairs = {(a, b) for a, b, _ in kept} | {(b, a) for a, b, _ in kept}
    rr.log("world/route", rr.LineStrips3D([[P[a], P[b]] for a, b in kept_pairs],
           colors=[(90, 90, 90)]), static=True)
    cut_seg = [[P[G._idx[e.source_id]], P[G._idx[e.target_id]]] for e in G.edges
               if (G._idx[e.source_id], G._idx[e.target_id]) not in kept_pairs]
    if cut_seg:
        rr.log("world/route_cut", rr.LineStrips3D(cut_seg, colors=[(230, 60, 60)]), static=True)
    # portal gates (magenta boxes)
    for i in portal_ix:
        rr.log(f"world/portal/{int(d['uid'][i])}", rr.Boxes3D(
            centers=[C_all[i]], half_sizes=[H_all[i]], quaternions=[Q_all[i]],
            colors=[(255, 0, 255)], fill_mode="solid"), static=True)

    def acol(c):
        r, g, b = colorsys.hsv_to_rgb((c * 0.61803) % 1.0, 0.65, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))

    for i in range(N):
        ent = f"world/obj/{uids[i]}"
        rr.log(ent, rr.Boxes3D(centers=[C[i]], half_sizes=[half[i]], quaternions=[quat[i]],
               colors=[acol(int(lab[i]))], fill_mode="majorwireframe"), static=True)
        rr.log(ent, rr.TextDocument(f"{lbl[i]} | area {lab[i]}"), static=True)
        crops = object_crops(sessions[i], sources, max_n=3)
        if crops:
            rr.log(ent, rr.Image(np.concatenate(crops, axis=1)), static=True)
    args.rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.rrd))
    print(f"rrd: {args.rrd}", flush=True)


if __name__ == "__main__":
    main()
