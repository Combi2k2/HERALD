"""Area clustering that CONSUMES a persisted merged route map (CPU only, no align/torch).

Loads the merged SceneMap objects + the fused RouteGraph + per-object attachment written by
merge_routes.py, computes the geodesic between anchor nav-nodes, and clusters with complete-linkage
(diameter cap d_max). Prints stats and renders.

    uv run python experiments/area-cluster-02/cluster_areas.py --d-max 8
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
from route_types import RouteGraph      # noqa: E402


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
    """A few resolved crop thumbnails for one object's per-session provenance."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, default=Path("runs/merge_multi3/merged.npz"))
    ap.add_argument("--art-dir", type=Path, default=HERE / "out", help="dir with merged_route.json + object_attach.json")
    ap.add_argument("--d-max", type=float, default=8.0)
    ap.add_argument("--linkage", choices=["complete", "average", "single"], default="complete",
                    help="cluster-distance rule: complete=farthest pair (compact/capped), "
                         "average=mean pair, single=nearest pair (chains)")
    ap.add_argument("--drop-labels", default="door,entrance,mirror",
                    help="comma-separated labels to drop before clustering (portals/wall fixtures)")
    ap.add_argument("--rrd", type=Path, default=HERE / "out" / "area_cluster_02_merged.rrd")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    d = np.load(args.merged, allow_pickle=True)
    labels = np.array([str(x) for x in d["label"]])
    att = json.loads((args.art_dir / "object_attach.json").read_text())

    # drop unwanted labels (door/entrance/mirror) before anything else, filtering all per-object arrays in sync
    drop = {s.strip() for s in args.drop_labels.split(",") if s.strip()}
    keep = np.array([lab not in drop for lab in labels])
    n_drop = int((~keep).sum())
    C = d["center"].astype(np.float64)[keep]
    half = d["half_size"].astype(np.float32)[keep]
    quat = d["quat_xyzw"].astype(np.float32)[keep]
    uids = [int(x) for x, k in zip(d["uid"], keep) if k]
    sessions = [s for s, k in zip(d["sessions"], keep) if k]
    labels = labels[keep]
    offset = np.asarray(att["offset"], float)[keep]
    # each object carries a SET of observing nav-nodes (every frame that saw it, snapped to the
    # fused route graph). Back-compat: accept the old single-"node" schema as a 1-element set.
    if "nodes" in att:
        node_sets_all = [ns for ns, k in zip(att["nodes"], keep) if k]
    else:
        node_sets_all = [[nid] for nid, k in zip(att["node"], keep) if k]
    N = len(C)
    print(f"dropped {n_drop} objects with labels {sorted(drop)} -> {N} objects remain", flush=True)

    G = RouteGraph.from_json(args.art_dir / "merged_route.json")
    node_sets = [[G._idx[nid] for nid in ns] for ns in node_sets_all]   # object -> node list-indices
    print(f"anchors/obj: med {int(np.median([len(s) for s in node_sets]))} "
          f"max {max(len(s) for s in node_sets)}", flush=True)

    from scipy.sparse.csgraph import dijkstra
    from sklearn.cluster import AgglomerativeClustering
    csr = G._csr()
    # object<->object geodesic = MIN over the two node-sets of the fused-graph node geodesic.
    # Run Dijkstra once from every distinct anchor node, reduce to per-object distance-to-all-nodes
    # (min over the object's own set), then reduce again over each target object's set.
    all_src = sorted({a for ns in node_sets for a in ns})
    src_row = {a: r for r, a in enumerate(all_src)}
    rows = dijkstra(csr, directed=False, indices=all_src)               # (S, M)
    obj2node = np.stack([rows[[src_row[a] for a in ns]].min(axis=0) for ns in node_sets])  # (N, M)
    geo = np.empty((N, N), float)
    for j, ns in enumerate(node_sets):
        geo[:, j] = obj2node[:, ns].min(axis=1)
    geo = np.minimum(geo, geo.T)                                        # symmetrize numerics
    # floor the proxy at the straight-line separation: true free-space distance is never below
    # Euclidean, and the node-geodesic loses that when two objects share a node (reports 0).
    eu = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    Dm = np.maximum(geo, eu)
    np.fill_diagonal(Dm, 0.0)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=args.d_max,
                                  metric="precomputed", linkage=args.linkage).fit_predict(Dm)
    n = int(lab.max()) + 1
    sizes = np.bincount(lab, minlength=n)

    def diam(c):
        ix = np.where(lab == c)[0]
        return float(Dm[np.ix_(ix, ix)].max()) if len(ix) > 1 else 0.0
    dmax = max(diam(c) for c in range(n))
    prov = np.array([len(sessions[i]) for i in range(N)])
    print(f"consumed: {args.art_dir}/merged_route.json ({len(G.nodes)} nodes) + object_attach.json", flush=True)
    print(f"objects {N} ({int((prov>=2).sum())} multi-session) | "
          f"attach offset med {np.median(offset):.2f}m p90 {np.percentile(offset,90):.2f}m", flush=True)
    print(f"areas {n} | singletons {int((sizes==1).sum())} | largest {sizes.max()} | "
          f"max geodesic diameter {dmax:.1f}m (cap {args.d_max})", flush=True)
    for thr in (5, 10, 20):
        print(f"  areas with >= {thr:2d} objs: {int((sizes>=thr).sum())}", flush=True)

    if args.no_viz:
        return
    import colorsys

    import rerun as rr
    import rerun.blueprint as rrb
    P = G.positions()
    pts, cols = d["points"], d["colors"]
    sources = d["meta"].item().get("sources", {})
    sub = np.random.default_rng(0).choice(len(pts), min(200000, len(pts)), replace=False)
    rr.init("area_cluster_02_merged")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)
    rr.log("world/route", rr.LineStrips3D([[P[a], P[b]] for a, b in G.edge_pairs()],
           colors=[(90, 90, 90)]), static=True)

    def acol(c):
        r, g, b = colorsys.hsv_to_rgb((c * 0.61803) % 1.0, 0.65, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))

    # Per object: the OBB box (area-coloured) carries BOTH its text label and its crop thumbnails.
    # No centroid.
    ncrops = 0
    for i in range(N):
        col = acol(int(lab[i]))
        ent = f"world/obj/{uids[i]}"
        # OBB box (no floating text in the 3D viewport)
        rr.log(ent, rr.Boxes3D(centers=[C[i]], half_sizes=[half[i]], quaternions=[quat[i]],
               colors=[col], fill_mode="majorwireframe"), static=True)
        # label + crops attached to the entity -> shown in the selection panel, not the viewport
        rr.log(ent, rr.TextDocument(f"{labels[i]} | area {lab[i]}"), static=True)
        crops = object_crops(sessions[i], sources, max_n=3)
        if crops:
            ncrops += 1
            rr.log(ent, rr.Image(np.concatenate(crops, axis=1)), static=True)
    print(f"rendered {N} OBBs (+label/crops on selection, no viewport text) ({ncrops} with crops)", flush=True)
    args.rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.rrd))
    print(f"rrd: {args.rrd}", flush=True)


if __name__ == "__main__":
    main()
