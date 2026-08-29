"""area-cluster-02: spatial area nodes via route-graph geodesic + ratio separability.

The converged design (single session P0000):
  - distance is the ROUTE-GRAPH GEODESIC, not Euclidean: d(i,j) = off_i + graph(n_i,n_j) + off_j,
    where each object attaches to its nearest nav-node n and off is the object->node offset.
  - NO embedding term (label-mean embeddings are near-binary; they fight room diversity).
  - separability = geodesic / Euclidean (tortuosity). ratio ~1 = open connected space; ratio >> 1
    = a barrier (wall) detours the path -> separate even if Euclidean-close.
  - clustering = connected components of "link iff euclid < d_link AND ratio < rho_max". No noise:
    every object lands in an area (singletons allowed).

Compares against a Euclidean-only baseline to show what the ratio splits (wall detection).

    uv run python experiments/area-cluster-02/run.py --traj P0000 --d-link 4 --rho-max 1.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RG = HERE.parent / "route-graph-01"
sys.path.insert(0, str(RG))
sys.path.insert(0, str(HERE.parent / "area-cluster-01"))
import route_graph as rg          # noqa: E402
import traj_io as tio             # noqa: E402
import scene_io as sio            # noqa: E402


def components(adj):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n, lab = connected_components(csr_matrix(adj), directed=False)
    return lab, n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", default="Hospital")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--map", type=Path, default=None, help="session .npz (default: <root>/<env>/recon/session_<traj>_v3.npz)")
    p.add_argument("--spacing", type=float, default=1.0)
    p.add_argument("--r-snap", type=float, default=0.8)
    p.add_argument("--method", choices=["complete", "cc"], default="complete",
                   help="complete = complete-linkage geodesic with diameter cap d_max; cc = ratio connected-components")
    p.add_argument("--d-max", type=float, default=8.0, help="complete-linkage geodesic diameter cap (m)")
    p.add_argument("--ratio-cut", action="store_true", help="complete: also cut links where ratio>rho_max (scale-free barrier)")
    p.add_argument("--d-link", type=float, default=4.0, help="cc: Euclidean cap (m) to consider linking")
    p.add_argument("--rho-max", type=float, default=1.8, help="max geodesic/euclid ratio (barrier gate)")
    p.add_argument("--max-offset", type=float, default=None,
                   help="drop objects whose observing depth (offset) exceeds this (proxy for a distance-to-frustum cap)")
    p.add_argument("--rrd", type=Path, default=HERE / "out" / "area_cluster_02.rrd")
    p.add_argument("--no-viz", action="store_true")
    args = p.parse_args()

    map_path = args.map or (args.root / args.env / "recon" / f"session_{args.traj}_v3.npz")
    objs = sio.load_objects(map_path)
    C = np.array([o.center for o in objs], np.float64)
    N = len(objs)

    pos, _ = tio.load_poses(tio.pose_path(args.root, args.env, args.traj, args.camera))
    G, _ = rg.build(pos, spacing=args.spacing, r_snap=args.r_snap, session=args.traj)
    P = G.positions()
    csr = G._csr()

    # attach each object to its OBSERVING camera node: among the frames that saw it, take the
    # pose of closest approach (best view), then its nearest nav-node. offset = object->that-pose
    # distance (the viewing depth). Falls back to nearest-center if no observation refs.
    dz = np.load(map_path, allow_pickle=True)
    sessions = dz["sessions"]
    node_idx = np.zeros(N, np.int64)
    offset = np.zeros(N, np.float64)
    n_obs = 0
    for i in range(N):
        frames = [int(f) for prov in sessions[i].values() for (f, _b) in prov.get("crops", [])]
        frames = [f for f in frames if 0 <= f < len(pos)]
        if frames:
            n_obs += 1
            fp = pos[frames]                                   # observing poses
            k = int(np.linalg.norm(fp - C[i], axis=1).argmin())  # closest approach
            anchor = fp[k]
            offset[i] = float(np.linalg.norm(C[i] - anchor))
        else:
            anchor = C[i]
            offset[i] = 0.0
        node_idx[i] = int(np.linalg.norm(P - anchor, axis=1).argmin())

    if args.max_offset is not None:                    # proxy for a distance-to-frustum cap
        keep = offset <= args.max_offset
        print(f"offset cap {args.max_offset}m: keep {int(keep.sum())}/{N} objects "
              f"(drop {int((~keep).sum())} seen only from far)", flush=True)
        objs = [o for o, k in zip(objs, keep) if k]
        C = C[keep]; node_idx = node_idx[keep]; offset = offset[keep]; N = len(objs)

    from scipy.sparse.csgraph import dijkstra
    rows = dijkstra(csr, directed=False, indices=node_idx)      # (N, M)
    geo = rows[:, node_idx]                                     # (N,N) anchor-node graph geodesic
    # NB: object distance = graph distance between their anchor nav-nodes. The object->node offset
    # (viewing depth) is perpendicular to the traversal path, so it is NOT added (doing so
    # overcounts: two touching objects seen from 7 m away would read as ~14 m apart).
    np.fill_diagonal(geo, 0.0)
    eu = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(eu > 1e-6, geo / eu, 1.0)

    off = np.eye(N, dtype=bool)

    def cluster_diam(lab, n):
        """geodesic diameter of each cluster (max pairwise geodesic)."""
        out = np.zeros(n)
        for c in range(n):
            idx = np.where(lab == c)[0]
            if len(idx) > 1:
                out[c] = geo[np.ix_(idx, idx)].max()
        return out

    if args.method == "complete":
        # complete-linkage on the geodesic -> every cluster's geodesic diameter <= d_max.
        # No single-linkage chaining; wall separation is automatic (geodesic across a wall is large).
        from sklearn.cluster import AgglomerativeClustering
        Dm = geo.copy()
        if args.ratio_cut:                       # optional scale-free barrier cut before size cap
            Dm = np.where(ratio > args.rho_max, 1e6, Dm)
        Dm = np.minimum(Dm, Dm.T); np.fill_diagonal(Dm, 0.0)
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=args.d_max,
                                      metric="precomputed", linkage="complete").fit_predict(Dm)
        n = int(lab.max()) + 1
        desc = f"complete-linkage geodesic, d_max={args.d_max}" + (f" +ratio-cut>{args.rho_max}" if args.ratio_cut else "")
    else:
        link = (eu < args.d_link) & ~off & (ratio < args.rho_max) & np.isfinite(geo)
        lab, n = components(link)
        desc = f"ratio connected-components, d_link={args.d_link} rho_max={args.rho_max}"

    sizes = np.bincount(lab, minlength=n)
    diams = cluster_diam(lab, n)
    print(f"map: {map_path.name} | {N} objects | route graph {len(G.nodes)} nodes", flush=True)
    print(f"attach offset (obj->nav-node): med {np.median(offset):.2f}m  p90 {np.percentile(offset,90):.2f}m", flush=True)
    print(f"\nmethod: {desc}", flush=True)
    print(f"areas: {n}  |  singletons: {int((sizes==1).sum())}  |  largest: {sizes.max()} objs  "
          f"|  max cluster geodesic diameter: {diams.max():.1f}m", flush=True)
    for thr in (5, 10, 20):
        print(f"  areas with >= {thr:2d} objs: {int((sizes>=thr).sum())}", flush=True)
    lab_r = lab                                  # for viz

    if args.no_viz:
        return
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except Exception as e:
        print(f"(viz skipped: {type(e).__name__})", flush=True); return
    d = np.load(map_path, allow_pickle=True)
    pts, cols = d["points"], d["colors"]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(pts), min(120000, len(pts)), replace=False)
    rr.init("area_cluster_02")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)
    segs = [[P[a], P[b]] for a, b in G.edge_pairs()]
    rr.log("world/route", rr.LineStrips3D(segs, colors=[(90, 90, 90)]), static=True)
    import colorsys
    def acol(cid):
        r, g, b = colorsys.hsv_to_rgb((cid * 0.61803) % 1.0, 0.65, 0.95)
        return (int(r*255), int(g*255), int(b*255))
    ocol = np.array([acol(int(lab_r[i])) for i in range(N)], np.uint8)
    rr.log("world/objects", rr.Points3D(C, colors=ocol, radii=0.28,
           labels=[f"{objs[i].label} a{lab_r[i]}" for i in range(N)]), static=True)
    args.rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.rrd))
    print(f"\nrrd: {args.rrd}  (objects coloured by area; grey = route graph)", flush=True)


if __name__ == "__main__":
    main()
