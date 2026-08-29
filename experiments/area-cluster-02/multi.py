"""area-cluster-02 MULTI-session: fused route graph over P0000/P0001/P0002 + geodesic areas.

Aligns each session cloud onto the canonical one (Sim3, same as merge_multi3), builds a route graph
per session in the canonical frame, merges them (snap) into one fused route graph, attaches the
merged map's objects via their per-session observing poses, and clusters with complete-linkage on
the geodesic (diameter cap d_max). Needs GPU (refine.align).

    uv run python experiments/area-cluster-02/multi.py --d-max 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "route-graph-01"))
import route_graph as rg          # noqa: E402
import traj_io as tio             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, default=Path("runs/merge_multi3/merged.npz"))
    ap.add_argument("--root", type=Path, default=Path("data/tartanground"))
    ap.add_argument("--env", default="Hospital")
    ap.add_argument("--sessions", nargs="+", default=["P0000", "P0001", "P0002"])
    ap.add_argument("--camera", default="lcam_front")
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--r-snap", type=float, default=0.8)
    ap.add_argument("--d-max", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rrd", type=Path, default=HERE / "out" / "area_cluster_02_multi.rrd")
    args = ap.parse_args()

    from herald.scene.refine import align

    d = np.load(args.merged, allow_pickle=True)
    C = d["center"].astype(np.float64)
    labels = [str(x) for x in d["label"]]
    sessions = d["sessions"]
    N = len(C)
    canon = args.sessions[0]
    recon = args.root / args.env / "recon"

    # 1) align each session cloud onto canonical -> Sim3 (T[canon] = identity)
    clouds = {s: np.load(recon / f"session_{s}_v3.npz", allow_pickle=True)["points"].astype(np.float32)
              for s in args.sessions}
    T: dict = {canon: None}
    for s in args.sessions[1:]:
        A = align(clouds[s], clouds[canon], yaw_seeds=24, iters=80, record=False,
                  device=args.device, verbose=False)
        T[s] = A.transform
        print(f"align {s}->{canon}: scale={A.transform.s:.4f} inliers={A.inlier_frac:.2%}", flush=True)

    # 2) transformed poses per session; build a route graph each, merge into one fused graph
    tp: dict = {}
    for s in args.sessions:
        p, _ = tio.load_poses(tio.pose_path(args.root, args.env, s, args.camera))
        tp[s] = p if T[s] is None else T[s].apply(p)
    fused = None
    for s in args.sessions:
        g, _ = rg.build(tp[s], spacing=args.spacing, r_snap=args.r_snap, session=s, id_prefix=f"{s}_")
        if fused is None:
            fused = g
        else:
            fused.merge(g, radius=args.r_snap)
    P = fused.positions()
    csr = fused._csr()
    print(f"fused route graph: {len(fused.nodes)} nodes, {len(fused.edges)} edges, "
          f"components {fused.n_components()}", flush=True)

    # 3) attach each object via its per-session observing poses (closest approach), in canon frame
    node_idx = np.zeros(N, np.int64)
    offset = np.zeros(N, np.float64)
    for i in range(N):
        cand = []
        for s, prov in sessions[i].items():
            if s not in tp:
                continue
            for (f, _b) in prov.get("crops", []):
                f = int(f)
                if 0 <= f < len(tp[s]):
                    cand.append(tp[s][f])
        if cand:
            cand = np.asarray(cand)
            k = int(np.linalg.norm(cand - C[i], axis=1).argmin())
            anchor = cand[k]
            offset[i] = float(np.linalg.norm(C[i] - anchor))
        else:
            anchor = C[i]
        node_idx[i] = int(np.linalg.norm(P - anchor, axis=1).argmin())

    # 4) geodesic + complete-linkage (diameter cap)
    from scipy.sparse.csgraph import dijkstra
    from sklearn.cluster import AgglomerativeClustering
    rows = dijkstra(csr, directed=False, indices=node_idx)
    geo = rows[:, node_idx]
    np.fill_diagonal(geo, 0.0)
    Dm = np.minimum(geo, geo.T)
    np.fill_diagonal(Dm, 0.0)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=args.d_max,
                                  metric="precomputed", linkage="complete").fit_predict(Dm)
    n = int(lab.max()) + 1
    sizes = np.bincount(lab, minlength=n)

    def diam(c):
        idx = np.where(lab == c)[0]
        return float(geo[np.ix_(idx, idx)].max()) if len(idx) > 1 else 0.0
    dmax = max(diam(c) for c in range(n))
    prov_mult = np.array([len(sessions[i]) for i in range(N)])
    print(f"\nobjects {N} ({int((prov_mult>=2).sum())} multi-session) | "
          f"attach offset med {np.median(offset):.2f}m p90 {np.percentile(offset,90):.2f}m", flush=True)
    print(f"areas {n} | singletons {int((sizes==1).sum())} | largest {sizes.max()} | "
          f"max geodesic diameter {dmax:.1f}m (cap {args.d_max})", flush=True)
    for thr in (5, 10, 20):
        print(f"  areas with >= {thr:2d} objs: {int((sizes>=thr).sum())}", flush=True)

    # 5) render
    import colorsys

    import rerun as rr
    import rerun.blueprint as rrb
    pts, cols = d["points"], d["colors"]
    sub = np.random.default_rng(0).choice(len(pts), min(200000, len(pts)), replace=False)
    rr.init("area_cluster_02_multi")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)
    segs = [[P[a], P[b]] for a, b in fused.edge_pairs()]
    rr.log("world/route", rr.LineStrips3D(segs, colors=[(90, 90, 90)]), static=True)

    def acol(c):
        r, g, b = colorsys.hsv_to_rgb((c * 0.61803) % 1.0, 0.65, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))
    oc = np.array([acol(int(lab[i])) for i in range(N)], np.uint8)
    rr.log("world/objects", rr.Points3D(C, colors=oc, radii=0.3,
           labels=[f"{labels[i]} a{lab[i]}" for i in range(N)]), static=True)
    args.rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.rrd))
    print(f"\nrrd: {args.rrd}", flush=True)


if __name__ == "__main__":
    main()
