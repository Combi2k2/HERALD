"""route-graph-01 entry: build a route graph from a session's camera trajectory.

    uv run python experiments/route-graph-01/run.py --env Hospital --traj P0000 \
        --spacing 1.0 --r-snap 0.8 --smooth-iters 10

Prints construction stats and renders a Rerun scene (raw trajectory, smoothed line, nav-nodes,
edges, and a geodesic-distance colouring from node 0) to out/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_graph as rg     # noqa: E402
import traj_io as tio        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tartanground"))
    p.add_argument("--env", default="Hospital")
    p.add_argument("--traj", default="P0000")
    p.add_argument("--camera", default="lcam_front")
    p.add_argument("--version", default="omni")
    p.add_argument("--spacing", type=float, default=1.0, help="arc-length sample spacing (m)")
    p.add_argument("--r-snap", type=float, default=0.8, help="voxel-snap radius (m); ~<= spacing")
    p.add_argument("--smooth-iters", type=int, default=10, help="Taubin iterations (0 = off)")
    p.add_argument("--cloud", type=Path, default=None, help="optional session .npz cloud to overlay")
    p.add_argument("--rrd", type=Path, default=HERE / "out" / "route_graph_01.rrd")
    p.add_argument("--no-viz", action="store_true")
    args = p.parse_args()

    pose_file = tio.pose_path(args.root, args.env, args.traj, args.camera, args.version)
    pos, _ = tio.load_poses(pose_file)
    G, node_of = rg.build(pos, spacing=args.spacing, r_snap=args.r_snap,
                          smooth_iters=args.smooth_iters, session=args.traj)

    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    elen = np.array([e.length for e in G.edges], np.float32)
    print(f"trajectory: {pose_file.name} | {len(pos)} poses | path length {seg.sum():.1f} m "
          f"| extent {(pos.max(0)-pos.min(0)).round(1)}", flush=True)
    print(f"RouteGraph: spacing={args.spacing} r_snap={args.r_snap} smooth={args.smooth_iters} "
          f"source=trajectory session={args.traj}", flush=True)
    print(f"  nodes: {len(G.nodes)} (type=nav)  |  edges: {len(G.edges)}  |  "
          f"components: {G.n_components()}", flush=True)
    if len(elen):
        print(f"  edge length  mean={elen.mean():.2f}m  min={elen.min():.2f}  max={elen.max():.2f}",
              flush=True)
    revisits = len(node_of) - len(np.unique(node_of))
    print(f"  resampled points {len(node_of)} -> {len(G.nodes)} nodes "
          f"({revisits} snapped onto existing = loop/overlap closures)", flush=True)

    # emit the RouteGraph
    json_out = HERE / "out" / "route_graph_01.json"
    G.to_json(json_out)
    print(f"  RouteGraph JSON -> {json_out}", flush=True)

    if args.no_viz:
        return
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except Exception as e:
        print(f"(viz skipped: rerun unavailable — {type(e).__name__})", flush=True)
        return
    rr.init("route_graph_01")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    if args.cloud and args.cloud.exists():
        d = np.load(args.cloud, allow_pickle=True)
        pts, cols = d["points"], d["colors"]
        sub = np.random.default_rng(0).choice(len(pts), min(120000, len(pts)), replace=False)
        rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)
    rr.log("world/traj_raw", rr.LineStrips3D([pos], colors=[(90, 90, 90)]), static=True)
    smooth = rg.taubin_smooth(pos, iters=args.smooth_iters) if args.smooth_iters else pos
    rr.log("world/traj_smooth", rr.LineStrips3D([smooth], colors=[(60, 160, 255)]), static=True)
    P = G.positions()
    # edges as line segments
    segs = [[P[i], P[j]] for i, j in G.edge_pairs()]
    rr.log("world/edges", rr.LineStrips3D(segs, colors=[(255, 170, 40)]), static=True)
    # nodes coloured by geodesic distance from node 0
    dist = G.geodesic(0)
    finite = np.isfinite(dist)
    dn = np.zeros_like(dist); dn[finite] = dist[finite] / (dist[finite].max() or 1.0)
    col = np.zeros((len(P), 3), np.uint8)
    col[:, 0] = (dn * 255).astype(np.uint8); col[:, 2] = ((1 - dn) * 255).astype(np.uint8)
    rr.log("world/nodes", rr.Points3D(P, colors=col, radii=0.25), static=True)
    args.rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.rrd))
    print(f"rrd: {args.rrd}  (nodes coloured by geodesic dist from node 0: blue=near, red=far)", flush=True)


if __name__ == "__main__":
    main()
