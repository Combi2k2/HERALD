"""Tier-2 ROUTE-GRAPH MERGE (experiment): produce the fused route map that area clustering consumes.

Aligns each session cloud onto the canonical one (Sim3), builds a route graph per session in the
canonical frame, merges them into ONE fused RouteGraph, attaches the merged map's objects to it
(via per-session observing poses), and PERSISTS three artifacts to --out-dir:
  - merged_route.json   the fused RouteGraph (nodes+edges, canonical frame)
  - object_attach.json  per-object anchor node id + offset (the object<->nav-node link)
  - sim3.json           the per-session Sim3 (scale, R, t) -- what the object merge threw away

Needs GPU (refine.align). Run once; clustering then consumes the artifacts on CPU.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--r-connect", "--r-snap", dest="r_connect", type=float, default=2.0,
                    help="proximity radius (m): nodes within this get a loop-closure/cross-session "
                         "edge (~corridor width). No node fusion; graph is append-only.")
    ap.add_argument("--dump-tag", default="v3", help="session cloud dump tag: session_<S>_<tag>.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out")
    ap.add_argument("--no-align", action="store_true",
                    help="skip Sim3 align, use identity (TartanGround global poses are pre-registered)")
    args = ap.parse_args()

    d = np.load(args.merged, allow_pickle=True)
    C = d["center"].astype(np.float64)
    uids = [int(x) for x in d["uid"]]
    sessions = d["sessions"]
    N = len(C)
    canon = args.sessions[0]
    recon = args.root / args.env / "recon"
    _identity = {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0], "inliers": 1.0}

    # 1) put every session in the canonical frame. TartanGround gives GLOBAL per-environment poses,
    # so sessions are already co-registered (measured Sim3 ~identity: <0.1deg, <11cm) -> --no-align
    # skips the costly RANSAC and uses identity. Real (independently-originated) sessions need align.
    if args.no_align:
        T = {s: None for s in args.sessions}                             # None == identity transform
        sim3 = {s: dict(_identity) for s in args.sessions}
    else:
        from herald.scene.refine import align
        clouds = {s: np.load(recon / f"session_{s}_{args.dump_tag}.npz",
                             allow_pickle=True)["points"].astype(np.float32) for s in args.sessions}
        T = {canon: None}
        sim3 = {canon: dict(_identity)}
        for s in args.sessions[1:]:
            A = align(clouds[s], clouds[canon], yaw_seeds=24, iters=80, record=False,
                      device=args.device, verbose=False)
            T[s] = A.transform
            sim3[s] = {"scale": float(A.transform.s), "R": np.asarray(A.transform.R).tolist(),
                       "t": np.asarray(A.transform.t).tolist(), "inliers": float(A.inlier_frac)}
        print(f"align {s}->{canon}: scale={A.transform.s:.4f} inliers={A.inlier_frac:.2%}", flush=True)

    # 2) transformed poses -> per-session OPEN route graph -> APPEND-ONLY union -> proximity edges.
    # No node fusion: every session's resampled nodes keep their measured position and a stable id;
    # loop closure and cross-session connectivity are added as proximity edges. This makes the graph
    # append-only (a new session never moves an existing node, so object attachments never remap).
    tp: dict = {}
    for s in args.sessions:
        p, _ = tio.load_poses(tio.pose_path(args.root, args.env, s, args.camera))
        tp[s] = p if T[s] is None else T[s].apply(p)
    fused = None
    for s in args.sessions:
        g = rg.build_open(tp[s], spacing=args.spacing, session=s, id_prefix=f"{s}_")
        if fused is None:
            fused = g
        else:
            fused.absorb(g)
    min_gap = int(round(args.r_connect / max(args.spacing, 1e-6))) + 1
    n_prox = fused.add_proximity_edges(args.r_connect, min_seq_gap=min_gap)
    print(f"route graph (append-only): {len(fused.nodes)} nodes, {len(fused.edges)} edges "
          f"({n_prox} proximity), components {fused.n_components()}", flush=True)

    # 3) MULTI-ANCHOR attach: every observing frame -> nearest fused nav-node (global; ids are stable
    # so no remap). The object's anchor is the SET of nav-nodes that saw it. Falls back to the <=3
    # crop refs when a dump predates observing-frame persistence, then to the object centre.
    Pf = fused.positions()
    node_sets, offsets, nsize = [], [], []
    for i in range(N):
        fused_nodes: set = set()
        best_off = None
        for s, prov in sessions[i].items():
            if s not in tp:
                continue
            frames = prov.get("frames") or [int(f) for (f, _b) in prov.get("crops", [])]
            for f in frames:
                f = int(f)
                if not (0 <= f < len(tp[s])):
                    continue
                pose = tp[s][f]
                fused_nodes.add(fused.nodes[int(np.linalg.norm(Pf - pose, axis=1).argmin())].id)
                off = float(np.linalg.norm(C[i] - pose))
                best_off = off if best_off is None else min(best_off, off)
        if not fused_nodes:                                             # no observations -> nearest to centre
            fused_nodes = {fused.nodes[int(np.linalg.norm(Pf - C[i], axis=1).argmin())].id}
            best_off = 0.0
        node_sets.append(sorted(fused_nodes))
        offsets.append(best_off if best_off is not None else 0.0)
        nsize.append(len(fused_nodes))

    # 4) persist  (object_attach.json now carries a node-SET per object)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fused.to_json(args.out_dir / "merged_route.json")
    (args.out_dir / "object_attach.json").write_text(json.dumps(
        {"uids": uids, "nodes": node_sets, "offset": offsets}))
    (args.out_dir / "sim3.json").write_text(json.dumps(sim3))
    print(f"attach: nodes/obj med {int(np.median(nsize))} max {max(nsize)} | "
          f"closest-approach offset med {np.median(offsets):.2f}m p90 {np.percentile(offsets,90):.2f}m", flush=True)
    print(f"persisted -> {args.out_dir}/merged_route.json, object_attach.json, sim3.json", flush=True)


if __name__ == "__main__":
    main()
