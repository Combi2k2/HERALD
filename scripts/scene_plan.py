#!/usr/bin/env python
"""scene_plan: NL query -> retrieved area shortlist -> LLM-ordered sequence -> nav-graph path.

Thin CLI over herald.scene.plan.pipeline.plan_route(), which already composes the three stages
end to end: descend() (mixed semantic+geodesic area retrieval), order_route() (one LLM call that
orders the shortlist into a visit sequence), and RouteGraph shortest-path (the actual walkable
route visiting each stop in order). This script only loads inputs, calls it, and prints/saves
the result -- no planning logic lives here (see pipeline.py's own docstring).

No query-decomposition front end: order_route() already reasons over multi-destination queries
itself from the one retrieved shortlist, so a separate LLM decomposition pass (as
scripts/query_decompose.py's older, superseded retrieval does) would be redundant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERALD_ROOT = Path(__file__).resolve().parents[1]
if str(_HERALD_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERALD_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", help="free-text navigation request, e.g. "
                   "'help me buy a cola and show me the way to buy ticket for train station'")
    p.add_argument("--graph", type=Path, required=True, help="area/object SceneGraph (.json) -- "
                    "ideally VLM-captioned (scene_area_classify.py's output): descend()'s "
                    "semantic score is only as good as the area text it scores against")
    p.add_argument("--route", type=Path, required=True, help="persistent route graph (.json)")
    p.add_argument("--robot-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--requester-pos", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                    help="defaults to --robot-pos if omitted (requester == robot start)")
    p.add_argument("--model", default="ollama:qwen2.5vl:3b")
    p.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--beta", type=float, default=None, help="default: herald.scene.plan.descent.DEFAULT_BETA (0.9, sweep-validated)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-required", type=int, default=0,
                    help="if given (>0), overrides --k with the adaptive k=2*n_required+1 rule "
                         "validated in experiments/planner-eval-01/result.md -- a fixed k trades "
                         "single-destination quality for multi-destination quality, this scales "
                         "k to the actual need per query")
    p.add_argument("--geodesic", action="store_true",
                    help="use the route-graph geodesic proximity term instead of straight-line "
                         "Euclidean (the mixed-geodesic condition validated as best)")
    p.add_argument("--iterative", action="store_true",
                    help="resolve one destination at a time (plan_route_iterative) instead of "
                         "one-shot (plan_route) -- see pipeline.py's docstring for the tradeoff")
    p.add_argument("--out", type=Path, default=None, help="optional: dump the plan as JSON")
    args = p.parse_args()

    from herald.scene.common.graph import SceneGraph
    from herald.scene.common.route import RouteGraph
    from herald.scene.plan.descent import DEFAULT_BETA
    from herald.scene.plan.pipeline import plan_route, plan_route_iterative
    from services.embeddings.encoder import TextEncoder

    graph = SceneGraph.from_json(args.graph)
    route = RouteGraph.from_json(args.route)
    encoder = TextEncoder(model_id=args.model_id)

    robot_pos = np.asarray(args.robot_pos, np.float64)
    requester_pos = np.asarray(args.requester_pos, np.float64) if args.requester_pos is not None else robot_pos
    beta = args.beta if args.beta is not None else DEFAULT_BETA
    k = 2 * args.n_required + 1 if args.n_required > 0 else args.k

    areas = [n for n in graph.nodes if n.level == "area" and n.geom is not None]
    print(f"graph: {len(areas)} areas. query: {args.query!r}. beta={beta} k={k} "
          f"geodesic={args.geodesic} iterative={args.iterative}", flush=True)

    if args.geodesic:
        targets = {a.uid: a.geom.offset for a in areas}
        distance_fn = route.geodesic_distance_fn(robot_pos, targets)
        distance_fn_factory = (lambda pos: route.geodesic_distance_fn(pos, targets)) if args.iterative else None
    else:
        distance_fn = None
        distance_fn_factory = None

    common = dict(graph=graph, route=route, encoder=encoder, robot_pos=robot_pos,
                  requester_pos=requester_pos, model=args.model, beta=beta, k=k)
    if args.iterative:
        result = plan_route_iterative(args.query, **common, distance_fn_factory=distance_fn_factory)
    else:
        result = plan_route(args.query, **common, distance_fn=distance_fn)

    by_uid = {n.uid: n for n in graph.nodes}
    print(f"\nshortlist ({len(result.shortlist)}):")
    for n in result.shortlist:
        print(f"  area {n.uid:3d}  {n.name}")
    print(f"\nordered sequence: {result.sequence}")
    for entry in result.sequence:
        if entry == "REQUESTER":
            print("  -> back to requester")
        else:
            node = by_uid.get(entry)
            print(f"  -> area {entry}  {node.name if node else '?'}")
    print(f"\npath: {len(result.path_nodes)} nav nodes, {result.path_length_m:.1f}m total", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "query": args.query,
            "shortlist": [{"uid": n.uid, "name": n.name} for n in result.shortlist],
            "sequence": result.sequence,
            "stops": result.stops,
            "path_nodes": result.path_nodes,
            "path_length_m": result.path_length_m,
        }, indent=2))
        print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
