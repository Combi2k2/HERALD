"""Planner stage 1: hierarchical top-K descent scoring candidate areas for a query.

s(v) = beta * cos(e_q, e_v) + (1 - beta) * rho(v)
rho(v) = exp(-(d(robot_pos, v.geom.offset) / sigma)^2)      # Gaussian proximity kernel

Writing C_l for the set retained at level l: starts with C_0 = the children of the graph's
root, expands the children of ALL of C_l (pooled, not per-node), scores that pooled set, and
keeps the K highest-scoring as C_{l+1}. Stops once C_l holds only level=="area" nodes (or runs
out of children to expand, whichever comes first).

Root: the current area-clustering pipeline (scripts/scene_area_cluster.py) never emits a
level=="site" node -- only "area" and "object" are populated today (see
experiments/hovsg-baseline-01/result.md's Finding on hierarchy depth). Falls back to every
top-level node (parent is None) as C_0 when no explicit site node exists, so on today's graphs
the descent runs its one and only level directly against every area -- correct, but not
demonstrating the multi-level pruning the formula is written for until intermediate levels
(floor/region) exist.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from herald.scene.common.graph import SceneGraph, SceneNode

DEFAULT_BETA = 0.9  # sweep-validated best on JapaneseCity/Hospital mixed-geodesic scoring,
                    # see experiments/planner-eval-01/result.md (was 0.7, never tuned before)
DEFAULT_SIGMA = 10.0  # meters -- typical room-to-room scale, not a tuned value


def rho(node_pos, robot_pos, sigma: float = DEFAULT_SIGMA,
        distance_fn: Callable[[object], float] | None = None) -> float:
    """distance_fn, if given, replaces the default straight-line distance-to-robot_pos with
    whatever proximity notion the caller wants (e.g. a route-graph geodesic distance proxy --
    see herald.scene.common.route.RouteGraph.geodesic_distance*) -- signature `distance_fn(pos)
    -> meters from the robot`. Keeps this module decoupled from RouteGraph: callers that want a
    geodesic term build the closure themselves (see experiments/planner-eval-01/eval_planner.py)."""
    if distance_fn is not None:
        d = float(distance_fn(node_pos))
    else:
        d = float(np.linalg.norm(np.asarray(node_pos, np.float64) - np.asarray(robot_pos, np.float64)))
    return float(np.exp(-(d / sigma) ** 2))


def descend(graph: SceneGraph, query_emb: np.ndarray, node_emb: dict[int, np.ndarray], robot_pos,
            *, beta: float = DEFAULT_BETA, sigma: float = DEFAULT_SIGMA, k: int = 5,
            distance_fn: Callable[[object], float] | None = None) -> list[SceneNode]:
    """query_emb: (D,) L2-normalized. node_emb: uid -> (D,) L2-normalized embedding, one entry
    per node the descent might need to score (area nodes at minimum; also any intermediate-level
    nodes if the graph has them). distance_fn: see rho() above -- None keeps today's exact
    Euclidean-to-robot_pos behavior. Returns the final retained area nodes (<=k of them, fewer
    if a level along the way had fewer than k scoreable children)."""
    by_uid = {n.uid: n for n in graph.nodes}
    site_nodes = [n for n in graph.nodes if n.level == "site"]
    if site_nodes:
        children = [by_uid[c] for s in site_nodes for c in s.children if c in by_uid]
    else:
        # No explicit site node in this graph (today's area-clustering pipeline never emits
        # one) -- every top-level (parentless) node stands in as a root child, so the descent
        # still runs its one real scoring+pruning pass instead of skipping straight to
        # "already terminal" just because these nodes happen to be level=="area" themselves.
        children = [n for n in graph.nodes if n.parent is None]

    while children:
        scored = []
        for c in children:
            if c.uid not in node_emb:
                continue
            sim = float(np.dot(query_emb, node_emb[c.uid]))
            r = rho(c.geom.offset, robot_pos, sigma, distance_fn) if c.geom is not None else 0.0
            scored.append((beta * sim + (1 - beta) * r, c))
        if not scored:
            return []
        scored.sort(key=lambda t: -t[0])
        retained = [c for _, c in scored[:k]]
        if all(n.level == "area" for n in retained):
            return retained
        child_uids = {c for n in retained for c in n.children}
        children = [by_uid[c] for c in child_uids if c in by_uid]

    return []
