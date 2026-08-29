"""Route-graph construction from a camera trajectory  (experiment route-graph-01).

Pipeline (a "simple feature"): a polyline of camera positions ->
  1. Taubin smooth   (shrink-free denoise; pure Laplacian would round corners into walls),
  2. arc-length resample at fixed spacing  (uniform edge lengths),
  3. voxel-snap       (merge nodes within r_snap -> loop closure: revisited/nearby passes fuse
                       into one node; this turns a temporally-ordered path into a spatial
                       connectivity graph, and it is wall-safe because only actually-traversed
                       places are ever connected),
  4. edges = sequential-along-path + snap-merges  (no unchecked kNN edges that could jump a wall).

Everything is numpy + scipy.sparse (no torch).
"""
from __future__ import annotations

import numpy as np


def taubin_smooth(P: np.ndarray, *, lam: float = 0.5, mu: float = -0.53, iters: int = 10) -> np.ndarray:
    """Shrink-free polyline smoothing; endpoints fixed."""
    Q = P.astype(np.float64).copy()
    for _ in range(iters):
        for factor in (lam, mu):
            lap = np.zeros_like(Q)
            lap[1:-1] = 0.5 * (Q[:-2] + Q[2:]) - Q[1:-1]
            Q[1:-1] += factor * lap[1:-1]
    return Q.astype(np.float32)


def resample_arclength(P: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline to points ~`spacing` apart along its arc length."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < spacing:
        return P[[0, -1]] if len(P) > 1 else P
    targets = np.arange(0.0, total, spacing)
    out = np.empty((len(targets), 3), np.float32)
    for k in range(3):
        out[:, k] = np.interp(targets, s, P[:, k])
    return out


def voxel_snap(P: np.ndarray, r_snap: float):
    """Merge points sharing a voxel of size r_snap. Returns node positions (M,3) and the
    per-input node id (len N)."""
    keys = np.floor(P / r_snap).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    M = len(uniq)
    nodes = np.zeros((M, 3), np.float64)
    cnt = np.zeros(M, np.int64)
    np.add.at(nodes, inv, P)
    np.add.at(cnt, inv, 1)
    nodes /= cnt[:, None]
    return nodes.astype(np.float32), inv


def build_open(positions: np.ndarray, *, spacing: float, smooth_iters: int = 10,
               source: str = "trajectory", session: str | None = None, id_prefix: str = "t"):
    """Construct an OPEN (non-fused) route graph from one camera trajectory: smooth -> arc-length
    resample -> one node per resampled point (NO voxel-snap) -> sequential edges along the path.

    Unlike `build`, nothing is merged: every node keeps its measured position and a stable id, so
    unioning sessions is append-only and loop closure / cross-session links are added later as
    *proximity edges* (`RouteGraph.add_proximity_edges`) rather than by fusing nodes. Each node
    records its per-session sequence index in `refs` so proximity linking can skip near-neighbours
    on the same pass (corner-cutting) while still allowing revisit loop closure."""
    from route_types import RouteGraph, RouteNode
    P = taubin_smooth(positions, iters=smooth_iters) if smooth_iters > 0 else positions
    R = resample_arclength(P, spacing)
    rn = [RouteNode(id=f"{id_prefix}{k}", pos=(float(p[0]), float(p[1]), float(p[2])),
                    type="nav", source=source, refs=[{"session": session, "seq": k}])
          for k, p in enumerate(R)]
    g = RouteGraph(rn, [])
    for k in range(len(R) - 1):
        g.add_edge(f"{id_prefix}{k}", f"{id_prefix}{k + 1}",
                   float(np.linalg.norm(R[k + 1] - R[k])), source=source)
    return g


def build(positions: np.ndarray, *, spacing: float, r_snap: float, smooth_iters: int = 10,
          source: str = "trajectory", session: str | None = None, id_prefix: str = "t"):
    """Construct a `route_types.RouteGraph` from a camera trajectory.

    Returns (graph, node_of_sample) where node_of_sample maps each resampled point to its snapped
    node index (for the loop-closure statistic).
    """
    from route_types import RouteGraph, RouteNode
    P = taubin_smooth(positions, iters=smooth_iters) if smooth_iters > 0 else positions
    R = resample_arclength(P, spacing)
    nodes, node_of = voxel_snap(R, r_snap)
    rn = [RouteNode(id=f"{id_prefix}{k}",
                    pos=(float(p[0]), float(p[1]), float(p[2])),
                    type="nav", source=source, radius=float(r_snap),
                    refs=[{"session": session}] if session else [])
          for k, p in enumerate(nodes)]
    g = RouteGraph(rn, [])
    seen: set = set()
    for a, b in zip(node_of[:-1], node_of[1:]):
        a, b = int(a), int(b)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        g.add_edge(f"{id_prefix}{a}", f"{id_prefix}{b}",
                   float(np.linalg.norm(nodes[a] - nodes[b])), source=source)
    return g, node_of
