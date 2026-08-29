"""Build a core RouteGraph from one camera trajectory: Taubin smooth -> arc-length resample ->
one node per sample + sequential edges. Cross-session/loop connectivity is added later by
RouteGraph.absorb. Each node records its session + sequence index in a SourceRef."""

from __future__ import annotations

import numpy as np

from herald.scene.common.route import RouteEdge, RouteGraph, RouteNode
from herald.scene.common.source import SourceRef


def _taubin(P: np.ndarray, *, lam: float = 0.5, mu: float = -0.53, iters: int = 10) -> np.ndarray:
    Q = P.astype(np.float64).copy()
    for _ in range(iters):
        for f in (lam, mu):
            lap = np.zeros_like(Q)
            lap[1:-1] = 0.5 * (Q[:-2] + Q[2:]) - Q[1:-1]
            Q[1:-1] += f * lap[1:-1]
    return Q


def _resample(P: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < spacing:
        return P[[0, -1]] if len(P) > 1 else P
    t = np.arange(0.0, total, spacing)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def build_route(positions, *, session: str, spacing: float = 1.0, smooth_iters: int = 10) -> RouteGraph:
    prefix = f"{session}_"
    P = np.asarray(positions, np.float64)
    P = _taubin(P, iters=smooth_iters) if smooth_iters > 0 else P
    R = _resample(P, spacing)
    nodes = [RouteNode(id=f"{prefix}{k}", pos=p, type="nav",
                       refs=[SourceRef("trajectory", str(session), metadata={"seq": k})])
             for k, p in enumerate(R)]
    edges = [RouteEdge(f"{prefix}{k}", f"{prefix}{k + 1}", "trajectory") for k in range(len(R) - 1)]
    return RouteGraph(nodes, edges)
