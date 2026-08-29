"""Prototype `RouteGraph` types  (experiment route-graph-01, per design-route-graph.md).

Throwaway wrapper (experiments rule 4): this is the structure we intend to promote to
`herald/scene/common/route.py`. It deliberately avoids importing core (keeps the experiment
torch-free) — `refs` is a plain list of dicts here; the promoted version uses core `SourceRef`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

RouteNodeType = Literal["nav", "poi"]
RouteNodeSource = Literal["osm", "trajectory", "derived"]
RouteEdgeSource = Literal["osm", "trajectory"]


@dataclass
class RouteNode:
    id: str
    pos: tuple[float, float, float]                 # 3D ENU, site frame
    type: RouteNodeType = "nav"
    source: RouteNodeSource = "osm"
    refs: list[dict] = field(default_factory=list)  # provenance (core: SourceRef); e.g. {session, frames}
    radius: float | None = None
    poi_ref: str | None = None                      # set iff type == "poi"


@dataclass
class RouteEdge:
    source_id: str
    target_id: str
    length: float
    weight: float | None = None
    source: RouteEdgeSource = "osm"


class RouteGraph:
    def __init__(self, nodes: list[RouteNode] | None = None, edges: list[RouteEdge] | None = None):
        self.nodes: list[RouteNode] = nodes or []
        self.edges: list[RouteEdge] = edges or []
        self._reindex()

    # ---- indexing ----
    def _reindex(self) -> None:
        self._idx = {n.id: i for i, n in enumerate(self.nodes)}
        if len(self._idx) != len(self.nodes):
            raise ValueError("duplicate RouteNode id")

    def add_node(self, n: RouteNode) -> None:
        self._idx[n.id] = len(self.nodes)
        self.nodes.append(n)

    def add_edge(self, a: str, b: str, length: float, *, weight=None, source="osm") -> None:
        self.edges.append(RouteEdge(a, b, float(length), weight, source))

    def get_node(self, node_id: str) -> RouteNode | None:
        i = self._idx.get(node_id)
        return self.nodes[i] if i is not None else None

    # ---- geometry / routing ----
    def positions(self) -> np.ndarray:
        return np.array([n.pos for n in self.nodes], np.float32)

    def edge_pairs(self) -> list[tuple[int, int]]:
        return [(self._idx[e.source_id], self._idx[e.target_id]) for e in self.edges]

    def _csr(self):
        from scipy.sparse import csr_matrix
        M = len(self.nodes)
        ij = np.array(self.edge_pairs(), np.int64).reshape(-1, 2)
        w = np.array([e.weight if e.weight is not None else e.length for e in self.edges], np.float64)
        rows = np.concatenate([ij[:, 0], ij[:, 1]]) if len(ij) else np.zeros(0, np.int64)
        cols = np.concatenate([ij[:, 1], ij[:, 0]]) if len(ij) else np.zeros(0, np.int64)
        data = np.concatenate([w, w]) if len(w) else np.zeros(0)
        return csr_matrix((data, (rows, cols)), shape=(M, M))

    def n_components(self) -> int:
        from scipy.sparse.csgraph import connected_components
        return int(connected_components(self._csr(), directed=False, return_labels=False))

    def geodesic(self, src) -> np.ndarray:
        """Dijkstra distances from `src` (node id or list-index) to every node, in list order."""
        from scipy.sparse.csgraph import dijkstra
        i = self._idx[src] if isinstance(src, str) else int(src)
        return dijkstra(self._csr(), directed=False, indices=i)

    def shortest_path(self, a: str, b: str) -> float:
        return float(self.geodesic(a)[self._idx[b]])

    def nearest(self, pos) -> str:
        d = np.linalg.norm(self.positions() - np.asarray(pos, np.float32), axis=1)
        return self.nodes[int(d.argmin())].id

    def merge(self, other: "RouteGraph", *, radius: float) -> dict[str, str]:
        """Snap `other`'s nodes into self within `radius` (loop closure / multi-session fusion);
        union refs, remap and append edges. Simple O(N*M) match (fine at these sizes). Returns
        the `remap` {other_node_id -> surviving self_node_id} so callers can carry per-node
        attachments (e.g. an object's observing nav-nodes) through the snap."""
        P = self.positions()
        remap: dict[str, str] = {}
        for n in other.nodes:
            if len(P):
                d = np.linalg.norm(P - np.asarray(n.pos, np.float32), axis=1)
                j = int(d.argmin())
                if d[j] <= radius:
                    tgt = self.nodes[j]
                    tgt.refs.extend(n.refs)
                    remap[n.id] = tgt.id
                    continue
            self.add_node(n)
            remap[n.id] = n.id
            P = self.positions()
        for e in other.edges:
            a, b = remap[e.source_id], remap[e.target_id]
            if a != b:
                self.add_edge(a, b, e.length, weight=e.weight, source=e.source)
        return remap

    def absorb(self, other: "RouteGraph") -> None:
        """Append-only union: add all of `other`'s nodes and edges verbatim, NO fusion (ids stay
        stable, positions untouched). Requires ids disjoint from self (session-prefixed). This is
        the non-destructive counterpart to `merge`: connectivity between the two is added afterwards
        by `add_proximity_edges`, not by collapsing nodes."""
        for n in other.nodes:
            self.add_node(n)
        for e in other.edges:
            self.add_edge(e.source_id, e.target_id, e.length, weight=e.weight, source=e.source)

    def add_proximity_edges(self, radius: float, *, min_seq_gap: int = 3) -> int:
        """Connect nodes within `radius` that aren't already directly linked, with edge weight =
        Euclidean distance. This replaces node-snapping for both jobs: intra-session loop closure
        (a revisited place links its two passes) and cross-session fusion (parallel trajectories
        link up) -- without ever moving or merging a node. To avoid corner-cutting a single pass,
        a candidate pair from the *same* session whose sequence indices differ by < `min_seq_gap`
        is skipped (they're already chained along the path); revisits (large gap) and cross-session
        pairs (different session) always qualify. Returns the number of edges added."""
        from scipy.spatial import cKDTree
        P = self.positions()
        if len(P) < 2:
            return 0
        existing = {(min(self._idx[e.source_id], self._idx[e.target_id]),
                     max(self._idx[e.source_id], self._idx[e.target_id])) for e in self.edges}

        def _seg(i):
            r = self.nodes[i].refs[0] if self.nodes[i].refs else {}
            return r.get("session"), r.get("seq")

        added = 0
        for i, j in cKDTree(P).query_pairs(radius):
            if (i, j) in existing:
                continue
            si, gi = _seg(i)
            sj, gj = _seg(j)
            if si is not None and si == sj and gi is not None and abs(int(gi) - int(gj)) < min_seq_gap:
                continue                                       # same pass, near-neighbour -> skip
            self.add_edge(self.nodes[i].id, self.nodes[j].id,
                          float(np.linalg.norm(P[i] - P[j])), source="derived")
            added += 1
        return added

    # ---- io ----
    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [asdict(n) for n in self.nodes], "edges": [asdict(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> "RouteGraph":
        nodes = [RouteNode(**{**n, "pos": tuple(n["pos"])}) for n in d["nodes"]]
        edges = [RouteEdge(**e) for e in d["edges"]]
        return cls(nodes, edges)

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path) -> "RouteGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))
