"""Route graph: a persistent, navigable node+edge lattice over the scene. Str-keyed nodes
(distinct from the int-uid SceneNode); a `poi` node references a SceneNode via `poi_uid`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from herald.scene.common.geometry import Vec3
from herald.scene.common.source import SourceRef

RouteNodeType = Literal["nav", "poi"]
RouteEdgeSource = Literal["osm", "trajectory", "derived"]


@dataclass
class RouteNode:
    id: str
    pos: Vec3
    type: RouteNodeType = "nav"
    refs: list[SourceRef] = field(default_factory=list)
    poi_uid: int | None = None

    def __post_init__(self) -> None:
        self.pos = Vec3(self.pos)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "pos": self.pos.tolist(), "type": self.type,
                "refs": [r.to_dict() for r in self.refs], "poi_uid": self.poi_uid}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RouteNode:
        return cls(d["id"], Vec3(d["pos"]), d.get("type", "nav"),
                   [SourceRef.from_dict(r) for r in d.get("refs", [])], d.get("poi_uid"))


@dataclass
class RouteEdge:
    source_id: str
    target_id: str
    source: RouteEdgeSource = "osm"

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "target_id": self.target_id, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RouteEdge:
        return cls(d["source_id"], d["target_id"], d.get("source", "osm"))


class RouteGraph:
    def __init__(self, nodes: list[RouteNode] | None = None, edges: list[RouteEdge] | None = None):
        self.nodes: list[RouteNode] = nodes or []
        self.edges: list[RouteEdge] = edges or []

    def absorb(self, other: RouteGraph) -> None:
        """Append another route graph's nodes and edges (no fusion; ids stay stable). Call
        add_proximity_edges afterwards to link the newly-absorbed nodes to the rest."""
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)

    def add_proximity_edges(self, radius: float, *, min_seq_gap: int = 3) -> int:
        """Link any node pair within `radius` that isn't already edged: intra-session loop closure
        (a revisited place joins its two passes) and cross-session fusion, without moving/merging a
        node. A same-session pair whose sequence indices differ by < min_seq_gap is skipped (already
        chained along the path). Returns the number of edges added."""
        from scipy.spatial import cKDTree
        if len(self.nodes) < 2:
            return 0
        P = np.array([n.pos for n in self.nodes], np.float32)
        idx = {n.id: i for i, n in enumerate(self.nodes)}
        existing = {(min(a, b), max(a, b)) for a, b in
                    ((idx[e.source_id], idx[e.target_id]) for e in self.edges)}

        def seg(i: int):
            r = self.nodes[i].refs[0] if self.nodes[i].refs else None
            return (r.assigned_id, r.metadata.get("seq")) if r else (None, None)

        added = 0
        for i, j in cKDTree(P).query_pairs(radius):
            if (i, j) in existing:
                continue
            si, gi = seg(i)
            sj, gj = seg(j)
            if si is not None and si == sj and gi is not None and gj is not None and abs(int(gi) - int(gj)) < min_seq_gap:
                continue
            self.edges.append(RouteEdge(self.nodes[i].id, self.nodes[j].id, "derived"))
            added += 1
        return added

    def _csr(self):
        from scipy.sparse import csr_matrix
        idx = {n.id: i for i, n in enumerate(self.nodes)}
        pos = np.array([n.pos for n in self.nodes], np.float32)
        ij = np.array([(idx[e.source_id], idx[e.target_id]) for e in self.edges], np.int64).reshape(-1, 2)
        w = np.linalg.norm(pos[ij[:, 0]] - pos[ij[:, 1]], axis=1) if len(ij) else np.zeros(0)
        rows = np.concatenate([ij[:, 0], ij[:, 1]]) if len(ij) else np.zeros(0, np.int64)
        cols = np.concatenate([ij[:, 1], ij[:, 0]]) if len(ij) else np.zeros(0, np.int64)
        csr = csr_matrix((np.concatenate([w, w]) if len(w) else np.zeros(0), (rows, cols)),
                         shape=(len(self.nodes), len(self.nodes)))
        return csr, idx

    def shortest_path(self, a: str, b: str) -> float:
        from scipy.sparse.csgraph import dijkstra
        csr, idx = self._csr()
        return float(dijkstra(csr, directed=False, indices=idx[a])[idx[b]])

    def shortest_path_nodes(self, a: str, b: str) -> list[str]:
        """Node-id path a->b (inclusive), not just its length. Empty list if unreachable."""
        from scipy.sparse.csgraph import dijkstra
        csr, idx = self._csr()
        ai, bi = idx[a], idx[b]
        dist, pred = dijkstra(csr, directed=False, indices=ai, return_predecessors=True)
        if not np.isfinite(dist[bi]):
            return []
        path = [bi]
        while path[-1] != ai:
            path.append(int(pred[path[-1]]))
        ids = [self.nodes[i].id for i in reversed(path)]
        return ids

    def nearest_node(self, pos, *, type: RouteNodeType | None = "nav") -> RouteNode | None:
        """The node closest (Euclidean) to `pos`, optionally restricted to one RouteNodeType."""
        candidates = [n for n in self.nodes if type is None or n.type == type]
        if not candidates:
            return None
        p = np.asarray(pos, np.float32)
        d = [float(np.linalg.norm(np.asarray(n.pos, np.float32) - p)) for n in candidates]
        return candidates[int(np.argmin(d))]

    def geodesic_distance(self, pos_a, pos_b) -> float:
        """Free-space distance proxy between two arbitrary points: project each onto its
        nearest nav node, run Dijkstra between them, and floor the result at the straight-line
        Euclidean separation -- a node-graph geodesic under-reports (even 0) when two points
        share a nearest node, same fix already validated in
        experiments/area-cluster-02/cluster_areas.py's object-clustering distance."""
        eu = float(np.linalg.norm(np.asarray(pos_a, np.float64) - np.asarray(pos_b, np.float64)))
        na, nb = self.nearest_node(pos_a), self.nearest_node(pos_b)
        if na is None or nb is None:
            return eu
        geo = self.shortest_path(na.id, nb.id)
        return max(geo, eu) if np.isfinite(geo) else eu

    def geodesic_distances_from(self, pos, targets: dict) -> dict:
        """Same proxy as geodesic_distance, but one source -> many targets: runs Dijkstra ONCE
        from `pos`'s nearest nav node instead of once per target. `targets`: {key: pos}. Returns
        {key: distance}."""
        from scipy.sparse.csgraph import dijkstra
        if not targets:
            return {}
        src = self.nearest_node(pos)
        if src is None:
            return {k: float(np.linalg.norm(np.asarray(pos, np.float64) - np.asarray(p, np.float64)))
                    for k, p in targets.items()}
        csr, idx = self._csr()
        dist = dijkstra(csr, directed=False, indices=idx[src.id])
        out = {}
        for k, p in targets.items():
            eu = float(np.linalg.norm(np.asarray(pos, np.float64) - np.asarray(p, np.float64)))
            nt = self.nearest_node(p)
            geo = float(dist[idx[nt.id]]) if nt is not None else float("inf")
            out[k] = max(geo, eu) if np.isfinite(geo) else eu
        return out

    def geodesic_distance_fn(self, from_pos, targets: dict) -> Callable[[object], float]:
        """Convenience wrapper around geodesic_distances_from: builds a
        herald.scene.plan.descent.descend()-compatible `distance_fn(pos) -> meters` closure.
        Keyed by a rounded-position lookup rather than dict identity/uid, since callers may pass
        a target's position by value rather than the exact same object (e.g.
        plan_route_iterative() re-derives it fresh each step)."""
        dists = self.geodesic_distances_from(from_pos, targets)

        def _key(pos) -> tuple:
            return tuple(np.round(np.asarray(pos, np.float64), 6))

        by_key = {_key(targets[k]): d for k, d in dists.items()}

        def distance_fn(pos) -> float:
            return by_key[_key(pos)]

        return distance_fn

    def add_poi(self, id: str, scene_uid: int, pos) -> RouteNode:
        """Register a POI route-node for a SceneNode (e.g. an area) at `pos`, wired into the nav
        graph via an edge to its nearest existing nav-node -- a POI with no edge would be
        unreachable by shortest_path_nodes. `id` must be unique among this graph's node ids."""
        node = RouteNode(id=id, pos=Vec3(pos), type="poi", poi_uid=scene_uid)
        self.nodes.append(node)
        nearest = self.nearest_node(pos, type="nav")
        if nearest is not None:
            self.edges.append(RouteEdge(node.id, nearest.id, "derived"))
        return node

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes], "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> RouteGraph:
        return cls([RouteNode.from_dict(n) for n in d["nodes"]],
                   [RouteEdge.from_dict(e) for e in d["edges"]])

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path) -> RouteGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))
