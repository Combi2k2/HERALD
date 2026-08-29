"""Route graph: a persistent, navigable node+edge lattice over the scene. Str-keyed nodes
(distinct from the int-uid SceneNode); a `poi` node references a SceneNode via `poi_uid`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

    def shortest_path(self, a: str, b: str) -> float:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        idx = {n.id: i for i, n in enumerate(self.nodes)}
        pos = np.array([n.pos for n in self.nodes], np.float32)
        ij = np.array([(idx[e.source_id], idx[e.target_id]) for e in self.edges], np.int64).reshape(-1, 2)
        w = np.linalg.norm(pos[ij[:, 0]] - pos[ij[:, 1]], axis=1) if len(ij) else np.zeros(0)
        rows = np.concatenate([ij[:, 0], ij[:, 1]]) if len(ij) else np.zeros(0, np.int64)
        cols = np.concatenate([ij[:, 1], ij[:, 0]]) if len(ij) else np.zeros(0, np.int64)
        csr = csr_matrix((np.concatenate([w, w]) if len(w) else np.zeros(0), (rows, cols)),
                         shape=(len(self.nodes), len(self.nodes)))
        return float(dijkstra(csr, directed=False, indices=idx[a])[idx[b]])

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
