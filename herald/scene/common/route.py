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

    def absorb(self, other: RouteGraph, *, radius: float) -> None:
        """Fold another route graph in: append its nodes and edges, then link its nodes to the
        pre-existing ones with proximity edges (any pair within `radius`)."""
        from scipy.spatial import cKDTree
        base = self.nodes[:]
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
        if base and other.nodes:
            tree = cKDTree(np.array([n.pos for n in base], np.float32))
            for n in other.nodes:
                for i in tree.query_ball_point(np.asarray(n.pos, np.float32), radius):
                    self.edges.append(RouteEdge(base[i].id, n.id, "derived"))

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
