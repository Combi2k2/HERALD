"""Scene graph data model for offline initialization.

The scene graph stores topology, captions, and embedding references for the global
planner. The site :class:`~herald.scene.common.repr.SceneRepr` bundles the shared
:class:`Frame` with the semantic graph and navigation graph. Model ids on the graph
header record which embedding and VLM models produced the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from herald.scene.common.geometry import Geometry
from herald.scene.common.source import SourceRef

NodeLevel = Literal[
    "site", "zone", "region", "structure", "floor", "space",
    "area", "object", "portal", "ego",
]
EdgeType = Literal["contains", "spatial"]


@dataclass
class SceneNode:
    uid: int
    level: NodeLevel
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    geom: Geometry | None = None
    votes: dict[str, int] = field(default_factory=dict)
    name: str = ""
    desc: str = ""
    supp: int = 0
    conf: float = 0.0
    refs: list[SourceRef] = field(default_factory=list)
    txt_embedding_ref: str | None = None
    img_embedding_ref: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "level": self.level,
            "parent": self.parent,
            "children": list(self.children),
            "geom": self.geom.to_dict() if self.geom is not None else None,
            "votes": dict(self.votes),
            "name": self.name,
            "desc": self.desc,
            "supp": self.supp,
            "conf": self.conf,
            "refs": [r.to_dict() for r in self.refs],
            "txt_embedding_ref": self.txt_embedding_ref,
            "img_embedding_ref": self.img_embedding_ref,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneNode:
        geom = data.get("geom")
        return cls(
            uid=int(data["uid"]),
            level=data["level"],
            parent=data.get("parent"),
            children=[int(c) for c in data.get("children", [])],
            geom=Geometry.from_dict(geom) if geom is not None else None,
            votes={str(k): int(v) for k, v in dict(data.get("votes") or {}).items()},
            name=str(data.get("name") or ""),
            desc=str(data.get("desc") or ""),
            supp=int(data.get("supp", 0)),
            conf=float(data.get("conf", 0.0)),
            refs=[SourceRef.from_dict(r) for r in data.get("refs", [])],
            txt_embedding_ref=data.get("txt_embedding_ref"),
            img_embedding_ref=data.get("img_embedding_ref"),
            attrs=dict(data.get("attrs") or {}),
        )


@dataclass
class SceneEdge:
    source: int
    target: int
    edge_type: EdgeType = "contains"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "edge_type": self.edge_type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneEdge:
        return cls(
            source=int(data["source"]),
            target=int(data["target"]),
            edge_type=data.get("edge_type", "contains"),
        )


@dataclass
class SceneGraph:
    emb_model_id: str
    vlm_model_id: str = ""
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    _by_uid: dict[int, SceneNode] = field(default_factory=dict, repr=False)

    def add_node(self, node: SceneNode) -> None:
        if node.uid in self._by_uid:
            raise ValueError(f"duplicate node uid: {node.uid!r}")
        self.nodes.append(node)
        self._by_uid[node.uid] = node

    def add_edge(self, source: int, target: int, *, edge_type: EdgeType = "contains") -> None:
        self.edges.append(SceneEdge(source=source, target=target, edge_type=edge_type))

    def get_node(self, uid: int) -> SceneNode | None:
        return self._by_uid.get(uid)

    def site_node(self) -> SceneNode | None:
        for node in self.nodes:
            if node.level == "site":
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "emb_model_id": self.emb_model_id,
            "vlm_model_id": self.vlm_model_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneGraph:
        graph = cls(
            emb_model_id=str(data.get("emb_model_id", "")),
            vlm_model_id=str(data.get("vlm_model_id", "")),
        )
        graph.nodes = [SceneNode.from_dict(n) for n in data.get("nodes", [])]
        graph.edges = [SceneEdge.from_dict(e) for e in data.get("edges", [])]

        for node in graph.nodes:
            graph._by_uid[node.uid] = node
        return graph

    def to_json(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> SceneGraph:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
