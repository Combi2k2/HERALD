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

NodeType = Literal[
    "site",
    "zone",
    "region",
    "structure",
    "floor",
    "space",
    "object",
    "portal",
]
EdgeType = Literal["contains", "spatial"]


@dataclass
class SourceRef:
    assigned_by: str
    assigned_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assigned_by": self.assigned_by,
            "assigned_id": self.assigned_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        return cls(
            assigned_by=data["assigned_by"],
            assigned_id=data["assigned_id"],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SceneNode:
    id: str
    pid: str | None
    type: NodeType
    geom: Geometry
    refs: list[SourceRef] = field(default_factory=list)
    observation_count: int = 0
    name: str = ""
    desc: str = ""
    role: str = ""
    category: str = ""
    function: str = ""
    txt_embedding_ref: str | None = None
    viz_embedding_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "pid": self.pid,
            "type": self.type,
            "geom": self.geom.to_dict(),
            "refs": [ref.to_dict() for ref in self.refs],
            "observation_count": self.observation_count,
            "name": self.name,
            "desc": self.desc,
            "role": self.role,
            "category": self.category,
            "function": self.function,
        }
        if self.txt_embedding_ref is not None:  payload["txt_embedding_ref"] = self.txt_embedding_ref
        if self.viz_embedding_ref is not None:  payload["viz_embedding_ref"] = self.viz_embedding_ref
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneNode:
        return cls(
            id=data["id"],
            type=data["type"],
            pid=data.get("pid"),
            geom=Geometry.from_dict(data["geom"]),
            refs=[SourceRef.from_dict(r) for r in data.get("refs", [])],
            observation_count=int(data.get("observation_count", 0)),
            name=str(data.get("name") or ""),
            desc=str(data.get("desc") or ""),
            role=str(data.get("role") or ""),
            category=str(data.get("category") or ""),
            function=str(data.get("function") or ""),
            txt_embedding_ref=data.get("txt_embedding_ref"),
            viz_embedding_ref=data.get("viz_embedding_ref"),
        )


@dataclass
class SceneEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType = "contains"

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SceneEdge:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data.get("edge_type", "contains"),
        )


@dataclass
class SceneGraph:
    emb_model_id: str
    vlm_model_id: str = ""
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    _by_id: dict[str, SceneNode] = field(default_factory=dict, repr=False)

    def add_node(self, node: SceneNode) -> None:
        if node.id in self._by_id:
            raise ValueError(f"duplicate node id: {node.id!r}")
        self.nodes.append(node)
        self._by_id[node.id] = node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_type: EdgeType = "contains",
    ) -> None:
        self.edges.append(SceneEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type
        ))

    def get_node(self, node_id: str) -> SceneNode | None:
        return self._by_id.get(node_id)

    def site_node(self) -> SceneNode | None:
        for node in self.nodes:
            if node.type == "site":
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
            graph._by_id[node.id] = node
        
        return graph

    def to_json(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> SceneGraph:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
