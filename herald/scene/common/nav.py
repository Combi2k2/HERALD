"""Navigation graph data model: routing over waypoints and portals.

Mirrors the :mod:`scene.common.graph` node interface (``id``/``type``/``refs``)
but forms a graph rather than a tree, so there is no parent id. Node positions
are ENU metres in the graph's :class:`Frame`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SourceRef

NavNodeType = Literal["waypoint", "portal"]
NavEdgeSource = Literal["osm", "trajectory"]


@dataclass
class NavNode:
    id: str
    pos: tuple[float, float]
    type: NavNodeType = "waypoint"
    refs: list[SourceRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pos": [self.pos[0], self.pos[1]],
            "type": self.type,
            "refs": [ref.to_dict() for ref in self.refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavNode:
        pos = data["pos"]
        return cls(
            id=data["id"],
            pos=(float(pos[0]), float(pos[1])),
            type=data.get("type", "waypoint"),
            refs=[SourceRef.from_dict(r) for r in data.get("refs", [])],
        )


@dataclass
class NavEdge:
    source_id: str
    target_id: str
    length: float
    weight: float | None = None
    source: NavEdgeSource = "osm"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "length": self.length,
            "source": self.source,
        }
        if self.weight is not None:
            payload["weight"] = self.weight
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavEdge:
        weight = data.get("weight")
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            length=float(data["length"]),
            weight=float(weight) if weight is not None else None,
            source=data.get("source", "osm"),
        )


@dataclass
class NavGraph:
    frame: Frame
    nodes: list[NavNode] = field(default_factory=list)
    edges: list[NavEdge] = field(default_factory=list)

    def add_node(self, node: NavNode) -> None:
        self.nodes.append(node)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        length: float,
        *,
        weight: float | None = None,
        source: NavEdgeSource = "osm",
    ) -> None:
        self.edges.append(
            NavEdge(
                source_id=source_id,
                target_id=target_id,
                length=length,
                weight=weight,
                source=source,
            )
        )

    def get_node(self, node_id: str) -> NavNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavGraph:
        return cls(
            frame=Frame.from_dict(data["frame"]),
            nodes=[NavNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[NavEdge.from_dict(e) for e in data.get("edges", [])],
        )

    def to_json(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> NavGraph:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
