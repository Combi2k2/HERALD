"""Navigation graph data model: routing over waypoints and portals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

NavNodeKind = Literal["waypoint", "portal"]
NavEdgeSource = Literal["osm", "trajectory"]


@dataclass
class NavNode:
    id: str
    pos: tuple[float, float]
    kind: NavNodeKind = "waypoint"
    region_id: str | None = None
    z: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "pos": [self.pos[0], self.pos[1]],
            "kind": self.kind,
        }
        if self.region_id is not None:
            payload["region_id"] = self.region_id
        if self.z is not None:
            payload["z"] = self.z
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavNode:
        pos = data["pos"]
        z = data.get("z")
        return cls(
            id=data["id"],
            pos=(float(pos[0]), float(pos[1])),
            kind=data.get("kind", "waypoint"),
            region_id=data.get("region_id"),
            z=float(z) if z is not None else None,
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
    crs: str = "wgs84"
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
            "crs": self.crs,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavGraph:
        return cls(
            crs=data.get("crs", "wgs84"),
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
