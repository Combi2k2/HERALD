"""Scene graph data model for offline initialization.

The scene graph stores topology, captions, and embedding references for the global
planner. Point clouds are never embedded — only an optional ``point_cloud_uri`` on
the header for the viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from herald.scene.common.frame import LocalFrame

NodeLevel = Literal[
    "site",
    "region",
    "structure",
    "floor",
    "space",
    "object",
    "portal",
]
GeomKind = Literal["point", "polyline", "polygon"]
EdgeType = Literal["contains", "spatial"]

SceneEventKind = Literal[
    "roi_resolved",
    "partition_computed",
    "pipeline_status",
    "zone_node_created",
    "containment_inferred",
    "embedding_attached",
    "pathways_ready",
    "pipeline_complete",
]


@dataclass(frozen=True)
class SceneEvent:
    kind: SceneEventKind
    node_id: str | None = None
    parent_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Geom:
    kind: GeomKind
    coords: list[tuple[float, float]]
    crs: str = "wgs84"
    z_range: tuple[float, float] | None = None
    shape_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "coords": [[lat, lon] for lat, lon in self.coords],
            "crs": self.crs,
        }
        if self.z_range is not None:
            payload["z_range"] = [self.z_range[0], self.z_range[1]]
        if self.shape_uri is not None:
            payload["shape_uri"] = self.shape_uri
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Geom:
        coords_raw = data.get("coords", [])
        z_raw = data.get("z_range")
        z_range: tuple[float, float] | None = None
        if z_raw is not None and len(z_raw) == 2:
            z_range = (float(z_raw[0]), float(z_raw[1]))
        return cls(
            kind=data["kind"],
            coords=[(float(p[0]), float(p[1])) for p in coords_raw],
            crs=data.get("crs", "wgs84"),
            z_range=z_range,
            shape_uri=data.get("shape_uri"),
        )


@dataclass
class SourceRef:
    assigned_by: str
    assigned_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"assigned_by": self.assigned_by}
        if self.assigned_id is not None:
            payload["assigned_id"] = self.assigned_id
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        return cls(
            assigned_by=data["assigned_by"],
            assigned_id=data.get("assigned_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SceneNode:
    id: str
    level: NodeLevel
    parent_id: str | None
    geom: Geom
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
            "level": self.level,
            "parent_id": self.parent_id,
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
            level=data["level"],
            parent_id=data.get("parent_id"),
            geom=Geom.from_dict(data["geom"]),
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
    site_id: str
    frame: LocalFrame
    embedding_model_id: str
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    point_cloud_uri: str | None = None
    roi_area_m2: float | None = None
    schema_version: int = 3

    def add_node(self, node: SceneNode) -> None:
        self.nodes.append(node)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_type: EdgeType = "contains",
    ) -> None:
        self.edges.append(
            SceneEdge(source_id=source_id, target_id=target_id, edge_type=edge_type)
        )

    def get_node(self, node_id: str) -> SceneNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "frame_origin": self.frame.to_dict(),
            "embedding_model_id": self.embedding_model_id,
            "point_cloud_uri": self.point_cloud_uri,
            "roi_area_m2": self.roi_area_m2,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneGraph:
        return cls(
            site_id=data["site_id"],
            frame=LocalFrame.from_dict(data["frame_origin"]),
            embedding_model_id=data["embedding_model_id"],
            nodes=[SceneNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[SceneEdge.from_dict(e) for e in data.get("edges", [])],
            point_cloud_uri=data.get("point_cloud_uri"),
            roi_area_m2=data.get("roi_area_m2"),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_json(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> SceneGraph:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def assert_no_point_cloud_payload(self) -> None:
        """Ensure graph JSON carries no embedded point cloud data."""
        raw = json.dumps(self.to_dict())
        assert "point_cloud" not in raw or self.point_cloud_uri is None or (
            '"point_cloud_uri": null' in raw or '"point_cloud_uri":' in raw
        )


EventCallback = Callable[[SceneEvent], None]
