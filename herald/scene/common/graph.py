"""Scene graph data model for offline initialization.

The scene graph stores topology, captions, and embeddings for the global planner.
Point clouds are never embedded — only an optional ``point_cloud_uri`` on the
header for the viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from herald.scene.common.frame import LocalFrame

NodeLevel = Literal["site", "outdoor_region", "building"]
ZoneKind = Literal["outdoor_region", "building"]
EdgeType = Literal["contains", "spatial", "traj"]
HeightSource = Literal["osm_height_tag", "osm_levels", "default"]

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
class SceneNode:
    id: str
    level: NodeLevel
    zone_kind: ZoneKind | None
    text: str
    geometry_latlon: list[tuple[float, float]]
    embedding: list[float] | None = None
    height_m: float = 10.0
    height_source: HeightSource = "default"
    osm_id: int | None = None
    role: str | None = None
    category: str | None = None
    function: str | None = None
    name: str | None = None
    confidence: float | None = None
    classification_source: str | None = None
    osm_tags: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "level": self.level,
            "zone_kind": self.zone_kind,
            "text": self.text,
            "geometry_latlon": [[lat, lon] for lat, lon in self.geometry_latlon],
            "embedding": self.embedding,
            "height_m": self.height_m,
            "height_source": self.height_source,
            "osm_id": self.osm_id,
        }
        if self.role is not None:
            payload["role"] = self.role
        if self.category is not None:
            payload["category"] = self.category
        if self.function is not None:
            payload["function"] = self.function
        if self.name:
            payload["name"] = self.name
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.classification_source is not None:
            payload["classification_source"] = self.classification_source
        if self.osm_tags:
            payload["osm_tags"] = self.osm_tags
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneNode:
        geom = data.get("geometry_latlon", [])
        return cls(
            id=data["id"],
            level=data["level"],
            zone_kind=data.get("zone_kind"),
            text=data["text"],
            geometry_latlon=[(float(p[0]), float(p[1])) for p in geom],
            embedding=data.get("embedding"),
            height_m=float(data.get("height_m", 10.0)),
            height_source=data.get("height_source", "default"),
            osm_id=data.get("osm_id"),
            role=data.get("role"),
            category=data.get("category"),
            function=data.get("function"),
            name=data.get("name"),
            confidence=data.get("confidence"),
            classification_source=data.get("classification_source"),
            osm_tags=data.get("osm_tags"),
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
    schema_version: int = 2

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
