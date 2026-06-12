"""Legacy v1/v2 dict migration and OSM-coupled construction helpers.

Kept out of the generic schema module (``graph.py``). Revise and reuse when
wiring v2 run loading or OSM-specific provenance back in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from herald.scene.common.geometry import Geometry, Vec3
from herald.scene.common.graph import NodeType, SceneNode, SourceRef

_V2_LEVEL_MAP: dict[str, NodeType] = {
    "outdoor_region": "region",
    "building": "structure",
}


def migrate_node_type(node_type: str) -> NodeType:
    return _V2_LEVEL_MAP.get(node_type, node_type)  # type: ignore[return-value]


def geom_from_v2(geometry_latlon: list[Any], *, height: float) -> Geometry:
    rows = [
        [float(p[0]), float(p[1]), 0.0] if len(p) >= 2 else [0.0, 0.0, 0.0]
        for p in geometry_latlon
    ]
    coords = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3), dtype=np.float64)
    offset = Vec3((0.0, 0.0, height)) if height > 0 else Vec3.zeros()
    geom_type = "point" if coords.shape[0] == 1 else "polygon"
    return Geometry(type=geom_type, coords=coords, frame="WGS", offset=offset)


def refs_from_v2(data: dict[str, Any]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    osm_id = data.get("osm_id")
    if osm_id is not None:
        tags = data.get("osm_tags") or {}
        refs.append(
            SourceRef(
                assigned_by="osm",
                assigned_id=f"way/{osm_id}",
                metadata={str(k): str(v) for k, v in tags.items()},
            )
        )
    return refs


def name_desc_from_v2(data: dict[str, Any]) -> tuple[str, str]:
    name = str(data.get("name") or "")
    if "desc" in data:
        return name, str(data.get("desc") or "")
    old_text = str(data.get("text") or "")
    if name:
        return name, old_text
    return "", old_text


def scene_node_from_v2(data: dict[str, Any]) -> SceneNode:
    name, desc = name_desc_from_v2(data)
    height = float(data.get("height") or data.get("height_m", 0.0))
    geom = geom_from_v2(data.get("geometry_latlon", []), height=height)
    return SceneNode(
        id=data["id"],
        type=migrate_node_type(data["type"]),
        pid=data.get("pid"),
        geom=geom,
        refs=refs_from_v2(data),
        observation_count=int(data.get("observation_count", 0)),
        name=name,
        desc=desc,
        role=str(data.get("role") or ""),
        category=str(data.get("category") or ""),
        function=str(data.get("function") or ""),
    )
