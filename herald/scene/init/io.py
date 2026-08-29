"""GeoJSON and annotation I/O for scene-init artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from herald.scene.common.graph import SceneGraph
from services.osm import OSMRawPolygon


def annotations_from_graph(graph: SceneGraph) -> dict:
    return {
        node.uid: {
            "role": node.attrs.get("role", ""),
            "category": node.attrs.get("category", ""),
            "function": node.attrs.get("function", ""),
            "name": node.name,
            "description": node.desc,
        }
        for node in sorted(graph.nodes, key=lambda n: n.uid)
        if node.attrs.get("role")
    }


def save_annotations(path: Path, graph: SceneGraph) -> None:
    path.write_text(json.dumps(annotations_from_graph(graph), indent=2), encoding="utf-8")


def raw_polygons_from_geojson(data: dict) -> list[OSMRawPolygon]:
    polygons: list[OSMRawPolygon] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        coords = [(float(lat), float(lon)) for lon, lat in ring]
        props = {str(k): str(v) for k, v in (feat.get("properties") or {}).items()}
        fid = str(feat.get("id", "way/0"))
        osm_type, _, osm_id_str = fid.partition("/")
        polygons.append(
            OSMRawPolygon(
                osm_id=int(osm_id_str or props.get("osm_id", 0)),
                osm_type=osm_type if osm_type in ("way", "relation") else "way",  # type: ignore[arg-type]
                tags=props,
                geometry=tuple(coords),
                primary_tag=props.get("primary_tag", "other"),  # type: ignore[arg-type]
                primary_value=props.get("primary_value", "unknown"),
            )
        )
    return polygons
