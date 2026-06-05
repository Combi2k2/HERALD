"""GeoJSON and annotation I/O for scene-init artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from herald.scene.init.classify import NodeClassification
from services.osm.client import OSMFeature, OSMRawPolygon


def classifications_to_json(classifications: dict[int, NodeClassification]) -> dict:
    return {
        str(osm_id): {
            "role": c.role,
            "category": c.category,
            "function": c.function,
            "name": c.name,
            "description": c.description,
            "confidence": c.confidence,
            "source": c.source,
        }
        for osm_id, c in sorted(classifications.items())
    }


def save_classifications(path: Path, classifications: dict[int, NodeClassification]) -> None:
    path.write_text(
        json.dumps(classifications_to_json(classifications), indent=2),
        encoding="utf-8",
    )


def pathways_to_geojson(pathways: list[OSMFeature]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [f.to_geojson_feature() for f in pathways],
    }


def pathways_from_geojson(data: dict) -> list[OSMFeature]:
    """Rehydrate pathway features saved as GeoJSON."""
    features: list[OSMFeature] = []
    for i, feat in enumerate(data.get("features", [])):
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = [(float(lat), float(lon)) for lon, lat in geom["coordinates"]]
        props = {str(k): str(v) for k, v in (feat.get("properties") or {}).items()}
        features.append(
            OSMFeature(
                osm_id=int(props.get("osm_id", i)),
                osm_type="way",
                tags=props,
                geometry=tuple(coords),
                kind="linestring",
                category="highway",
            )
        )
    return features


def raw_polygons_from_geojson(data: dict) -> list[OSMRawPolygon]:
    """Rehydrate raw OSM polygons saved as GeoJSON."""
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
