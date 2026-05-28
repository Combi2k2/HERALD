"""Offline scene graph initialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from herald.osm.client import LatLon, OSMClient, OSMFeature
from herald.scene.embedding import Encoder, StubEncoder
from herald.scene.frame import LocalFrame
from herald.scene.graph import EventCallback, SceneEvent, SceneGraph, SceneNode
from herald.scene.partition import (
    extract_buildings,
    filter_walkable_highways,
    partition_outdoor_zones,
)
from herald.scene.roi import ROI

if TYPE_CHECKING:
    pass


@dataclass
class BuildResult:
    graph: SceneGraph
    pathways: list[OSMFeature]


def _compose_zone_caption(feat_tags: dict[str, str], *, kind: str, name: str | None) -> str:
    label = name or "unnamed"
    top_tags = sorted(feat_tags.items())[:5]
    tag_str = ", ".join(f"{k}={v}" for k, v in top_tags)
    return f"{kind} named {label}, tags: {tag_str}"


def _compose_building_caption(tags: dict[str, str], name: str | None) -> str:
    building_type = tags.get("building", "yes")
    return _compose_zone_caption(tags, kind=f"building ({building_type})", name=name)


def _compose_outdoor_caption(zone_id: str, area_m2: float) -> str:
    return f"outdoor region {zone_id}, area {area_m2:.0f} m²"


def _emit(callback: EventCallback | None, event: SceneEvent) -> None:
    if callback is not None:
        callback(event)


def build_scene_graph(
    roi: ROI,
    *,
    client: OSMClient | None = None,
    encoder: Encoder | None = None,
    on_event: EventCallback | None = None,
) -> BuildResult:
    """Build a hierarchical scene graph from OSM data within ``roi``."""
    osm_client = client or OSMClient()
    enc = encoder or StubEncoder()

    lat_c, lon_c = roi.centroid_latlon()
    _emit(
        on_event,
        SceneEvent(
            kind="roi_resolved",
            payload={
                "centroid": {"lat": lat_c, "lon": lon_c},
                "area_m2": roi.area_m2(),
                "vertices": roi.latlon_vertices(),
            },
        ),
    )

    raw = osm_client.query_in_polygon(roi.latlon_vertices())
    buildings_raw = roi.clip_features(raw.buildings)
    highways_raw = roi.clip_features(raw.highways)

    buildings = extract_buildings(buildings_raw)
    walkable = filter_walkable_highways(highways_raw)
    partition = partition_outdoor_zones(roi, buildings, walkable)

    _emit(
        on_event,
        SceneEvent(
            kind="partition_computed",
            payload={
                "outdoor_count": len(partition.outdoor_zones),
                "building_count": len(partition.buildings),
            },
        ),
    )

    frame = LocalFrame(origin=LatLon(lat=lat_c, lon=lon_c))
    graph = SceneGraph(
        site_id="site_000",
        frame=frame,
        embedding_model_id=enc.model_id,
        roi_area_m2=roi.area_m2(),
    )

    site_text = f"site centered at ({lat_c:.5f}, {lon_c:.5f})"
    site_emb = enc.encode([site_text])[0].tolist()
    site_node = SceneNode(
        id=graph.site_id,
        level="site",
        zone_kind=None,
        text=site_text,
        geometry_latlon=roi.latlon_vertices(),
        embedding=site_emb,
        height_m=0.0,
        height_source="default",
    )
    graph.add_node(site_node)
    _emit(
        on_event,
        SceneEvent(
            kind="zone_node_created",
            node_id=site_node.id,
            payload={"level": "site"},
        ),
    )

    outdoor_nodes: dict[str, SceneNode] = {}
    outdoor_texts: list[str] = []
    outdoor_ids: list[str] = []

    for zone in partition.outdoor_zones:
        text = _compose_outdoor_caption(zone.zone_id, zone.area_m2)
        outdoor_texts.append(text)
        outdoor_ids.append(zone.zone_id)

    if outdoor_texts:
        outdoor_embs = enc.encode(outdoor_texts)
    else:
        outdoor_embs = []

    for zone, text, emb in zip(partition.outdoor_zones, outdoor_texts, outdoor_embs):
        node = SceneNode(
            id=zone.zone_id,
            level="outdoor_region",
            zone_kind="outdoor_region",
            text=text,
            geometry_latlon=zone.polygon_latlon,
            embedding=emb.tolist(),
        )
        graph.add_node(node)
        graph.add_edge(graph.site_id, node.id, edge_type="contains")
        outdoor_nodes[node.id] = node
        _emit(
            on_event,
            SceneEvent(
                kind="zone_node_created",
                node_id=node.id,
                parent_id=graph.site_id,
                payload={
                    "level": "outdoor_region",
                    "area_m2": zone.area_m2,
                    "geometry_latlon": zone.polygon_latlon,
                },
            ),
        )
        _emit(
            on_event,
            SceneEvent(
                kind="containment_inferred",
                node_id=node.id,
                parent_id=graph.site_id,
            ),
        )

    building_texts = [
        _compose_building_caption(b.tags, b.name) for b in partition.buildings
    ]
    if building_texts:
        building_embs = enc.encode(building_texts)
    else:
        building_embs = []

    for b, text, emb in zip(partition.buildings, building_texts, building_embs):
        node_id = f"building_{b.osm_id}"
        parent_id = partition.building_to_outdoor.get(b.osm_id)
        node = SceneNode(
            id=node_id,
            level="building",
            zone_kind="building",
            text=text,
            geometry_latlon=b.footprint_latlon,
            embedding=emb.tolist(),
            height_m=b.height_m,
            height_source=b.height_source,  # type: ignore[arg-type]
            osm_id=b.osm_id,
        )
        graph.add_node(node)
        if parent_id is not None:
            graph.add_edge(parent_id, node.id, edge_type="contains")
            _emit(
                on_event,
                SceneEvent(
                    kind="containment_inferred",
                    node_id=node.id,
                    parent_id=parent_id,
                ),
            )
        _emit(
            on_event,
            SceneEvent(
                kind="zone_node_created",
                node_id=node.id,
                parent_id=parent_id,
                payload={
                    "level": "building",
                    "osm_id": b.osm_id,
                    "geometry_latlon": b.footprint_latlon,
                    "height_m": b.height_m,
                },
            ),
        )
        _emit(
            on_event,
            SceneEvent(
                kind="embedding_attached",
                node_id=node.id,
                payload={"text": text[:80]},
            ),
        )

    _emit(on_event, SceneEvent(kind="pipeline_complete", payload={"node_count": len(graph.nodes)}))
    return BuildResult(graph=graph, pathways=partition.pathway_features)


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


def raw_polygons_from_geojson(data: dict) -> list:
    """Rehydrate raw OSM polygons saved as GeoJSON."""
    from herald.osm.client import OSMRawPolygon

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
