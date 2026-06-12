"""Offline scene graph initialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from shapely.geometry import Polygon
from PIL import Image

from herald.scene.common.geometry import Frame, Geometry, Vec3
from herald.scene.common._legacy_events import EventCallback, SceneEvent
from herald.scene.common.graph import SceneGraph, SceneNode, SourceRef
from herald.scene.common.progress import iter_progress
from herald.scene.common.roi import ROI
from herald.scene.init.classify import build_feat, level_for_role
from herald.scene.init.hierarchy import build_tree
from herald.scene.init.pathways import filter_walkable_highways
from services.aerial.fetch import AerialMeta
from services.embeddings.encoder import Encoder, StubEncoder
from services.osm.client import OSMClient, OSMFeature, OSMRawPolygon
from utils.osm_filter import polygon_disposition
from utils.osm_helpers import context_tags

SITE_NODE_ID = "site_000"
MIN_NODE_AREA = 30.0
MAX_NODE_AREA = 20_000.0


def _closed_ring(coords) -> list[list[float]]:
    ring = [[float(p[0]), float(p[1]), 0.0] for p in coords]
    if len(ring) < 3:
        return []
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _seed_graph(
    raw_polygons: list[OSMRawPolygon],
    roi: ROI,
    frame: Frame,
    *,
    emb_model_id: str,
    vlm_model_id: str,
) -> SceneGraph:
    graph = SceneGraph(emb_model_id=emb_model_id, vlm_model_id=vlm_model_id)
    graph.add_node(
        SceneNode(
            id=SITE_NODE_ID,
            pid=None,
            type="site",
            geom=Geometry(type="polygon", frame="WGS", coords=_closed_ring(roi.latlon_vertices())),
        )
    )
    for poly in raw_polygons:
        ring = _closed_ring(poly.geometry)
        if len(ring) < 4:
            continue
        area = float(Polygon([frame.wgs2utm(p[0], p[1]) for p in ring]).area)
        if polygon_disposition(
            poly.tags, area=area, min_area=MIN_NODE_AREA, max_area=MAX_NODE_AREA
        ) != "hierarchy":
            continue
        graph.add_node(
            SceneNode(
                id=f"node_{poly.osm_id}",
                pid=SITE_NODE_ID,
                type="region",
                geom=Geometry(type="polygon", frame="WGS", coords=ring),
                refs=[
                    SourceRef(
                        assigned_by="osm",
                        assigned_id=f"{poly.osm_type}/{poly.osm_id}",
                        metadata={"tags": dict(poly.tags), "area": area},
                    )
                ],
            )
        )
    return graph

CheckpointCallback = Callable[[SceneGraph], None]


def _stage(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _emit(cb: EventCallback | None, event: SceneEvent) -> None:
    if cb is not None:
        cb(event)


def _tags(node: SceneNode) -> dict[str, str]:
    for ref in node.refs:
        if ref.assigned_by == "osm":
            raw = ref.metadata.get("tags")
            if isinstance(raw, dict):
                return dict(raw)
    return {}


def _parse_height(tags: dict[str, str]) -> float:
    if "height" in tags:
        try:
            return float(tags["height"].split()[0].replace("m", ""))
        except ValueError:
            pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"].split()[0]) * 3.0
        except ValueError:
            pass
    return 10.0


def _finalize(graph: SceneGraph, frame: Frame, *, show_progress: bool) -> None:
    for node in iter_progress(
        graph.nodes,
        desc="Finalizing geometry",
        total=len(graph.nodes),
        disable=not show_progress,
    ):
        offset = Vec3.zeros()
        if node.type != "site":
            node.type = level_for_role(node.role)  # type: ignore[assignment]
            if node.refs:
                node.refs[0].metadata = context_tags(_tags(node))
            if node.role == "structure":
                h = _parse_height(_tags(node))
                if h > 0:
                    offset = Vec3((0.0, 0.0, h))
        ring = [(float(r[0]), float(r[1])) for r in node.geom.coords]
        enu = np.array([[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in ring], dtype=np.float64)
        node.geom = Geometry(
            type="point" if enu.shape[0] == 1 else "polygon",
            coords=enu,
            frame="ENU",
            offset=offset,
        )


@dataclass
class BuildResult:
    graph: SceneGraph
    frame: Frame
    pathways: list[OSMFeature]


def build_scene_graph(
    roi: ROI,
    raw_polygons: list[OSMRawPolygon],
    *,
    frame: Frame,
    client: OSMClient | None = None,
    highways: list[OSMFeature] | None = None,
    encoder: Encoder | None = None,
    on_event: EventCallback | None = None,
    on_checkpoint: CheckpointCallback | None = None,
    checkpoint_interval: int = 25,
    aerial_image: Image.Image | None = None,
    aerial_meta: AerialMeta | None = None,
    use_vlm: bool = False,
    vlm_model: str = "ollama:qwen2.5vl:3b",
    show_progress: bool = True,
) -> BuildResult:
    enc = encoder or StubEncoder()
    osm_client = client or OSMClient()

    lat_c, lon_c = roi.latlon_centroid()
    _emit(
        on_event,
        SceneEvent(
            kind="roi_resolved",
            payload={
                "centroid": {"lat": lat_c, "lon": lon_c},
                "area": roi.area(),
                "vertices": roi.latlon_vertices(),
            },
        ),
    )

    graph = _seed_graph(
        raw_polygons,
        roi,
        frame,
        emb_model_id=enc.model_id,
        vlm_model_id=vlm_model,
    )

    _stage("Computing containment hierarchy…")
    build_tree(graph, frame)
    site = graph.site_node()
    site_children = sum(1 for n in graph.nodes if n.pid == SITE_NODE_ID)
    _stage(f"Hierarchy: {len(graph.nodes)} nodes, {site_children} site children")
    _emit(
        on_event,
        SceneEvent(
            kind="partition_computed",
            payload={
                "hierarchy_nodes": len(graph.nodes),
                "site_children": site_children,
            },
        ),
    )

    _stage("Extracting walkable pathways from OSM data…")
    if highways is None:
        highways = osm_client.query_in_polygon(roi.latlon_vertices()).highways
    walkable = filter_walkable_highways(highways)
    _stage(f"Pathways: {len(walkable)} walkable segments")

    _stage(
        f"{'VLM classifying' if use_vlm else 'Classifying'} "
        f"{len(graph.nodes) - 1} polygon(s)…"
    )
    build_feat(
        graph,
        frame,
        aerial=aerial_image,
        aerial_zoom=aerial_meta.zoom if aerial_meta else 18,
        use_vlm=use_vlm,
        vlm_model=vlm_model,
        show_progress=show_progress,
    )

    _finalize(graph, frame, show_progress=show_progress)

    if on_checkpoint is not None:
        on_checkpoint(graph)

    _emit(on_event, SceneEvent(kind="pathways_ready", payload={"count": len(walkable)}))
    _emit(
        on_event,
        SceneEvent(kind="pipeline_complete", payload={"node_count": len(graph.nodes)}),
    )
    return BuildResult(graph=graph, frame=frame, pathways=walkable)
