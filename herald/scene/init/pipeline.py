"""Offline scene graph initialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from herald.scene.common.frame import LocalFrame
from herald.scene.common.graph import EventCallback, SceneEvent, SceneGraph, SceneNode
from herald.scene.common.progress import iter_progress
from herald.scene.common.roi import ROI
from herald.scene.init.classify import (
    NodeClassification,
    classify_from_osm_template,
    classify_hierarchy_nodes,
    fallback_classification,
    level_for_role,
    zone_kind_for_role,
)
from herald.scene.init.hierarchy import (
    SITE_OSM_ID,
    ContainmentForest,
    build_containment_forest,
    graph_node_id,
)
from herald.scene.init.pathways import filter_walkable_highways
from services.aerial.fetch import AerialMeta
from services.embeddings.encoder import Encoder, StubEncoder
from services.osm.client import LatLon, OSMClient, OSMFeature, OSMRawPolygon
from utils.osm_helpers import context_tags

CheckpointCallback = Callable[[SceneGraph, dict[int, NodeClassification]], None]


def _stage(message: str) -> None:
    print(f"  {message}", flush=True)


@dataclass
class BuildResult:
    graph: SceneGraph
    pathways: list[OSMFeature]
    forest: ContainmentForest
    classifications: dict[int, NodeClassification]


def _parse_height(tags: dict[str, str]) -> tuple[float, str]:
    if "height" in tags:
        raw = tags["height"].split()[0].replace("m", "")
        try:
            return float(raw), "osm_height_tag"
        except ValueError:
            pass
    if "building:levels" in tags:
        try:
            levels = float(tags["building:levels"].split()[0])
            return levels * 3.0, "osm_levels"
        except ValueError:
            pass
    return 10.0, "default"


def _node_event_payload(node: SceneNode, *, area_m2: float | None = None) -> dict:
    payload = {
        "level": node.level,
        "geometry_latlon": node.geometry_latlon,
        "role": node.role,
        "category": node.category,
        "function": node.function,
        "name": node.name or "",
        "description": node.text,
        "confidence": node.confidence,
        "classification_source": node.classification_source,
        "height_m": node.height_m,
        "height_source": node.height_source,
        "osm_id": node.osm_id,
        "osm_tags": node.osm_tags or {},
    }
    if area_m2 is not None:
        payload["area_m2"] = area_m2
    return payload


def _site_classification(roi: ROI, *, mapped_polygons: int) -> NodeClassification:
    lat_c, lon_c = roi.centroid_latlon()
    text = (
        f"site centered at ({lat_c:.5f}, {lon_c:.5f}), {mapped_polygons} mapped polygons"
    )
    return NodeClassification(
        role="site",
        category="ground_other",
        function="none",
        name="",
        description=text,
        confidence=1.0,
        source="template",
    )


def _emit(callback: EventCallback | None, event: SceneEvent) -> None:
    if callback is not None:
        callback(event)


def _resolve_classification(
    forest: ContainmentForest,
    osm_id: int,
    hierarchy_classifications: dict[int, NodeClassification],
    *,
    roi: ROI,
) -> NodeClassification:
    if osm_id == SITE_OSM_ID:
        return hierarchy_classifications.get(
            SITE_OSM_ID,
            _site_classification(
                roi, mapped_polygons=max(0, len(forest.nodes) - 1)
            ),
        )

    if osm_id in hierarchy_classifications:
        return hierarchy_classifications[osm_id]

    node = forest.nodes[osm_id]
    template = classify_from_osm_template(node)
    if template is not None:
        return template

    return fallback_classification(node)


def _node_depth(forest: ContainmentForest, osm_id: int) -> int:
    if osm_id == SITE_OSM_ID:
        return 0
    depth = 0
    parent_id = forest.nodes[osm_id].parent_osm_id
    while parent_id is not None and parent_id in forest.nodes:
        depth += 1
        parent_id = forest.nodes[parent_id].parent_osm_id
    return depth


def _classify_hierarchy(
    forest: ContainmentForest,
    roi: ROI,
    *,
    aerial_image: Image.Image | None,
    aerial_meta: AerialMeta | None,
    use_vlm: bool,
    vlm_model: str,
    on_event: EventCallback | None,
    show_progress: bool = True,
) -> dict[int, NodeClassification]:
    """Classify OSM polygons (site node uses a fixed template)."""
    polygon_nodes = [
        n for n in forest.nodes.values() if n.osm_id != SITE_OSM_ID
    ]
    ordered_nodes = sorted(
        polygon_nodes,
        key=lambda n: (_node_depth(forest, n.osm_id), n.osm_id),
    )
    depths = {n.osm_id: _node_depth(forest, n.osm_id) for n in ordered_nodes}
    total = len(ordered_nodes)

    _emit(
        on_event,
        SceneEvent(
            kind="pipeline_status",
            payload={
                "stage": "classify",
                "message": f"Classifying {total} hierarchy polygons",
                "total": total,
            },
        ),
    )

    if use_vlm:
        _stage(f"VLM classifying {total} polygon(s) (aerial + OSM per polygon)…")
    else:
        _stage(f"Classifying {total} polygon(s) from OSM tags (no VLM)…")

    def _on_classified(node, classification: NodeClassification) -> None:
        _emit(
            on_event,
            SceneEvent(
                kind="pipeline_status",
                payload={
                    "stage": "classify",
                    "osm_id": node.osm_id,
                    "message": f"Classified osm/{node.osm_id} ({classification.source})",
                    "source": classification.source,
                },
            ),
        )

    results = classify_hierarchy_nodes(
        ordered_nodes,
        use_vlm=use_vlm,
        aerial_image=aerial_image,
        aerial_meta=aerial_meta,
        vlm_model=vlm_model,
        depths=depths,
        show_progress=show_progress,
        on_node_classified=_on_classified,
    )
    results[SITE_OSM_ID] = _site_classification(roi, mapped_polygons=total)
    return results


CHECKPOINT_INTERVAL = 25


def build_scene_graph(
    roi: ROI,
    raw_polygons: list[OSMRawPolygon],
    *,
    client: OSMClient | None = None,
    highways: list[OSMFeature] | None = None,
    encoder: Encoder | None = None,
    on_event: EventCallback | None = None,
    on_checkpoint: CheckpointCallback | None = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    aerial_image: Image.Image | None = None,
    aerial_meta: AerialMeta | None = None,
    use_vlm: bool = False,
    vlm_model: str = "ollama:qwen2.5vl:3b",
    show_progress: bool = True,
) -> BuildResult:
    """Build a containment-based scene graph from raw OSM polygons."""
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

    _stage("Computing containment hierarchy…")
    forest = build_containment_forest(raw_polygons, roi)
    _stage(
        f"Hierarchy: {len(forest.nodes)} nodes, "
        f"{len(forest.site_children)} site children, "
        f"{len(forest.others)} other polygons"
    )
    pending_preview = [
        {
            "osm_id": node.osm_id,
            "geometry_latlon": node.geom,
            "area_m2": node.area,
            "parent_osm_id": node.parent_osm_id,
        }
        for node in forest.nodes.values()
    ]
    _emit(
        on_event,
        SceneEvent(
            kind="partition_computed",
            payload={
                "hierarchy_nodes": len(forest.nodes),
                "site_children": len(forest.site_children),
                "others": len(forest.others),
                "pending_nodes": pending_preview,
            },
        ),
    )

    _stage("Extracting walkable pathways from OSM data…")
    if highways is not None:
        highways_raw = highways
    else:
        highways_raw = osm_client.query_in_polygon(roi.latlon_vertices()).highways
    walkable = filter_walkable_highways(highways_raw)
    _stage(f"Pathways: {len(walkable)} walkable segments")

    hierarchy_classifications = _classify_hierarchy(
        forest,
        roi,
        aerial_image=aerial_image,
        aerial_meta=aerial_meta,
        use_vlm=use_vlm,
        vlm_model=vlm_model,
        on_event=on_event,
        show_progress=show_progress,
    )

    frame = LocalFrame(origin=LatLon(lat=lat_c, lon=lon_c))
    graph = SceneGraph(
        site_id="site_000",
        frame=frame,
        embedding_model_id=enc.model_id,
        roi_area_m2=roi.area_m2(),
        schema_version=2,
    )

    ordered_osm_ids = sorted(
        forest.nodes.keys(),
        key=lambda osm_id: (_node_depth(forest, osm_id), osm_id),
    )
    all_classifications: dict[int, NodeClassification] = {}

    for step, osm_id in enumerate(
        iter_progress(
            ordered_osm_ids,
            desc="Building scene nodes",
            total=len(ordered_osm_ids),
            disable=not show_progress,
        ),
        start=1,
    ):
        hnode = forest.nodes[osm_id]
        classification = _resolve_classification(
            forest, osm_id, hierarchy_classifications, roi=roi
        )
        all_classifications[osm_id] = classification

        _emit(
            on_event,
            SceneEvent(
                kind="pipeline_status",
                payload={
                    "stage": "build",
                    "index": step,
                    "total": len(ordered_osm_ids),
                    "osm_id": osm_id,
                    "message": f"Building node {step}/{len(ordered_osm_ids)} (osm/{osm_id})",
                },
            ),
        )

        if osm_id == SITE_OSM_ID:
            level = "site"
            zone_kind = None
            height_m = 0.0
            height_source = "default"
        else:
            height_m, height_source = _parse_height(hnode.tags)
            level = level_for_role(classification.role)  # type: ignore[assignment]
            zone_kind = zone_kind_for_role(classification.role)  # type: ignore[assignment]

        parent_osm_id = hnode.parent_osm_id
        parent_graph_id = (
            graph_node_id(parent_osm_id) if parent_osm_id is not None else None
        )

        emb = enc.encode([classification.description])[0].tolist()
        scene_node = SceneNode(
            id=graph_node_id(osm_id),
            level=level,  # type: ignore[arg-type]
            zone_kind=zone_kind,  # type: ignore[arg-type]
            text=classification.description,
            geometry_latlon=hnode.geom,
            embedding=emb,
            height_m=height_m if classification.role == "structure" else 0.0,
            height_source=height_source if classification.role == "structure" else "default",  # type: ignore[arg-type]
            osm_id=osm_id,
            role=classification.role,
            category=classification.category,
            function=classification.function,
            name=classification.name or None,
            confidence=classification.confidence,
            classification_source=classification.source,
            osm_tags=None
            if osm_id == SITE_OSM_ID
            else (context_tags(hnode.tags) or None),
        )
        graph.add_node(scene_node)
        if parent_graph_id is not None:
            graph.add_edge(parent_graph_id, scene_node.id, edge_type="contains")
        _emit(
            on_event,
            SceneEvent(
                kind="zone_node_created",
                node_id=scene_node.id,
                parent_id=parent_graph_id,
                payload=_node_event_payload(scene_node, area_m2=hnode.area),
            ),
        )
        if parent_graph_id is not None:
            _emit(
                on_event,
                SceneEvent(
                    kind="containment_inferred",
                    node_id=scene_node.id,
                    parent_id=parent_graph_id,
                ),
            )
        _emit(
            on_event,
            SceneEvent(
                kind="embedding_attached",
                node_id=scene_node.id,
                payload={"text": classification.description[:80]},
            ),
        )
        if on_checkpoint is not None and (
            step % checkpoint_interval == 0 or step == len(ordered_osm_ids)
        ):
            on_checkpoint(graph, dict(all_classifications))

    _emit(
        on_event,
        SceneEvent(
            kind="pathways_ready",
            payload={"count": len(walkable)},
        ),
    )

    _emit(
        on_event,
        SceneEvent(kind="pipeline_complete", payload={"node_count": len(graph.nodes)}),
    )
    return BuildResult(
        graph=graph,
        pathways=walkable,
        forest=forest,
        classifications=all_classifications,
    )
