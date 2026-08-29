"""Offline scene graph initialization from OSM polygons."""

from __future__ import annotations

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from herald.scene.common.geometry import Frame, Geometry
from herald.scene.common.graph import SceneGraph, SceneNode, SourceRef
from herald.scene.common.progress import iter_progress
from herald.scene.common.repr import SceneRepr
from herald.scene.common.roi import ROI

from herald.scene.init.hierarchy import build_tree, MAX_NODE_AREA, MIN_NODE_AREA
from herald.scene.init.classify import build_feat
from herald.scene.init.pathways import build_path

from services.aerial import AerialMeta
from services.embeddings.encoder import Encoder, StubEncoder
from services.osm import OSMClient, OSMFeature, OSMRawPolygon
from services.vlm import VLMClient

from utils.osm_filter import polygon_disposition
from utils.osm_helpers import context_tags, osm_tags

SITE_UID = 0


def _wgs_ring(coords) -> list[list[float]]:
    ring = [[float(p[0]), float(p[1]), 0.0] for p in coords]
    if len(ring) < 3:
        return []
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def seed_graph(
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
            uid=SITE_UID,
            parent=None,
            level="site",
            geom=Geometry(type="polygon", frame="WGS", coords=_wgs_ring(roi.latlon_vertices())),
        )
    )

    uid = SITE_UID
    for poly in raw_polygons:
        ring = _wgs_ring(poly.geometry)
        if len(ring) < 4:
            continue
        area = float(Polygon([frame.wgs2utm(p[0], p[1]) for p in ring]).area)
        if polygon_disposition(
            poly.tags,
            area=area,
            min_area=MIN_NODE_AREA,
            max_area=MAX_NODE_AREA,
        ) != "hierarchy":
            continue
        uid += 1
        graph.add_node(
            SceneNode(
                uid=uid,
                parent=SITE_UID,
                level="region",
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


def build_scene_graph(
    roi: ROI,
    raw_polygons: list[OSMRawPolygon],
    *,
    frame: Frame,
    client: OSMClient | None = None,
    highways: list[OSMFeature] | None = None,
    encoder: Encoder | None = None,
    aerial_image: Image.Image | None = None,
    osm_map_image: Image.Image | None = None,
    aerial_meta: AerialMeta | None = None,
    use_vlm: bool = False,
    vlm_model: str = "ollama:qwen2.5vl:3b",
    show_progress: bool = True,
) -> SceneRepr:
    enc = encoder or StubEncoder()
    osm = client or OSMClient()

    graph = seed_graph(
        raw_polygons,
        roi,
        frame,
        emb_model_id=enc.model_id,
        vlm_model_id=vlm_model,
    )

    print("  Computing containment hierarchy…", flush=True)
    build_tree(graph, frame)
    print(f"  Hierarchy: {len(graph.nodes)} nodes", flush=True)

    mode = "VLM classifying" if use_vlm else "Classifying"
    print(f"  {mode} {len(graph.nodes) - 1} polygon(s)…", flush=True)
    build_feat(
        graph,
        frame,
        VLMClient(vlm_model) if use_vlm else None,
        aerial_image,
        osm_map_image,
        mosaic_bbox=aerial_meta.bbox if aerial_meta else None,
        mosaic_zoom=aerial_meta.zoom if aerial_meta else 18,
        show_progress=show_progress,
    )

    for node in iter_progress(
        graph.nodes,
        desc="Finalizing geometry",
        total=len(graph.nodes),
        disable=not show_progress,
    ):
        if node.level != "site" and node.refs:
            node.refs[0].metadata = context_tags(osm_tags(node))
        ring = [(float(r[0]), float(r[1])) for r in node.geom.coords]
        enu = np.array([[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in ring], dtype=np.float64)
        node.geom = Geometry(
            type="point" if enu.shape[0] == 1 else "polygon",
            coords=enu,
            frame="ENU",
        )

    print("  Building navigation graph…", flush=True)
    if highways is None:
        highways = osm.query_in_polygon(roi.latlon_vertices()).highways
    nav = build_path(highways, frame)
    print(
        f"  Nav graph: {len(nav.nodes)} nodes, {len(nav.edges)} edges",
        flush=True,
    )

    return SceneRepr(frame=frame, graph=graph, nav=nav)
