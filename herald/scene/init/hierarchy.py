"""Build a containment hierarchy from OSM polygon footprints."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph, SceneEdge

MIN_NODE_AREA = 30.0
MAX_NODE_AREA = 20_000.0
CONTAINMENT_RATIO = 0.90


def build_tree(
    graph: SceneGraph,
    frame: Frame,
    *,
    min_area: float = MIN_NODE_AREA,
    max_area: float = MAX_NODE_AREA,
    containment_ratio: float = CONTAINMENT_RATIO,
) -> None:
    records: list[tuple[Polygon, float, int]] = []

    for i, node in enumerate(graph.nodes):
        ring = [[p[0], p[1]] for p in node.geom.coords]
        ring = ring + ([ring[0]] if ring[0] != ring[-1] else [])

        if len(ring) < 3:
            continue

        if node.geom.frame == "WGS":    ring = [frame.wgs2utm(*p) for p in ring]
        if node.geom.frame == "ENU":    ring = [frame.enu2utm(*p) for p in ring]

        poly = Polygon(ring)
        area = float(poly.area)

        if node.level != "site":
            if area < min_area: continue
            if area > max_area: continue

        records.append((poly, area, i))
    
    if not records:
        return

    tree = STRtree([p for p, _, _ in records])

    for poly, area, id in records:
        best_area = float("inf")
        best_pid: int | None = None
        for j in tree.query(poly, predicate="intersects"):
            parent_poly, parent_area, pid = records[j]
            if pid == id or parent_area <= area:
                continue
            if parent_poly.intersection(poly).area >= containment_ratio * area and parent_area < best_area:
                best_area = parent_area
                best_pid = pid

        if best_pid is not None:
            graph.nodes[id].parent = graph.nodes[best_pid].uid
            graph.edges.append(SceneEdge(
                graph.nodes[best_pid].uid,
                graph.nodes[id].uid,
                "contains"
            ))
