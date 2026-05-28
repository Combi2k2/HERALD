"""Rerun-based 3D scene graph viewer.

Local ENU is mapped into Rerun as RIGHT_HAND_Z_UP:
  X = East, Y = North, Z = Up — ground lies in the XY plane at Z=0.

Polygons are drawn as closed LineStrips3D through their stored vertices (no
retriangulation), matching the Folium / GeoJSON rings exactly.

RUF (X=east, Y=up, Z=north) is left-handed and Rerun mirrors it; do not use it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from herald.osm.client import OSMFeature, OSMRawPolygon
from herald.scene.frame import LocalFrame
from herald.scene.graph import SceneEvent, SceneGraph, SceneNode

if TYPE_CHECKING:
    pass

# Line radii in metres (campus-scale view needs visible width, not cm strokes).
_RADIUS_CONTOUR = 0.5
_RADIUS_RAW_POLYGON = 1.0
_RADIUS_PATHWAY = 1.0
_GROUND_OFFSET = 0.02


def _closed_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ring:
        return ring
    if ring[0] == ring[-1]:
        return list(ring)
    return [*ring, ring[0]]


# RGBA 0–255
_COLOR_ROI = [255, 255, 255, 180]
_COLOR_OUTDOOR = [56, 189, 248, 220]
_COLOR_BUILDING_FOOTPRINT = [59, 130, 246, 255]
_COLOR_BUILDING_BOX = [37, 99, 235, 120]
_COLOR_PATHWAY = [234, 88, 12, 255]

# Raw OSM polygon layers (RGBA 0–255), keyed by primary tag
_RAW_OSM_COLORS: dict[str, list[int]] = {
    "building": [96, 165, 250, 200],
    "landuse": [74, 222, 128, 170],
    "leisure": [192, 132, 252, 170],
    "amenity": [244, 114, 182, 170],
    "natural": [52, 211, 153, 170],
    "water": [34, 211, 238, 190],
    "waterway": [6, 182, 212, 190],
    "highway": [251, 146, 60, 170],
    "place": [250, 204, 21, 150],
    "man_made": [148, 163, 184, 170],
    "barrier": [239, 68, 68, 150],
    "untagged": [250, 250, 250, 120],
    "other": [156, 163, 175, 140],
}
_RAW_DEFAULT_COLOR = [156, 163, 175, 140]


def _require_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "rerun-sdk is required for the viewer. Install with: uv sync --group viewer"
        ) from exc
    return rr


def _enu_to_rr(east: float, north: float, up: float = 0.0) -> list[float]:
    """Map local East-North-Up metres to Rerun RIGHT_HAND_Z_UP coordinates."""
    return [east, north, up]


class RerunSceneViewer:
    """Stream scene graph construction events to Rerun."""

    def __init__(
        self,
        *,
        app_id: str = "herald-scene",
        spawn: bool = True,
        show_building_boxes: bool = False,
    ) -> None:
        rr = _require_rerun()
        rr.init(app_id, spawn=spawn)
        # RIGHT_HAND_Z_UP: X=East, Y=North, Z=Up — campus map on the XY ground plane.
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._rr = rr
        self._frame: LocalFrame | None = None
        self._show_building_boxes = show_building_boxes

    def set_frame(self, frame: LocalFrame) -> None:
        self._frame = frame

    def publish(self, event: SceneEvent) -> None:
        if event.kind == "roi_resolved":
            self._log_roi(event)
        elif event.kind == "zone_node_created":
            self._log_zone(event)
        elif event.kind == "containment_inferred":
            self._log_containment(event)

    def render_graph(
        self,
        graph: SceneGraph,
        *,
        pathways: list[OSMFeature] | None = None,
    ) -> None:
        """Render a complete saved graph (offline replay)."""
        self.set_frame(graph.frame)
        site = next((n for n in graph.nodes if n.level == "site"), graph.nodes[0])
        self._log_contour("site/roi/contour", site.geometry_latlon, _COLOR_ROI)

        for node in graph.nodes:
            if node.level == "outdoor_region":
                self._log_contour(
                    f"site/zones/{node.id}/contour",
                    node.geometry_latlon,
                    _COLOR_OUTDOOR,
                )
            elif node.level == "building":
                parent = self._parent_of(graph, node.id) or "unassigned"
                self._log_building(node, parent)

        if pathways:
            self._log_pathways(pathways)

        for edge in graph.edges:
            if edge.edge_type != "contains":
                continue
            parent = graph.get_node(edge.source_id)
            child = graph.get_node(edge.target_id)
            if parent is None or child is None:
                continue
            self._log_edge(parent, child, edge.source_id, edge.target_id)

    def render_raw_osm_polygons(
        self,
        polygons: list[OSMRawPolygon],
        *,
        static: bool = True,
    ) -> None:
        """Draw raw OSM polygon rings exactly as stored, grouped by primary tag."""
        if self._frame is None or not polygons:
            return
        by_tag: dict[str, list[OSMRawPolygon]] = {}
        for poly in polygons:
            by_tag.setdefault(poly.primary_tag, []).append(poly)

        for tag, group in sorted(by_tag.items()):
            color = _RAW_OSM_COLORS.get(tag, _RAW_DEFAULT_COLOR)
            strips: list[list[list[float]]] = []
            colors: list[list[int]] = []
            for poly in group:
                strip = self._ring_to_rr(_closed_ring(list(poly.geometry)))
                if len(strip) < 3:
                    continue
                strips.append(strip)
                colors.append(color)
            if not strips:
                continue
            self._rr.log(
                f"osm/raw/{tag}",
                self._rr.LineStrips3D(
                    strips,
                    colors=colors,
                    radii=[_RADIUS_RAW_POLYGON],
                ),
                static=static,
            )

    def _parent_of(self, graph: SceneGraph, node_id: str) -> str | None:
        for edge in graph.edges:
            if edge.target_id == node_id and edge.edge_type == "contains":
                return edge.source_id
        return None

    def _ring_to_rr(self, ring: list[tuple[float, float]], *, up: float = 0.05) -> list[list[float]]:
        """Project a lat/lon ring to Rerun; vertices follow the source polygon order."""
        if self._frame is None:
            return [[0.0, 0.0, up]]
        return [
            _enu_to_rr(*self._frame.to_local(lat, lon), up)
            for lat, lon in ring
        ]

    def _line_to_rr(self, coords: list[tuple[float, float]], *, up: float = 0.1) -> list[list[float]]:
        if self._frame is None:
            return [[0.0, 0.0, up]]
        return [
            _enu_to_rr(*self._frame.to_local(lat, lon), up)
            for lat, lon in coords
        ]

    def _centroid_rr(
        self, ring: list[tuple[float, float]], *, up: float = 0.0
    ) -> tuple[float, float, float]:
        if not ring or self._frame is None:
            return (0.0, 0.0, up)
        east = north = 0.0
        for lat, lon in ring:
            e, n = self._frame.to_local(lat, lon)
            east += e
            north += n
        n_pts = len(ring)
        return _enu_to_rr(east / n_pts, north / n_pts, up)

    def _bbox_from_footprint(
        self, ring: list[tuple[float, float]], height_m: float
    ) -> tuple[list[list[float]], list[list[float]]]:
        if self._frame is None or not ring:
            return ([[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]])
        easts, norths = [], []
        for lat, lon in ring:
            e, n = self._frame.to_local(lat, lon)
            easts.append(e)
            norths.append(n)
        min_e, max_e = min(easts), max(easts)
        min_n, max_n = min(norths), max(norths)
        cx = (min_e + max_e) / 2
        cy = (min_n + max_n) / 2
        sx = max(max_e - min_e, 1.0)
        sy = max(max_n - min_n, 1.0)
        centers = [[cx, cy, height_m / 2]]
        half_sizes = [[sx / 2, sy / 2, height_m / 2]]
        return centers, half_sizes

    def _log_contour(
        self,
        path: str,
        ring: list[tuple[float, float]],
        color: list[int],
        *,
        static: bool = True,
        radius: float = _RADIUS_CONTOUR,
    ) -> None:
        strip = self._ring_to_rr(_closed_ring(ring), up=_GROUND_OFFSET + 0.03)
        if len(strip) < 3:
            return
        self._rr.log(
            path,
            self._rr.LineStrips3D(
                [strip],
                colors=[color],
                radii=[radius],
            ),
            static=static,
        )

    def _log_building(self, node: SceneNode, parent_seg: str, *, static: bool = True) -> None:
        base = f"site/zones/{parent_seg}/buildings/{node.id}"
        self._log_contour(
            f"{base}/footprint",
            node.geometry_latlon,
            _COLOR_BUILDING_FOOTPRINT,
            static=static,
            radius=_RADIUS_CONTOUR,
        )
        if not self._show_building_boxes:
            return
        centers, half_sizes = self._bbox_from_footprint(
            node.geometry_latlon, node.height_m
        )
        self._rr.log(
            f"{base}/box",
            self._rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                colors=[_COLOR_BUILDING_BOX],
            ),
            static=static,
        )

    def _log_pathways(
        self, pathways: list[OSMFeature], *, static: bool = True
    ) -> None:
        strips = []
        for hw in pathways:
            if hw.kind != "linestring" or len(hw.geometry) < 2:
                continue
            strips.append(self._line_to_rr(list(hw.geometry)))
        if not strips:
            return
        self._rr.log(
            "site/pathways",
            self._rr.LineStrips3D(
                strips,
                colors=[_COLOR_PATHWAY],
                radii=[_RADIUS_PATHWAY],
            ),
            static=static,
        )

    def _log_roi(self, event: SceneEvent) -> None:
        verts = event.payload.get("vertices", [])
        if self._frame is None and "centroid" in event.payload:
            from herald.osm.client import LatLon

            c = event.payload["centroid"]
            self._frame = LocalFrame(origin=LatLon(lat=c["lat"], lon=c["lon"]))
        self._log_contour("site/roi/contour", verts, _COLOR_ROI, static=False)

    def _log_zone(self, event: SceneEvent) -> None:
        level = event.payload.get("level")
        geom = event.payload.get("geometry_latlon")
        if not geom or not event.node_id:
            return
        ring = [(float(p[0]), float(p[1])) for p in geom]
        if level == "outdoor_region":
            self._log_contour(
                f"site/zones/{event.node_id}/contour",
                ring,
                _COLOR_OUTDOOR,
                static=False,
            )
        elif level == "building":
            parent = event.parent_id or "unassigned"
            height_m = float(event.payload.get("height_m", 10.0))
            node = SceneNode(
                id=event.node_id,
                level="building",
                zone_kind="building",
                text="",
                geometry_latlon=ring,
                height_m=height_m,
            )
            self._log_building(node, parent, static=False)

    def _log_containment(self, event: SceneEvent) -> None:
        if event.parent_id is None or event.node_id is None:
            return
        # Edges drawn in bulk during render_graph; skip per-event text spam.

    def _log_edge(
        self,
        parent: SceneNode,
        child: SceneNode,
        parent_id: str,
        child_id: str,
    ) -> None:
        if self._frame is None:
            return
        p = self._centroid_rr(parent.geometry_latlon, up=0.5)
        if child.level == "building":
            c = self._centroid_rr(child.geometry_latlon, up=child.height_m / 2)
        else:
            c = self._centroid_rr(child.geometry_latlon, up=0.5)
        vectors = [[c[0] - p[0], c[1] - p[1], c[2] - p[2]]]
        self._rr.log(
            f"graph/edges/{parent_id}-{child_id}",
            self._rr.Arrows3D(origins=[list(p)], vectors=vectors),
            static=True,
        )


def publish_event(viewer: RerunSceneViewer, event: SceneEvent) -> None:
    viewer.publish(event)
