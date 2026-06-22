"""Rerun-based 3D scene graph viewer.

Local ENU is mapped into Rerun as RIGHT_HAND_Z_UP:
  X = East, Y = North, Z = Up — ground lies in the XY plane at Z=0.

Ground (Z=0): OSM footprint outlines (``osm/raw/*``).
Upper layers: ancestor contours only, raised by containment depth.
"""

from __future__ import annotations

from typing import Any

from services.osm.client import OSMFeature, OSMRawPolygon
from herald.scene.common.geometry import Frame
import numpy as np
from herald.scene.common.geometry import Geometry
from herald.scene.common.graph import SceneGraph, SceneNode
from herald.scene.common.nav import NavGraph
from herald.viewer.hierarchy_layout import (
    GROUND_Z,
    LAYER_THICKNESS,
    SITE_ID,
    ancestor_layer_z,
    children_map_from_pairs,
    depth_from_site,
    is_containment_leaf,
    max_distance_to_leaf,
)
from herald.viewer.mesh import layer_z_with_epsilon
from herald.viewer.style import (
    CATEGORY_COLORS,
    COLOR_PATHWAY,
    color_for_node,
    metadata_lines,
)

_RAW_OSM_COLORS: dict[str, list[int]] = {
    "building": [96, 165, 250],
    "building:part": [96, 165, 250],
    "landuse": [74, 222, 128],
    "leisure": [192, 132, 252],
    "amenity": [244, 114, 182],
    "natural": [52, 211, 153],
    "water": [34, 211, 238],
    "waterway": [6, 182, 212],
    "highway": [251, 146, 60],
    "place": [250, 204, 21],
    "man_made": [148, 163, 184],
    "barrier": [239, 68, 68],
    "untagged": [161, 161, 170],
    "other": [161, 161, 170],
}
_RAW_DEFAULT_COLOR = [161, 161, 170]

# Screen-space stroke widths (pixels at 100% UI scale) — not meter tubes.
_LINE_WIDTH_RAW_PX = 1.0
_LINE_WIDTH_CONTOUR_PX = 1.5
_LINE_WIDTH_PATHWAY_PX = 1.25

# Nav graph: nodes are volumetric spheres sized in scene metres (not pixels).
_NAV_NODE_RADIUS = 2.0
_NAV_NODE_Z = GROUND_Z
_NAV_EDGE_Z = GROUND_Z + 0.1
_COLOR_NAV_NODE = [250, 204, 21]
_COLOR_NAV_EDGE = [234, 179, 8]

_DISPLAY_TAG_ORDER = (
    "building",
    "building:part",
    "landuse",
    "leisure",
    "amenity",
    "natural",
    "water",
    "waterway",
    "highway",
    "place",
    "man_made",
    "barrier",
    "tourism",
    "sport",
)


def _display_tag(tags: dict[str, str]) -> str:
    """Semantic OSM tag for coloring — not alphabetical ``primary_tag``."""
    for key in _DISPLAY_TAG_ORDER:
        if key in tags:
            return key
    if not tags:
        return "untagged"
    return "other"


def _require_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "rerun-sdk is required for the viewer. Install with: uv sync --group viewer"
        ) from exc
    return rr


def _closed_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ring:
        return ring
    if ring[0] == ring[-1]:
        return list(ring)
    return [*ring, ring[0]]


def _closed_coords(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords
    if np.allclose(coords[0], coords[-1]):
        return coords
    return np.vstack([coords, coords[0:1]])


def _enu_to_rr(east: float, north: float, up: float = 0.0) -> list[float]:
    return [east, north, up]


class RerunSceneViewer:
    """Stream scene graph construction events to Rerun."""

    def __init__(
        self,
        *,
        app_id: str = "herald-scene",
        spawn: bool = True,
    ) -> None:
        rr = _require_rerun()
        rr.init(app_id, spawn=spawn)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._rr = rr
        self._frame: Frame | None = None
        self._node_store: dict[str, SceneNode] = {}
        self._containment_edges: list[tuple[str, str]] = []
        self._layout_cache: dict[str, int] = {}
        self._site_id: str | None = None
        self._scene_span = 500.0
        self._send_blueprint()

    def _send_blueprint(self) -> None:
        try:
            import rerun.blueprint as rrb
        except (ImportError, ModuleNotFoundError, AttributeError):
            return

        span = self._scene_span
        eye_dist = max(1200.0, span * 1.6)
        self._rr.send_blueprint(
            rrb.Blueprint(
                rrb.Spatial3DView(
                    origin="/",
                    name="HERALD Scene",
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital,
                        position=(eye_dist, -eye_dist, eye_dist * 0.55),
                        look_target=(0.0, 0.0, span * 0.15),
                    ),
                ),
                collapse_panels=True,
            )
        )

    def _update_scene_span_coords(self, coords: np.ndarray) -> None:
        for row in coords:
            self._scene_span = max(
                self._scene_span, abs(float(row[0])), abs(float(row[1])), 1.0
            )

    def _update_scene_span_latlon(self, ring: list[tuple[float, float]]) -> None:
        if self._frame is None or not ring:
            return
        for lat, lon in ring:
            east, north = self._frame.wgs2enu(lat, lon)
            self._scene_span = max(
                self._scene_span, abs(east), abs(north), 1.0
            )

    def _line_width(self, px: float):
        return self._rr.Radius.ui_points(px)

    def _coords_to_rr(self, coords: np.ndarray, *, z: float) -> list[list[float]]:
        return [_enu_to_rr(float(row[0]), float(row[1]), z) for row in coords]

    def _ring_to_rr(
        self, ring: list[tuple[float, float]], *, z: float
    ) -> list[list[float]]:
        if self._frame is None:
            return [[0.0, 0.0, z]]
        return [
            _enu_to_rr(*self._frame.wgs2enu(lat, lon), z)
            for lat, lon in ring
        ]

    def _log_contour(
        self,
        path: str,
        coords: np.ndarray,
        color: list[int],
        *,
        z: float,
        width_px: float = _LINE_WIDTH_CONTOUR_PX,
        static: bool = True,
    ) -> None:
        if self._frame is None:
            return
        z = layer_z_with_epsilon(z, path)
        strip = self._coords_to_rr(_closed_coords(coords), z=z)
        if len(strip) < 3:
            return
        rgba = [*color[:3], 255] if len(color) == 3 else list(color)
        self._rr.log(
            path,
            self._rr.LineStrips3D(
                [strip],
                colors=[rgba],
                radii=[self._line_width(width_px)],
            ),
            static=static,
        )

    def _geom_coords_enu(self, geom: Geometry) -> np.ndarray:
        if geom.frame == "ENU":
            return geom.coords
        if self._frame is None:
            return geom.coords
        rows = [
            [*self._frame.wgs2enu(float(row[0]), float(row[1])), float(row[2])]
            for row in geom.coords
        ]
        return np.array(rows, dtype=np.float64)

    def set_frame(self, frame: Frame) -> None:
        self._frame = frame
        self._scene_span = 500.0
        self._send_blueprint()
        self._rr.log(
            "pipeline/status",
            self._rr.TextLog(
                "Ground map: osm/raw/* outlines. Upper layers: ancestor contours only "
                f"({LAYER_THICKNESS:.0f} m × max depth to leaf). Select entities for attrs."
            ),
            static=True,
        )

    def log_pathways(
        self, pathways: list[OSMFeature], *, static: bool = True
    ) -> None:
        self._log_pathways(pathways, static=static)

    def render_nav_graph(self, nav: NavGraph, *, static: bool = True) -> None:
        """Render nav-graph nodes as volumetric spheres and edges as lines.

        Node positions are ENU metres; ``_NAV_NODE_RADIUS`` is in scene metres
        so each node renders as a real 3D sphere rather than a screen-space dot.
        """
        if not nav.nodes:
            return

        positions = []
        for node in nav.nodes:
            east, north = node.pos
            self._scene_span = max(self._scene_span, abs(east), abs(north), 1.0)
            positions.append(_enu_to_rr(east, north, _NAV_NODE_Z))
        self._rr.log(
            "nav/nodes",
            self._rr.Points3D(
                positions,
                colors=[[*_COLOR_NAV_NODE, 255]],
                radii=_NAV_NODE_RADIUS,
            ),
            static=static,
        )

        pos_by_id = {node.id: node.pos for node in nav.nodes}
        strips = []
        for edge in nav.edges:
            a, b = pos_by_id.get(edge.source_id), pos_by_id.get(edge.target_id)
            if a is None or b is None:
                continue
            strips.append([_enu_to_rr(*a, _NAV_EDGE_Z), _enu_to_rr(*b, _NAV_EDGE_Z)])
        if strips:
            self._rr.log(
                "nav/edges",
                self._rr.LineStrips3D(
                    strips,
                    colors=[[*_COLOR_NAV_EDGE, 255]],
                    radii=[self._line_width(_LINE_WIDTH_PATHWAY_PX)],
                ),
                static=static,
            )
        self._send_blueprint()

    def render_graph(
        self,
        graph: SceneGraph,
        *,
        frame: Frame,
        pathways: list[OSMFeature] | None = None,
    ) -> None:
        """Render a complete saved graph (offline replay)."""
        self.set_frame(frame)
        self._node_store.clear()
        self._containment_edges.clear()
        self._layout_cache.clear()
        site = graph.site_node()
        self._site_id = site.id if site is not None else None

        for edge in graph.edges:
            if edge.edge_type == "contains":
                self._containment_edges.append((edge.source_id, edge.target_id))

        for node in graph.nodes:
            self._node_store[node.id] = node

        self._refresh_hierarchy_layers(static=True)

        if pathways:
            self._log_pathways(pathways, static=True)

        self._log_legend(static=True)

    def render_raw_osm_polygons(
        self,
        polygons: list[OSMRawPolygon],
        *,
        static: bool = True,
    ) -> None:
        if self._frame is None or not polygons:
            return
        for poly in polygons:
            self._update_scene_span_latlon(list(poly.geometry))

        by_tag: dict[str, list[OSMRawPolygon]] = {}
        for poly in polygons:
            tag = _display_tag(dict(poly.tags))
            by_tag.setdefault(tag, []).append(poly)

        for tag, group in sorted(by_tag.items()):
            color = _RAW_OSM_COLORS.get(tag, _RAW_DEFAULT_COLOR)
            strips: list[list[list[float]]] = []
            colors: list[list[int]] = []
            for poly in group:
                z = layer_z_with_epsilon(GROUND_Z, poly.osm_id)
                strip = self._ring_to_rr(_closed_ring(list(poly.geometry)), z=z)
                if len(strip) < 3:
                    continue
                strips.append(strip)
                colors.append([*color[:3], 255])
            if not strips:
                continue
            self._rr.log(
                f"osm/raw/{tag}",
                self._rr.LineStrips3D(
                    strips,
                    colors=colors,
                    radii=[self._line_width(_LINE_WIDTH_RAW_PX)],
                ),
                static=static,
            )
        self._send_blueprint()

    def _children_map(self) -> dict[str, list[str]]:
        return children_map_from_pairs(self._containment_edges)

    def _ancestors_of(self, node_id: str, children_map: dict[str, list[str]]) -> set[str]:
        ancestors: set[str] = set()
        parent_id = self._parent_of_node_in(node_id, children_map)
        while parent_id is not None:
            ancestors.add(parent_id)
            parent_id = self._parent_of_node_in(parent_id, children_map)
        return ancestors

    def _parent_of_node_in(
        self, node_id: str, children_map: dict[str, list[str]]
    ) -> str | None:
        for parent_id, child_ids in children_map.items():
            if node_id in child_ids:
                return parent_id
        return None

    def _log_node_metadata(
        self,
        node: SceneNode,
        *,
        base: str,
        children_map: dict[str, list[str]],
        static: bool,
        layer_z: float | None = None,
    ) -> None:
        site_id = self._site_id or SITE_ID
        tree_depth = depth_from_site(
            node.id, children_map, site_id=site_id, cache=self._layout_cache
        )
        dist = max_distance_to_leaf(
            node.id, children_map, cache=self._layout_cache
        )
        meta_lines = metadata_lines(node)
        meta_lines.append(f"tree_depth_from_site: {tree_depth}")
        meta_lines.append(f"tree_depth_to_leaf: {dist}")
        if layer_z is not None:
            meta_lines.append(f"layer_z: {layer_z:.1f}")
        self._rr.log(
            f"{base}/metadata",
            self._rr.TextLog("\n".join(meta_lines)),
            static=static,
        )
        extra: dict[str, Any] = {
            "tree_depth_to_leaf": dist,
            "tree_depth_from_site": tree_depth,
        }
        if layer_z is not None:
            extra["layer_z"] = layer_z
        self._log_any_values(f"{base}/attrs", node, static=static, **extra)

    def _render_ancestor_layer(
        self,
        node: SceneNode,
        *,
        children_map: dict[str, list[str]],
        static: bool,
    ) -> None:
        if self._frame is None or node.type == "site":
            return

        site_id = self._site_id or SITE_ID
        layer_z = ancestor_layer_z(
            node.id, children_map, site_id=site_id, cache=self._layout_cache
        )
        base = f"site/layers/{node.id}"
        self._update_scene_span_coords(self._geom_coords_enu(node.geom))

        if layer_z is not None:
            color = color_for_node(node)
            self._log_contour(
                f"{base}/footprint",
                self._geom_coords_enu(node.geom),
                color,
                z=layer_z,
                width_px=_LINE_WIDTH_CONTOUR_PX,
                static=static,
            )

        self._log_node_metadata(
            node,
            base=base,
            children_map=children_map,
            static=static,
            layer_z=layer_z,
        )

    def _render_leaf_metadata(
        self,
        node: SceneNode,
        *,
        children_map: dict[str, list[str]],
        static: bool,
    ) -> None:
        base = f"site/nodes/{node.id}"
        self._log_node_metadata(
            node,
            base=base,
            children_map=children_map,
            static=static,
            layer_z=GROUND_Z,
        )

    def _render_nodes(
        self, node_ids: set[str], *, children_map: dict[str, list[str]], static: bool
    ) -> None:
        for node_id in sorted(node_ids):
            node = self._node_store.get(node_id)
            if node is None:
                continue
            if node.type == "site":
                continue
            if is_containment_leaf(node_id, children_map):
                self._render_leaf_metadata(
                    node, children_map=children_map, static=static
                )
            else:
                self._render_ancestor_layer(
                    node, children_map=children_map, static=static
                )

    def _refresh_hierarchy_layers(self, *, static: bool) -> None:
        if not self._node_store:
            return
        children_map = self._children_map()
        self._layout_cache.clear()
        self._render_nodes(set(self._node_store), children_map=children_map, static=static)

    def _line_to_rr(
        self, coords: list[tuple[float, float]], *, up: float = GROUND_Z + 0.05
    ) -> list[list[float]]:
        if self._frame is None:
            return [[0.0, 0.0, up]]
        return [
            _enu_to_rr(*self._frame.wgs2enu(lat, lon), up)
            for lat, lon in coords
        ]

    def _log_any_values(
        self, path: str, node: SceneNode, *, static: bool, **extra: Any
    ) -> None:
        osm_ref = next(
            (ref.assigned_id for ref in node.refs if ref.assigned_by == "osm" and ref.assigned_id),
            None,
        )
        values: dict[str, Any] = {
            "id": node.id,
            "type": node.type,
            "role": node.role or "",
            "category": node.category or "",
            "function": node.function or "",
            **extra,
        }
        if osm_ref:
            values["osm_ref"] = osm_ref
        if node.name:
            values["name"] = node.name
        if node.desc:
            values["description"] = node.desc
        self._rr.log(path, self._rr.AnyValues(**values), static=static)

    def _log_legend(self, *, static: bool = True) -> None:
        lines = [
            "Ground (Z=0): OSM outline map under osm/raw/*.",
            f"Upper layers: ancestor contours only, +{LAYER_THICKNESS:.0f} m per tree level.",
            "Select site/nodes/*/attrs or site/layers/*/attrs to inspect.",
            "",
        ]
        for category, rgba in sorted(CATEGORY_COLORS.items()):
            lines.append(f"  {category}: rgb({rgba[0]}, {rgba[1]}, {rgba[2]})")
        self._rr.log(
            "legend/categories",
            self._rr.TextLog("\n".join(lines)),
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
            "osm/pathways",
            self._rr.LineStrips3D(
                strips,
                colors=[[*COLOR_PATHWAY[:3], 255]],
                radii=[self._line_width(_LINE_WIDTH_PATHWAY_PX)],
            ),
            static=static,
        )
