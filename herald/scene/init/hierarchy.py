"""Build a containment forest from OSM polygon footprints."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from herald.scene.common.roi import ROI
from services.osm.client import OSMRawPolygon
from utils.osm_filter import polygon_disposition

SITE_OSM_ID = 0

MIN_NODE_AREA_M2 = 30.0
MAX_NODE_AREA_M2 = 20_000.0
CONTAINMENT_RATIO = 0.90


def graph_node_id(osm_id: int) -> str:
    """Map hierarchy ``osm_id`` to scene-graph node id (site stays ``site_000``)."""
    return "site_000" if osm_id == SITE_OSM_ID else f"node_{osm_id}"


@dataclass(frozen=True)
class HierarchyNode:
    osm_id: int
    osm_type: str
    tags: dict[str, str]
    geom: list[tuple[float, float]]
    area: float
    parent_osm_id: int | None = None
    child_osm_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ContainmentForest:
    nodes: dict[int, HierarchyNode]
    others: tuple[int, ...]

    @property
    def site_children(self) -> tuple[int, ...]:
        site = self.nodes.get(SITE_OSM_ID)
        if site is None:
            return ()
        return site.child_osm_ids


def _utm_transformer(roi: ROI) -> Transformer:
    lat_c, lon_c = roi.centroid_latlon()
    utm_zone = int((lon_c + 180) / 6) + 1
    hemisphere = "north" if lat_c >= 0 else "south"
    crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _ring_to_polygon(
    ring: list[tuple[float, float]], transformer: Transformer
) -> Polygon | None:
    if len(ring) < 3:
        return None
    closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
    try:
        coords = [transformer.transform(lon, lat) for lat, lon in closed]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def _containment_ratio(inner: Polygon, outer: Polygon) -> float:
    if inner.is_empty or inner.area <= 0:
        return 0.0
    return inner.intersection(outer).area / inner.area


def _closed_roi_ring(roi: ROI) -> list[tuple[float, float]]:
    ring = list(roi.latlon_vertices())
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def build_containment_forest(
    polygons: list[OSMRawPolygon],
    roi: ROI,
    *,
    min_area_m2: float = MIN_NODE_AREA_M2,
    max_area_m2: float = MAX_NODE_AREA_M2,
    containment_ratio: float = CONTAINMENT_RATIO,
) -> ContainmentForest:
    """Assign parent/child links using real containment with area thresholds."""
    to_utm = _utm_transformer(roi)

    records: list[tuple[OSMRawPolygon, Polygon, float]] = []
    others: list[int] = []

    for poly in polygons:
        if poly.osm_id == SITE_OSM_ID:
            others.append(poly.osm_id)
            continue
        ring = list(poly.geometry)
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        utm_poly = _ring_to_polygon(ring, to_utm)
        if utm_poly is None:
            others.append(poly.osm_id)
            continue
        area_m2 = float(utm_poly.area)
        disposition = polygon_disposition(
            poly.tags,
            area_m2=area_m2,
            min_area_m2=min_area_m2,
            max_area_m2=max_area_m2,
        )
        if disposition != "hierarchy":
            others.append(poly.osm_id)
            continue
        records.append((poly, utm_poly, area_m2))

    records.sort(key=lambda item: item[2])

    parent: dict[int, int | None] = {poly.osm_id: None for poly, _, _ in records}
    children: dict[int, list[int]] = {poly.osm_id: [] for poly, _, _ in records}

    geoms = [utm_poly for _, utm_poly, _ in records]
    tree = STRtree(geoms) if geoms else None

    for idx, (osm_poly, utm_poly, area) in enumerate(records):
        best_parent: int | None = None
        best_parent_area = float("inf")
        if tree is None:
            continue
        for j in tree.query(utm_poly, predicate="intersects"):
            if j == idx:
                continue
            parent_poly, parent_utm, parent_area = records[j]
            if parent_area <= area:
                continue
            ratio = _containment_ratio(utm_poly, parent_utm)
            if ratio >= containment_ratio and parent_area < best_parent_area:
                best_parent = parent_poly.osm_id
                best_parent_area = parent_area
        if best_parent is not None:
            parent[osm_poly.osm_id] = best_parent
            children[best_parent].append(osm_poly.osm_id)

    nodes: dict[int, HierarchyNode] = {}

    for poly, _, area_m2 in records:
        parent_id = parent[poly.osm_id] or SITE_OSM_ID
        ring = list(poly.geometry)
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        nodes[poly.osm_id] = HierarchyNode(
            osm_id=poly.osm_id,
            osm_type=poly.osm_type,
            tags=dict(poly.tags),
            geom=ring,
            area=area_m2,
            parent_osm_id=parent_id,
            child_osm_ids=tuple(sorted(children[poly.osm_id])),
        )

    site_child_ids = tuple(
        sorted(oid for oid, node in nodes.items() if node.parent_osm_id == SITE_OSM_ID)
    )
    nodes[SITE_OSM_ID] = HierarchyNode(
        osm_id=SITE_OSM_ID,
        osm_type="site",
        tags={},
        geom=_closed_roi_ring(roi),
        area=roi.area_m2(),
        parent_osm_id=None,
        child_osm_ids=site_child_ids,
    )

    return ContainmentForest(
        nodes=nodes,
        others=tuple(sorted(set(others))),
    )
