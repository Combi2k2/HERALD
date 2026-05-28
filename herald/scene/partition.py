"""Partition ROI free space into outdoor zones using walkable pathways."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge, unary_union

from herald.osm.client import OSMFeature
from herald.scene.roi import ROI

WALKABLE_HIGHWAY_TYPES = frozenset(
    {"footway", "path", "pedestrian", "steps", "cycleway", "living_street"}
)
DEFAULT_PATHWAY_BUFFER_M = 3.0
MIN_OUTDOOR_AREA_M2 = 50.0


@dataclass(frozen=True)
class BuildingCandidate:
    osm_id: int
    name: str | None
    footprint_latlon: list[tuple[float, float]]
    tags: dict[str, str]
    height_m: float
    height_source: str


@dataclass(frozen=True)
class OutdoorZone:
    zone_id: str
    polygon_latlon: list[tuple[float, float]]
    area_m2: float


@dataclass(frozen=True)
class PartitionResult:
    outdoor_zones: list[OutdoorZone]
    buildings: list[BuildingCandidate]
    building_to_outdoor: dict[int, str]
    pathway_features: list[OSMFeature]


def _utm_transformer(roi: ROI) -> Transformer:
    lat_c, lon_c = roi.centroid_latlon()
    utm_zone = int((lon_c + 180) / 6) + 1
    hemisphere = "north" if lat_c >= 0 else "south"
    crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _latlon_to_utm(
    ring: list[tuple[float, float]], transformer: Transformer
) -> Polygon:
    coords = [transformer.transform(lon, lat) for lat, lon in ring]
    return Polygon(coords)


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


def _feature_to_polygon_latlon(feat: OSMFeature) -> list[tuple[float, float]] | None:
    if feat.kind != "polygon":
        return None
    ring = list(feat.geometry)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def filter_walkable_highways(highways: list[OSMFeature]) -> list[OSMFeature]:
    return [
        h
        for h in highways
        if h.kind == "linestring" and h.tags.get("highway") in WALKABLE_HIGHWAY_TYPES
    ]


def extract_buildings(buildings: list[OSMFeature]) -> list[BuildingCandidate]:
    candidates: list[BuildingCandidate] = []
    for b in buildings:
        ring = _feature_to_polygon_latlon(b)
        if ring is None:
            continue
        height_m, height_source = _parse_height(b.tags)
        candidates.append(
            BuildingCandidate(
                osm_id=b.osm_id,
                name=b.name,
                footprint_latlon=ring,
                tags=b.tags,
                height_m=height_m,
                height_source=height_source,
            )
        )
    return candidates


def _utm_to_latlon(
    x: float, y: float, to_wgs: Transformer
) -> tuple[float, float]:
    lon, lat = to_wgs.transform(x, y)
    return (lat, lon)


def _polygon_utm_to_latlon(poly: Polygon, to_wgs: Transformer) -> list[tuple[float, float]]:
    return [_utm_to_latlon(x, y, to_wgs) for x, y in poly.exterior.coords]


def partition_outdoor_zones(
    roi: ROI,
    buildings: list[BuildingCandidate],
    walkable_highways: list[OSMFeature],
    *,
    pathway_buffer_m: float = DEFAULT_PATHWAY_BUFFER_M,
) -> PartitionResult:
    """Split ROI free space into outdoor zones separated by buffered pathways."""
    utm_roi, to_wgs = roi.to_utm_polygon()
    to_utm = _utm_transformer(roi)

    building_polys = [
        _latlon_to_utm(b.footprint_latlon, to_utm) for b in buildings
    ]
    building_union = unary_union(building_polys) if building_polys else Polygon()
    free_space = utm_roi.difference(building_union)

    lines: list[LineString] = []
    for hw in walkable_highways:
        if hw.kind != "linestring":
            continue
        utm_coords = [to_utm.transform(lon, lat) for lat, lon in hw.geometry]
        if len(utm_coords) >= 2:
            lines.append(LineString(utm_coords))

    if lines:
        merged = linemerge(MultiLineString(lines))
        if merged.is_empty:
            barrier = Polygon()
        elif merged.geom_type == "LineString":
            barrier = merged.buffer(pathway_buffer_m)
        else:
            barrier = unary_union([g.buffer(pathway_buffer_m) for g in merged.geoms])
        if not barrier.is_empty:
            free_space = free_space.difference(barrier)

    if free_space.is_empty:
        polys: list[Polygon] = []
    elif free_space.geom_type == "Polygon":
        polys = [free_space]
    elif free_space.geom_type == "MultiPolygon":
        polys = list(free_space.geoms)
    else:
        polys = []

    polys = [p for p in polys if p.area >= MIN_OUTDOOR_AREA_M2]

    outdoor_zones: list[OutdoorZone] = []
    for i, poly in enumerate(polys):
        outdoor_zones.append(
            OutdoorZone(
                zone_id=f"outdoor_{i:03d}",
                polygon_latlon=_polygon_utm_to_latlon(poly, to_wgs),
                area_m2=poly.area,
            )
        )

    building_to_outdoor: dict[int, str] = {}
    for building, bpoly in zip(buildings, building_polys):
        best_zone: str | None = None
        best_overlap = 0.0
        for opoly, zone in zip(polys, outdoor_zones):
            overlap = bpoly.intersection(opoly).area
            if overlap > best_overlap:
                best_overlap = overlap
                best_zone = zone.zone_id
        if best_zone is None and outdoor_zones:
            c = bpoly.centroid
            for opoly, zone in zip(polys, outdoor_zones):
                if opoly.contains(c):
                    best_zone = zone.zone_id
                    break
        if best_zone is None and outdoor_zones:
            best_zone = outdoor_zones[0].zone_id
        if best_zone is not None:
            building_to_outdoor[building.osm_id] = best_zone

    return PartitionResult(
        outdoor_zones=outdoor_zones,
        buildings=buildings,
        building_to_outdoor=building_to_outdoor,
        pathway_features=walkable_highways,
    )
