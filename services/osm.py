"""OpenStreetMap queries via the Overpass API and location resolution."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import requests

GeometryKind = Literal["polygon", "linestring", "point"]
OSMElementType = Literal["node", "way", "relation"]
OSMPrimaryTag = str  # display grouping key, e.g. "building", "shop", "untagged"

DEFAULT_OVERPASS_URLS = (
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
CONNECT_TIMEOUT_S = 5.0
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RAW_POLYGON_TIMEOUT_S = 90.0
RAW_POLYGON_OVERPASS_TIMEOUT_S = 60
MAX_RETRIES_PER_URL = 2
RETRY_BACKOFF_S = (2.0, 5.0)
RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})
MAX_POLYGON_VERTICES = 500


@dataclass(frozen=True)
class LatLon:
    """WGS84 position in degrees."""

    lat: float
    lon: float

    def __iter__(self):
        yield self.lat
        yield self.lon


@dataclass(frozen=True)
class LocationConfig:
    """Where to read coordinates when not passed explicitly."""

    env_lat: str = "HERALD_LAT"
    env_lon: str = "HERALD_LON"


def lat_lon_from_env(config: LocationConfig | None = None) -> LatLon:
    """Read latitude and longitude from environment variables."""
    cfg = config or LocationConfig()
    lat_s = os.environ.get(cfg.env_lat)
    lon_s = os.environ.get(cfg.env_lon)
    if lat_s is None or lon_s is None:
        raise ValueError(
            f"Set {cfg.env_lat} and {cfg.env_lon} environment variables, "
            "or pass lat/lon explicitly."
        )
    return LatLon(lat=float(lat_s), lon=float(lon_s))


def resolve_location(
    lat: float | None = None,
    lon: float | None = None,
) -> LatLon:
    """Resolve (lat, lon) from explicit arguments or environment variables.

    Priority:
    1. Explicit ``lat`` and ``lon``
    2. ``HERALD_LAT`` / ``HERALD_LON`` environment variables
    """
    if lat is not None and lon is not None:
        return LatLon(lat=lat, lon=lon)
    return lat_lon_from_env()


@dataclass(frozen=True)
class OSMFeature:
    """A single OSM element with geometry and tags."""

    osm_id: int
    osm_type: OSMElementType
    tags: dict[str, str]
    geometry: tuple[tuple[float, float], ...]
    """Vertices as (lat, lon). Closed rings repeat the first point."""

    kind: GeometryKind
    category: Literal["building", "highway"]

    @property
    def name(self) -> str | None:
        return self.tags.get("name")

    def to_geojson_geometry(self) -> dict[str, Any]:
        """GeoJSON geometry (Polygon or LineString); coordinates are [lon, lat]."""
        coords = [[lon, lat] for lat, lon in self.geometry]
        if self.kind == "polygon":
            return {"type": "Polygon", "coordinates": [coords]}
        if self.kind == "linestring":
            return {"type": "LineString", "coordinates": coords}
        return {"type": "Point", "coordinates": coords[0]}

    def to_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "id": f"{self.osm_type}/{self.osm_id}",
            "properties": {**self.tags, "category": self.category},
            "geometry": self.to_geojson_geometry(),
        }


@dataclass
class OSMQueryResult:
    """Buildings and highways returned from a single Overpass query."""

    center: LatLon
    radius_m: float
    buildings: list[OSMFeature] = field(default_factory=list)
    highways: list[OSMFeature] = field(default_factory=list)


@dataclass(frozen=True)
class OSMRawPolygon:
    """Any OSM element with polygon geometry inside the query region."""

    osm_id: int
    osm_type: OSMElementType
    tags: dict[str, str]
    geometry: tuple[tuple[float, float], ...]
    primary_tag: OSMPrimaryTag
    primary_value: str

    @property
    def name(self) -> str | None:
        return self.tags.get("name")

    def to_geojson_feature(self) -> dict[str, Any]:
        coords = [[lon, lat] for lat, lon in self.geometry]
        return {
            "type": "Feature",
            "id": f"{self.osm_type}/{self.osm_id}",
            "properties": {
                **self.tags,
                "primary_tag": self.primary_tag,
                "primary_value": self.primary_value,
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }


@dataclass
class OSMRawPolygonResult:
    """Polygon geometries plus line highways from a broad Overpass scan."""

    center: LatLon
    polygons: list[OSMRawPolygon] = field(default_factory=list)
    highways: list[OSMFeature] = field(default_factory=list)

    def counts_by_tag(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for poly in self.polygons:
            counts[poly.primary_tag] = counts.get(poly.primary_tag, 0) + 1
        return counts

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "properties": {
                "center": {"lat": self.center.lat, "lon": self.center.lon},
                "counts": self.counts_by_tag(),
                "total": len(self.polygons),
            },
            "features": [p.to_geojson_feature() for p in self.polygons],
        }


def _validate_lat_lon(lat: float, lon: float) -> tuple[float, float]:
    lat_f, lon_f = float(lat), float(lon)
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        raise ValueError(f"non-finite coordinates: lat={lat!r}, lon={lon!r}")
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        raise ValueError(f"coordinates out of range: lat={lat_f}, lon={lon_f}")
    return lat_f, lon_f


def _validate_polygon_vertices(
    vertices_latlon: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(vertices_latlon) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    if len(vertices_latlon) > MAX_POLYGON_VERTICES:
        raise ValueError(f"polygon exceeds {MAX_POLYGON_VERTICES} vertices")
    validated: list[tuple[float, float]] = []
    for lat, lon in vertices_latlon:
        validated.append(_validate_lat_lon(lat, lon))
    return validated


def _poly_coords_from_vertices(vertices_latlon: Sequence[tuple[float, float]]) -> str:
    verts = _validate_polygon_vertices(vertices_latlon)
    ring = verts[:-1] if len(verts) > 1 and verts[0] == verts[-1] else verts
    return " ".join(f"{lat} {lon}" for lat, lon in ring)


def _build_overpass_raw_polygons_query(vertices_latlon: Sequence[tuple[float, float]]) -> str:
    """Fetch every way/relation intersecting the ROI (no tag filter).

    A single poly-bounded scan returns the complete set of features — including
    untagged polygons that still matter to the scene schema. This is also the
    *fastest* option: a per-tag union is slower because Overpass re-runs the
    point-in-polygon test once per tag clause. Polygon-vs-line geometry is
    selected client-side.
    """
    poly_coords = _poly_coords_from_vertices(vertices_latlon)
    return f"""
[out:json][timeout:{RAW_POLYGON_OVERPASS_TIMEOUT_S}];
(
  way(poly:"{poly_coords}");
  relation(poly:"{poly_coords}");
);
out tags geom;
""".strip()


def _build_overpass_polygon_query(vertices_latlon: Sequence[tuple[float, float]]) -> str:
    """Build Overpass QL with ``poly:"lat lon ..."`` filter."""
    poly_coords = _poly_coords_from_vertices(vertices_latlon)
    return f"""
[out:json][timeout:25];
(
  way["building"](poly:"{poly_coords}");
  relation["building"](poly:"{poly_coords}");
  way["highway"](poly:"{poly_coords}");
  relation["highway"](poly:"{poly_coords}");
);
out tags geom;
""".strip()


def _vertices_from_element(element: dict[str, Any]) -> list[tuple[float, float]]:
    geom = element.get("geometry")
    if not geom:
        return []
    return [(float(p["lat"]), float(p["lon"])) for p in geom]


def _ring_is_closed(vertices: Sequence[tuple[float, float]], tol: float = 1e-9) -> bool:
    if len(vertices) < 4:
        return False
    a, b = vertices[0], vertices[-1]
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _infer_geometry_kind(
    element: dict[str, Any], vertices: Sequence[tuple[float, float]]
) -> GeometryKind:
    tags = element.get("tags") or {}
    if tags.get("area") == "yes" or tags.get("area:highway"):
        return "polygon"
    if element.get("type") == "way" and _ring_is_closed(vertices):
        return "polygon"
    if len(vertices) == 1:
        return "point"
    return "linestring"


def _parse_element(element: dict[str, Any], category: Literal["building", "highway"]) -> OSMFeature | None:
    osm_type = element.get("type")
    if osm_type not in ("node", "way", "relation"):
        return None

    vertices = _vertices_from_element(element)
    if not vertices:
        return None

    kind = _infer_geometry_kind(element, vertices)
    tags = {str(k): str(v) for k, v in (element.get("tags") or {}).items()}

    return OSMFeature(
        osm_id=int(element["id"]),
        osm_type=osm_type,
        tags=tags,
        geometry=tuple(vertices),
        kind=kind,
        category=category,
    )


def _categorize_element(element: dict[str, Any]) -> Literal["building", "highway"] | None:
    tags = element.get("tags") or {}
    if "building" in tags:
        return "building"
    if "highway" in tags:
        return "highway"
    return None


def _primary_polygon_tag(tags: dict[str, str]) -> tuple[str, str]:
    """Pick a stable label for coloring; does not filter features."""
    if not tags:
        return "untagged", ""
    for key in sorted(tags):
        if key.startswith("_"):
            continue
        return key, tags[key]
    return "untagged", ""


def _parse_raw_polygon(element: dict[str, Any]) -> OSMRawPolygon | None:
    osm_type = element.get("type")
    if osm_type not in ("way", "relation"):
        return None
    vertices = _vertices_from_element(element)
    if not vertices:
        return None
    kind = _infer_geometry_kind(element, vertices)
    if kind != "polygon":
        return None
    tags = {str(k): str(v) for k, v in (element.get("tags") or {}).items()}
    primary_tag, primary_value = _primary_polygon_tag(tags)
    return OSMRawPolygon(
        osm_id=int(element["id"]),
        osm_type=osm_type,
        tags=tags,
        geometry=tuple(vertices),
        primary_tag=primary_tag,
        primary_value=primary_value,
    )


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Whether to retry the *same* endpoint.

    Only transient server-busy HTTP statuses are worth retrying in place.
    Timeouts and connection errors mean the endpoint is unhealthy, so the
    caller fails over to the next endpoint immediately instead of waiting
    through another full read timeout on a dead host.
    """
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in RETRYABLE_HTTP_STATUS
    return False


class OSMClient:
    """Client for Overpass API queries around a point."""

    def __init__(
        self,
        overpass_urls: Sequence[str] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session: requests.Session | None = None,
    ):
        self.overpass_urls = tuple(overpass_urls or DEFAULT_OVERPASS_URLS)
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "HERALD/0.1 (campus navigation research)",
                "Accept": "application/json",
            }
        )

    def _post_overpass(
        self,
        query: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """POST Overpass QL with per-endpoint retries on transient failures."""
        read_timeout = timeout_s if timeout_s is not None else self.timeout_s
        last_error: Exception | None = None
        for url in self.overpass_urls:
            for attempt in range(MAX_RETRIES_PER_URL):
                try:
                    response = self._session.post(
                        url,
                        data=query.encode("utf-8"),
                        headers={"Content-Type": "text/plain; charset=utf-8"},
                        timeout=(CONNECT_TIMEOUT_S, read_timeout),
                    )
                    response.raise_for_status()
                    return response.json()
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                    if attempt + 1 < MAX_RETRIES_PER_URL and _is_retryable_http_error(
                        exc
                    ):
                        time.sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
                        continue
                    break
        raise RuntimeError(
            f"All Overpass endpoints failed after retries "
            f"({', '.join(self.overpass_urls)})"
        ) from last_error

    def query_in_polygon(
        self,
        vertices_latlon: Sequence[tuple[float, float]],
    ) -> OSMQueryResult:
        """Fetch buildings + highways inside the polygon (lat, lon vertices).

        Auto-closes the ring if needed. Returns an ``OSMQueryResult`` whose
        ``center`` is the polygon centroid and ``radius_m`` is 0.0 — the
        polygon shape is the source of truth, not a circle.
        """
        verts = _validate_polygon_vertices(vertices_latlon)
        if verts[0] != verts[-1]:
            verts = verts + [verts[0]]
        query = _build_overpass_polygon_query(verts)
        payload = self._post_overpass(query)
        ring = verts[:-1] if verts[0] == verts[-1] else verts
        lat_c = sum(v[0] for v in ring) / len(ring)
        lon_c = sum(v[1] for v in ring) / len(ring)
        return self._parse_response(lat_c, lon_c, 0.0, payload)

    def query_raw_polygons_in_polygon(
        self,
        vertices_latlon: Sequence[tuple[float, float]],
    ) -> OSMRawPolygonResult:
        """Fetch all polygon geometries intersecting the ROI (unfiltered Overpass query)."""
        verts = _validate_polygon_vertices(vertices_latlon)
        if verts[0] != verts[-1]:
            verts = verts + [verts[0]]
        query = _build_overpass_raw_polygons_query(verts)
        payload = self._post_overpass(
            query,
            timeout_s=max(self.timeout_s, DEFAULT_RAW_POLYGON_TIMEOUT_S),
        )
        ring = verts[:-1] if verts[0] == verts[-1] else verts
        lat_c = sum(v[0] for v in ring) / len(ring)
        lon_c = sum(v[1] for v in ring) / len(ring)
        return self._parse_raw_polygon_response(lat_c, lon_c, payload)

    def _parse_raw_polygon_response(
        self,
        lat: float,
        lon: float,
        payload: dict[str, Any],
    ) -> OSMRawPolygonResult:
        result = OSMRawPolygonResult(center=LatLon(lat, lon))
        seen: set[tuple[str, int]] = set()
        for element in payload.get("elements", []):
            osm_type = element.get("type")
            osm_id = element.get("id")
            if osm_type not in ("way", "relation") or osm_id is None:
                continue
            key = (str(osm_type), int(osm_id))
            if key in seen:
                continue
            feature = _parse_raw_polygon(element)
            if feature is not None:
                seen.add(key)
                result.polygons.append(feature)
                continue
            if _categorize_element(element) == "highway":
                highway = _parse_element(element, "highway")
                if highway is not None and highway.kind == "linestring":
                    seen.add(key)
                    result.highways.append(highway)
        return result

    def _parse_response(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        payload: dict[str, Any],
    ) -> OSMQueryResult:
        result = OSMQueryResult(center=LatLon(lat, lon), radius_m=radius_m)
        for element in payload.get("elements", []):
            category = _categorize_element(element)
            if category is None:
                continue
            feature = _parse_element(element, category)
            if feature is None:
                continue
            if category == "building":
                result.buildings.append(feature)
            else:
                result.highways.append(feature)
        return result


__all__ = [
    "LatLon",
    "LocationConfig",
    "OSMClient",
    "OSMFeature",
    "OSMQueryResult",
    "OSMRawPolygon",
    "OSMRawPolygonResult",
    "lat_lon_from_env",
    "resolve_location",
]
