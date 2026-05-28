"""Region-of-interest polygon helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pyproj import Geod
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform

from herald.osm.client import OSMFeature

DEFAULT_MAX_AREA_M2 = 5_000_000.0  # 5 km²
_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ROI:
    """Geographic region of interest as a shapely polygon (lon/lat internally)."""

    polygon: Polygon
    max_area_m2: float = DEFAULT_MAX_AREA_M2

    def __post_init__(self) -> None:
        if self.polygon.is_empty or not self.polygon.is_valid:
            raise ValueError("ROI polygon must be non-empty and valid")
        area = self.area_m2()
        if area > self.max_area_m2:
            raise ValueError(
                f"ROI area {area / 1e6:.2f} km² exceeds limit "
                f"{self.max_area_m2 / 1e6:.2f} km²"
            )

    @classmethod
    def from_latlon_ring(
        cls,
        vertices: Sequence[tuple[float, float]],
        *,
        max_area_m2: float = DEFAULT_MAX_AREA_M2,
    ) -> ROI:
        """Build ROI from (lat, lon) vertices."""
        if len(vertices) < 3:
            raise ValueError("ROI needs at least 3 vertices")
        ring = list(vertices)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        # shapely uses (lon, lat)
        poly = Polygon([(lon, lat) for lat, lon in ring])
        return cls(polygon=poly, max_area_m2=max_area_m2)

    @classmethod
    def from_bbox(
        cls,
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        max_area_m2: float = DEFAULT_MAX_AREA_M2,
    ) -> ROI:
        return cls.from_latlon_ring(
            [(south, west), (south, east), (north, east), (north, west)],
            max_area_m2=max_area_m2,
        )

    @classmethod
    def from_geojson(cls, path: Path | str, *, max_area_m2: float = DEFAULT_MAX_AREA_M2) -> ROI:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        geom = data.get("geometry", data)
        poly = shape(geom)
        if poly.geom_type != "Polygon":
            raise ValueError(f"expected Polygon GeoJSON, got {poly.geom_type}")
        return cls(polygon=poly, max_area_m2=max_area_m2)

    def latlon_vertices(self) -> list[tuple[float, float]]:
        """Exterior ring as (lat, lon) tuples."""
        coords = list(self.polygon.exterior.coords)
        return [(lat, lon) for lon, lat in coords]

    def centroid_latlon(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return (c.y, c.x)

    def area_m2(self) -> float:
        """Geodesic area in square meters."""
        lons, lats = self.polygon.exterior.coords.xy
        area, _ = _GEOD.polygon_area_perimeter(lons, lats)
        return abs(area)

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {"area_m2": self.area_m2()},
            "geometry": mapping(self.polygon),
        }

    def clip_features(self, features: Sequence[OSMFeature]) -> list[OSMFeature]:
        """Keep features whose geometry intersects the ROI."""
        kept: list[OSMFeature] = []
        for feat in features:
            if feat.kind == "polygon":
                coords = [(lon, lat) for lat, lon in feat.geometry]
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    geom = Polygon(coords)
                else:
                    geom = Polygon(coords + [coords[0]])
            elif feat.kind == "linestring":
                from shapely.geometry import LineString

                geom = LineString([(lon, lat) for lat, lon in feat.geometry])
            else:
                from shapely.geometry import Point

                lat, lon = feat.geometry[0]
                geom = Point(lon, lat)
            if geom.intersects(self.polygon):
                kept.append(feat)
        return kept

    def to_utm_polygon(self) -> tuple[Any, Any]:
        """Project polygon to local UTM for metric operations.

        Returns (utm_polygon, transformer_back) where transformer_back maps
        UTM coords back to (lon, lat).
        """
        from pyproj import Transformer

        lat_c, lon_c = self.centroid_latlon()
        utm_zone = int((lon_c + 180) / 6) + 1
        hemisphere = "north" if lat_c >= 0 else "south"
        crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84"
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        def _fwd(x: float, y: float, z: float | None = None) -> tuple[float, float]:
            return to_utm.transform(x, y)

        utm_poly = transform(_fwd, self.polygon)
        return utm_poly, to_wgs
