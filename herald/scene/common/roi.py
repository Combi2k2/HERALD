"""Region-of-interest polygon helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pyproj import Geod
from shapely.geometry import Polygon, mapping, shape

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ROI:
    """Geographic region of interest as a shapely polygon (lon/lat internally)."""

    polygon: Polygon

    def __post_init__(self) -> None:
        if self.polygon.is_empty or not self.polygon.is_valid:
            raise ValueError("ROI polygon must be non-empty and valid")

    @classmethod
    def from_polygon(cls, poly: Sequence[tuple[float, float]]) -> ROI:
        """Build ROI from (lat, lon) vertices (ring need not be closed)."""
        if len(poly) < 3:
            raise ValueError("ROI needs at least 3 vertices")
        
        ring = list(poly)
        ring = ring + ([ring[0]] if ring[0] != ring[-1] else [])
        poly = Polygon([(lon, lat) for lat, lon in ring])

        return cls(polygon=poly)

    @classmethod
    def from_geojson(cls, path: Path | str) -> ROI:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        geom = data.get("geometry", data)
        poly = shape(geom)
        if poly.geom_type != "Polygon":
            raise ValueError(f"expected Polygon GeoJSON, got {poly.geom_type}")
        return cls(polygon=poly)

    def latlon_vertices(self) -> list[tuple[float, float]]:
        """Exterior ring as (lat, lon) tuples."""
        coords = list(self.polygon.exterior.coords)
        return [(lat, lon) for lon, lat in coords]

    def latlon_centroid(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return (c.y, c.x)

    def area(self) -> float:
        """Geodesic area in square meters."""
        lons, lats = self.polygon.exterior.coords.xy
        area, _ = _GEOD.polygon_area_perimeter(lons, lats)
        return abs(area)
    
    def bbox(self) -> tuple[float, float, float, float]:
        """Return south, west, north, east bounds."""
        minx, miny, maxx, maxy = self.polygon.bounds
        return (miny, minx, maxy, maxx)

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {
                "area": self.area(),
                "bbox": self.bbox(),
            },
            "geometry": mapping(self.polygon),
        }
