"""Local ENU coordinate frame centered on a site origin."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer

from services.osm.client import LatLon


@dataclass(frozen=True)
class LocalFrame:
    """Azimuthal equidistant projection centered on ``origin``."""

    origin: LatLon

    def __post_init__(self) -> None:
        aeqd = (
            "+proj=aeqd +lat_0={lat} +lon_0={lon} +ellps=WGS84 +units=m +no_defs"
        ).format(lat=self.origin.lat, lon=self.origin.lon)
        to_local = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
        to_wgs = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
        object.__setattr__(self, "_to_local", to_local)
        object.__setattr__(self, "_to_wgs", to_wgs)

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert WGS84 (lat, lon) to local ENU (east, north) meters."""
        east, north = self._to_local.transform(lon, lat)
        return (east, north)

    def to_wgs84(self, east: float, north: float) -> tuple[float, float]:
        """Convert local ENU (east, north) to WGS84 (lat, lon)."""
        lon, lat = self._to_wgs.transform(east, north)
        return (lat, lon)

    def ring_to_local(
        self, vertices_latlon: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        return [self.to_local(lat, lon) for lat, lon in vertices_latlon]

    def to_dict(self) -> dict[str, float]:
        return {"lat": self.origin.lat, "lon": self.origin.lon}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> LocalFrame:
        return cls(origin=LatLon(lat=data["lat"], lon=data["lon"]))
