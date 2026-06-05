"""Resolve a WGS84 position for OSM queries."""

from __future__ import annotations

import os
from dataclasses import dataclass

from services.osm.client import LatLon


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
