"""Vector primitives, site coordinate frame, and scene geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pyproj import Transformer

from herald.data.paths import FRAME_JSON

GeometryType = Literal["point", "polyline", "polygon"]
GeometryFrame = Literal["ENU", "WGS", "UTM"]


def _utm_crs(lat: float, lon: float) -> str:
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    return f"+proj=utm +zone={zone} +{hemisphere} +ellps=WGS84"


@dataclass(frozen=True)
class Frame:
    """Site coordinate reference: WGS84 origin, ENU anchor, and UTM zone."""

    lat: float
    lon: float
    east: float = 0.0
    north: float = 0.0

    def __post_init__(self) -> None:
        aeqd = (
            "+proj=aeqd"
            f" +lat_0={self.lat}"
            f" +lon_0={self.lon}"
            " +ellps=WGS84 +units=m +no_defs"
        )
        utm = _utm_crs(self.lat, self.lon)
        object.__setattr__(self, "_wgs2enu", Transformer.from_crs("EPSG:4326", aeqd, always_xy=True))
        object.__setattr__(self, "_enu2wgs", Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True),)
        object.__setattr__(self, "_wgs2utm", Transformer.from_crs("EPSG:4326", utm, always_xy=True),)
        object.__setattr__(self, "_utm2wgs", Transformer.from_crs(utm, "EPSG:4326", always_xy=True),)

    @classmethod
    def from_origin(cls, lat: float, lon: float) -> Frame:
        return cls(lat=lat, lon=lon, east=0.0, north=0.0)

    def wgs2enu(self, lat: float, lon: float) -> tuple[float, float]:
        east, north = self._wgs2enu.transform(lon, lat)
        return (east, north)

    def enu2wgs(self, east: float, north: float) -> tuple[float, float]:
        lon, lat = self._enu2wgs.transform(east, north)
        return (lat, lon)

    def wgs2utm(self, lat: float, lon: float) -> tuple[float, float]:
        easting, northing = self._wgs2utm.transform(lon, lat)
        return (easting, northing)

    def utm2wgs(self, easting: float, northing: float) -> tuple[float, float]:
        lon, lat = self._utm2wgs.transform(easting, northing)
        return (lat, lon)

    def utm2enu(self, easting: float, northing: float) -> tuple[float, float]:
        lat, lon = self.utm2wgs(easting, northing)
        return self.wgs2enu(lat, lon)

    def enu2utm(self, east: float, north: float) -> tuple[float, float]:
        lat, lon = self.enu2wgs(east, north)
        return self.wgs2utm(lat, lon)

    def to_dict(self) -> dict[str, float]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "east": self.east,
            "north": self.north,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Frame:
        return cls(
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            east=float(data.get("east", 0.0)),
            north=float(data.get("north", 0.0)),
        )

    @classmethod
    def load(cls, path: Path = FRAME_JSON) -> Frame | None:
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Path = FRAME_JSON) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class Vector(np.ndarray):
    """Base class for vector primitives."""
    _dim: int = 3

    @property
    def dim(self) -> int:
        return self._dim

    def __new__(cls, values: Sequence[float] | np.ndarray) -> Vector:
        values = values.tolist() if isinstance(values, np.ndarray) else list(values)
        values = values + [0.0] * max(0, cls._dim - len(values))
        return np.array(values[:cls._dim], dtype=np.float64).view(cls)

    @classmethod
    def zeros(cls) -> Vector:
        return cls(np.zeros(cls._dim, dtype=np.float64))


class Vec2(Vector):
    """1D float64 array with shape ``(2,)``."""
    _dim = 2


class Vec3(Vector):
    """1D float64 array with shape ``(3,)``."""
    _dim = 3


class Vec4(Vector):
    """1D float64 array with shape ``(4,)``."""
    _dim = 4


@dataclass
class Geometry:
    type: GeometryType
    frame: GeometryFrame = "ENU"
    coords: list[Vec3] = field(default_factory=lambda: [Vec3.zeros()])
    offset: Vec3 = field(default_factory=Vec3.zeros)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64)
        if self.coords.ndim == 1:
            self.coords = self.coords.reshape(1, 3)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        self.offset = Vec3(self.offset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "frame": self.frame,
            "coords": self.coords.tolist(),
            "offset": self.offset.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Geometry:
        geom_type = data.get("type")
        geom_frame = data.get("frame")

        coords = [
            [
                float(p[0]),
                float(p[1]),
                float(p[2] if len(p) == 3 else 0.0),
            ]
            for p in data.get("coords", [[0.0, 0.0, 0.0]])
        ]

        return cls(
            type=geom_type,
            frame=geom_frame,
            coords=np.array(coords, dtype=np.float64),
            offset=Vec3(data.get("offset", [0.0, 0.0, 0.0])),
        )
