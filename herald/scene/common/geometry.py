"""Vector primitives, site coordinate frame, scene geometry, and camera frame geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pyproj import Transformer

from herald.data.paths import FRAME_JSON

GeometryType = Literal["point", "polyline", "polygon", "obb"]
GeometryFrame = Literal["ENU", "WGS", "UTM", "NED"]


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
    offset: Vec3 = field(default_factory=Vec3.zeros)     # obb: box centre
    half_size: Vec3 | None = None
    quat_xyzw: Vec4 | None = None

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64)
        if self.coords.ndim == 1:
            self.coords = self.coords.reshape(1, 3)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        self.offset = Vec3(self.offset)
        if self.half_size is not None:
            self.half_size = Vec3(self.half_size)
        if self.quat_xyzw is not None:
            self.quat_xyzw = Vec4(self.quat_xyzw)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "frame": self.frame,
            "coords": self.coords.tolist(),
            "offset": self.offset.tolist(),
        }
        if self.half_size is not None:
            d["half_size"] = self.half_size.tolist()
        if self.quat_xyzw is not None:
            d["quat_xyzw"] = self.quat_xyzw.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Geometry:
        coords = [[float(p[0]), float(p[1]), float(p[2] if len(p) == 3 else 0.0)]
                  for p in data.get("coords", [[0.0, 0.0, 0.0]])]
        hs = data.get("half_size")
        q = data.get("quat_xyzw")
        return cls(
            type=data.get("type"),
            frame=data.get("frame"),
            coords=np.array(coords, dtype=np.float64),
            offset=Vec3(data.get("offset", [0.0, 0.0, 0.0])),
            half_size=Vec3(hs) if hs is not None else None,
            quat_xyzw=Vec4(q) if q is not None else None,
        )


@dataclass
class FrameGeometry:
    """Per-frame pinhole geometry: intrinsics, camera-to-world pose, and depth."""

    index: int
    K: np.ndarray
    c2w: np.ndarray
    depth: np.ndarray
    conf: np.ndarray | None = None
    rgb_ref: str | None = None

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.c2w = np.asarray(self.c2w, dtype=np.float64).reshape(4, 4)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {self.depth.shape}")
        if self.conf is not None:
            self.conf = np.asarray(self.conf, dtype=np.float32)
            if self.conf.shape != self.depth.shape:
                raise ValueError("conf must match depth shape")

    @property
    def hw(self) -> tuple[int, int]:
        return (int(self.depth.shape[0]), int(self.depth.shape[1]))

    @property
    def cam_center(self) -> np.ndarray:
        return self.c2w[:3, 3].astype(np.float64)


@dataclass
class Sim3:
    """Similarity transform g = s·R·l + t between two metric frames.

    Used to chain per-window reconstructions into one frame today; intended to
    become the per-window optimization variable for bundle adjustment later.
    """

    s: float = 1.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.s = float(self.s)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @classmethod
    def from_poses(cls, g_c2w: np.ndarray, l_c2w: np.ndarray) -> Sim3:
        """Estimate the Sim3 mapping local camera poses onto global ones."""
        g_c2w = np.asarray(g_c2w, dtype=np.float64)
        l_c2w = np.asarray(l_c2w, dtype=np.float64)
        gc, lc = g_c2w[:, :3, 3], l_c2w[:, :3, 3]
        iu = np.triu_indices(len(gc), 1)
        dg = np.linalg.norm(gc[None] - gc[:, None], axis=-1)[iu]
        dl = np.linalg.norm(lc[None] - lc[:, None], axis=-1)[iu]
        ok = dl > 1e-9
        s = float(np.median(dg[ok] / dl[ok])) if ok.any() else 1.0
        m = np.einsum("nij,nkj->ik", g_c2w[:, :3, :3], l_c2w[:, :3, :3])
        u, _, vt = np.linalg.svd(m)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        r = u @ vt
        t = gc.mean(axis=0) - s * r @ lc.mean(axis=0)
        return cls(s, r, t)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform (N, 3) points."""
        return self.s * np.asarray(points, dtype=np.float64) @ self.R.T + self.t

    def apply_pose(self, c2w: np.ndarray) -> np.ndarray:
        """Transform (..., 4, 4) camera-to-world poses."""
        out = np.asarray(c2w, dtype=np.float64).copy()
        out[..., :3, :3] = self.R @ out[..., :3, :3]
        out[..., :3, 3] = self.s * out[..., :3, 3] @ self.R.T + self.t
        return out
