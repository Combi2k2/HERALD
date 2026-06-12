"""Legacy geometry helpers for v2 run data."""

from __future__ import annotations

from typing import Any

import numpy as np

from herald.scene.common.geometry import Geometry, Vec3


def geom_from_v2(geometry_latlon: list[Any], *, height: float) -> Geometry:
    rows = [
        [float(p[0]), float(p[1]), 0.0] if len(p) >= 2 else [0.0, 0.0, 0.0]
        for p in geometry_latlon
    ]
    coords = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3), dtype=np.float64)
    offset = Vec3((0.0, 0.0, height)) if height > 0 else Vec3.zeros()
    geom_type = "point" if coords.shape[0] == 1 else "polygon"
    return Geometry(type=geom_type, coords=coords, frame="WGS", offset=offset)
