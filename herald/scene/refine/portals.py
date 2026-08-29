"""Portal geometry shared by area clustering and rendering: does a route edge pass through a
portal's gate (cross its thin-axis plane within the panel rectangle + margin)."""

from __future__ import annotations

import numpy as np


def edge_crosses_gate(p0, p1, center, R, half, margin: float = 0.3) -> bool:
    thin = int(np.argmin(half))
    n = R[:, thin]
    dvec = p1 - p0
    denom = float(n @ dvec)
    if abs(denom) < 1e-9:
        return False
    t = float(n @ (center - p0)) / denom
    if not (0.0 <= t <= 1.0):
        return False
    rel = (p0 + t * dvec) - center
    return all(abs(float(rel @ R[:, ax])) <= half[ax] + margin for ax in range(3) if ax != thin)
