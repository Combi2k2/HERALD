"""Register the reconstruction frame into the site ENU frame.

VGGT reconstructs in an arbitrary, up-to-scale world frame. To place objects in
the site's metric ENU frame we solve a similarity transform (Sim3: scale +
rotation + translation) from the VGGT camera centers to a reference trajectory
expressed in ENU (provided poses or GPS). When no reference is available we
return the identity transform flagged ``registered=False`` rather than fabricate
a georeferencing — downstream consumers can see the objects are recon-local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from services.reconstruction.base import FrameGeometry


@dataclass
class Sim3:
    """Similarity transform ``x' = scale * R @ x + t`` (recon frame -> ENU)."""

    scale: float = 1.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    registered: bool = False

    def __post_init__(self) -> None:
        self.scale = float(self.scale)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map points (N, 3) from the recon frame into the ENU frame."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return self.scale * (pts @ self.R.T) + self.t

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(scale=1.0, R=np.eye(3), t=np.zeros(3), registered=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "registered": self.registered,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sim3":
        return cls(
            scale=float(data.get("scale", 1.0)),
            R=np.asarray(data.get("R", np.eye(3).tolist()), dtype=np.float64),
            t=np.asarray(data.get("t", [0.0, 0.0, 0.0]), dtype=np.float64),
            registered=bool(data.get("registered", False)),
        )


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True) -> Sim3:
    """Least-squares Sim3 aligning ``src`` to ``dst`` (Umeyama 1991).

    Both are (N, 3). Returns the transform mapping ``src`` onto ``dst``.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    r = u @ s @ vt
    if with_scale:
        var_s = (sc**2).sum() / n
        scale = float((d * np.diag(s)).sum() / var_s) if var_s > 0 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * (r @ mu_s)
    return Sim3(scale=scale, R=r, t=t, registered=True)


def register_to_enu(
    geometries: Sequence[FrameGeometry],
    reference_positions: np.ndarray | None = None,
    *,
    min_points: int = 3,
) -> Sim3:
    """Align VGGT camera centers to a reference ENU trajectory.

    ``reference_positions`` is (N, 3) ENU positions, one per geometry frame (same
    order). With fewer than ``min_points`` matched points (or no reference) the
    transform is the identity flagged ``registered=False``.
    """
    if reference_positions is None:
        return Sim3.identity()
    ref = np.asarray(reference_positions, dtype=np.float64).reshape(-1, 3)
    src = np.array([fg.cam_center for fg in geometries], dtype=np.float64)
    n = min(len(src), len(ref))
    if n < min_points:
        return Sim3.identity()
    return umeyama(src[:n], ref[:n], with_scale=True)
