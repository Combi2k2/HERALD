"""Tier-1 per-session reconstruction: RGB(-D) -> static scene cloud + static OBBs.

SessionRecon (pipeline.py) drives the Boxer detect/lift/track engine
(services.boxer), accumulates the scene cloud (utils.SceneCloud), and drops
in-video movers via the corridor filter (corridor.py) -> a SessionResult
(types.py). Geometry comes from GT (datasets) or VGGT (services.vggt), adapted in
geometry.py.

types/corridor/geometry are numpy-only and import eagerly; SessionRecon pulls in
Boxer (torch), so it is loaded lazily -- importing this package stays cheap.
"""

from herald.scene.recon.corridor import classify_static
from herald.scene.recon.geometry import GtGeometry, VggtGeometry
from herald.scene.recon.types import SceneMap, SceneObject, SessionResult

__all__ = ["SessionRecon", "SceneMap", "SessionResult", "SceneObject",
           "classify_static", "GtGeometry", "VggtGeometry"]


def __getattr__(name: str):
    if name == "SessionRecon":
        from herald.scene.recon.pipeline import SessionRecon

        return SessionRecon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
