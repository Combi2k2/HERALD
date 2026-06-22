"""Feed-forward 3D reconstruction backends (depth + camera pose from RGB).

VGGT is the only real model backend; :class:`GeometryBackend` is a protocol kept
as a test seam. ``VGGTBackend`` is importable without torch (its heavy imports
are lazy); constructing/running it requires the optional ``recon`` extra.
"""

from services.reconstruction.base import FrameGeometry, GeometryBackend
from services.reconstruction.vggt import VGGTBackend

__all__ = ["FrameGeometry", "GeometryBackend", "VGGTBackend"]
