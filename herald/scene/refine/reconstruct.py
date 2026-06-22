"""Geometry façade: provided-data passthrough, else VGGT inference.

This is the single seam the pipeline uses to obtain per-frame geometry. If the
stream already carries poses + depth + intrinsics, those are returned directly
(no model run). Otherwise a :class:`GeometryBackend` (VGGT by default, or a fake
in tests) infers them from the frame images.
"""

from __future__ import annotations

from services.reconstruction.base import FrameGeometry, GeometryBackend
from herald.scene.refine.stream import Stream


def reconstruct(
    stream: Stream,
    backend: GeometryBackend | None = None,
    *,
    prefer_provided: bool = True,
) -> list[FrameGeometry]:
    """Return per-frame :class:`FrameGeometry` for the stream."""
    if prefer_provided and stream.has_provided_geometry():
        return stream.provided_geometry()
    if backend is None:
        from services.reconstruction.vggt import VGGTBackend

        backend = VGGTBackend()
    # Path-based backends (VGGT) want file paths; array-backed streams (tests with
    # a fake backend) have none, so fall back to the frame objects the fake ignores.
    try:
        frames = stream.image_paths()
    except ValueError:
        frames = list(stream.frames)
    return list(backend.infer(frames))
