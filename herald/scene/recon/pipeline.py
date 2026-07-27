"""End-to-end RGB-stream reconstruction: VGGT geometry + SAM2 masks -> Fuser.

ReconStream rolls the three recon submodules into one push/flush interface.
Feed frames with push(); geometry is either supplied per frame (a VGGT-style
{"K","c2w","depth"[,"conf"]} dict or FrameGeometry, e.g. dataset GT) or
predicted by a VggtStream, and SAM2 segments every frame into instance masks.
Each frame's (masks, rgb) is queued alongside its geometry; because VGGT may
buffer frames into chunks while SAM2 emits per frame, the two queues drain in
lockstep so each fused frame pairs the right geometry with the right masks.

The Fuser associates masks into objects by appearance + occupancy (see
fusion.py), so it needs both the masks and the RGB frame they came from.

flush() is a non-destructive cloud snapshot (call it between pushes to watch
the scene build up); finish() drains the VGGT buffer and returns the final
cloud.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from herald.scene.recon.fusion import Fuser
from herald.scene.recon.sam2 import Sam2Segmenter
from herald.scene.recon.utils import as_rgb


class ReconStream:
    """Push RGB (+ optional geometry) frames, flush a semantic point cloud."""

    def __init__(self, sam2: Sam2Segmenter, fuser: Fuser, *, vggt=None) -> None:
        self.sam2 = sam2      # segments every frame into instance masks
        self.vggt = vggt      # VggtStream, or None when geometry is supplied to push()
        self.fuser = fuser
        self._geo: deque = deque()
        self._lab: deque = deque()    # (masks, rgb) per frame, in order

    def _drain(self) -> None:
        while self._geo and self._lab:
            masks, rgb = self._lab.popleft()
            self.fuser.push(self._geo.popleft(), masks, rgb)

    def push(self, rgb, geo=None, masks=None) -> None:
        """Feed one frame. `rgb` is an image path (required if VGGT predicts
        geometry) or an RGB array; `geo` is a per-frame VGGT-style dict /
        FrameGeometry that, when given, is used directly instead of running
        VGGT. Pass `masks` to reuse masks already computed by
        `self.sam2.segment(rgb)` (e.g. to render them) instead of segmenting
        again."""
        rgb = as_rgb(rgb)
        if geo is None:
            if self.vggt is None:
                raise RuntimeError("no geometry given and no VggtStream to predict it")
            self._geo.extend(self.vggt.push(rgb))
        else:
            self._geo.append(geo)
        self._lab.append((self.sam2.segment(rgb) if masks is None else masks, rgb))
        self._drain()

    def flush(self, **kw) -> dict:
        """Non-destructive snapshot of the frames fused so far."""
        return self.fuser.flush(**kw)

    def finish(self, **kw) -> dict:
        """Drain the VGGT buffer, then return the final cloud."""
        if self.vggt is not None:
            self._geo.extend(self.vggt.finish())
        self._drain()
        return self.fuser.flush(**kw)

    def run_video(self, frames: Sequence, geos: Sequence | None = None, **kw) -> dict:
        """Whole-sequence convenience: push every frame, then finish()."""
        for i, f in enumerate(frames):
            self.push(f, None if geos is None else geos[i])
        return self.finish(**kw)
