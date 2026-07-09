"""End-to-end RGB-stream reconstruction: VGGT geometry + SAM2 labels -> Fuser.

ReconStream rolls the three recon submodules into one push/flush interface.
Feed frames with push(); geometry is either supplied per frame (a VGGT-style
{"K","c2w","depth"[,"conf"]} dict, e.g. dataset GT) or predicted by a
VggtStream, and SAM2 labels every frame. VGGT and SAM2 each buffer into their
own chunks and emit one result per input frame in order, so two queues drain
in lockstep into the Fuser — each fused frame pairs the right geometry with
the right labels regardless of the two chunk sizes.

flush() is a non-destructive cloud snapshot (call it between pushes to watch
the scene build up); finish() drains the VGGT/SAM2 buffers and returns the
final cloud.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from herald.scene.recon.fusion import Fuser
from herald.scene.recon.sam2 import Sam2Stream
from herald.scene.recon.utils import as_rgb

class ReconStream:
    """Push RGB (+ optional geometry) frames, flush a semantic point cloud."""

    def __init__(self, sam2: Sam2Stream, fuser: Fuser, *, vggt=None) -> None:
        self.sam2 = sam2      # labels every frame
        self.vggt = vggt      # VggtStream, or None when geometry is supplied to push()
        self.fuser = fuser
        self._geo: deque = deque()
        self._lab: deque = deque()

    def _drain(self) -> None:
        while self._geo and self._lab:
            lab, conf = self._lab.popleft()
            self.fuser.push(self._geo.popleft(), lab, sem_conf=conf)

    def push(self, rgb, geo=None) -> None:
        """Feed one frame. `rgb` is an image path (required if VGGT predicts
        geometry) or an RGB array; `geo` is a per-frame VGGT-style dict that,
        when given, is used directly instead of running VGGT."""
        if geo is None:
            if self.vggt is None:
                raise RuntimeError("no geometry given and no VggtStream to predict it")
            self._geo.extend(self.vggt.push(rgb))
        else:
            self._geo.append(geo)
        self._lab.extend(self.sam2.push(as_rgb(rgb)))
        self._drain()

    def flush(self, **kw) -> dict:
        """Non-destructive snapshot of the frames fused so far."""
        return self.fuser.flush(**kw)

    def finish(self, **kw) -> dict:
        """Drain the VGGT/SAM2 buffers, then return the final cloud."""
        if self.vggt is not None:
            self._geo.extend(self.vggt.finish())
        self._lab.extend(self.sam2.finish())
        self._drain()
        return self.fuser.flush(**kw)

    def run_video(self, frames: Sequence, geos: Sequence | None = None, **kw) -> dict:
        """Whole-sequence convenience: push every frame, then finish()."""
        for i, f in enumerate(frames):
            self.push(f, None if geos is None else geos[i])
        return self.finish(**kw)
