"""Per-session geometry sources: each yields indexed (frame_idx, rgb, K, c2w, depth).

Two backends feed the same SessionRecon loop:
  - GtGeometry   -- dataset ground-truth K/c2w/depth (e.g. TartanGround).
  - VggtGeometry -- monocular geometry from services.vggt.VggtStream (no GT depth).

VGGT lives in services; it is wired in here rather than duplicated. Note its poses
are up-to-scale and not gravity-aligned to NED, so the VGGT path needs scale/gauge
reconciliation before Boxer's NED assumption holds (deferred -- GT path is exact)."""

from __future__ import annotations

import numpy as np
from PIL import Image


class GtGeometry:
    """Dataset ground-truth geometry. `traj` loads RGB by index; `geo.geometry(i)`
    returns an object with .K, .c2w, .depth (the TartanGround adapters)."""

    def __init__(self, traj, geo):
        self.traj, self.geo = traj, geo

    def frames(self, indices):
        for i in indices:
            g = self.geo.geometry(i)
            yield i, self.traj.load_rgb(i).astype(np.uint8), g.K, g.c2w, g.depth


class VggtGeometry:
    """Monocular geometry via services.vggt.VggtStream. RGB is reloaded to the
    model's working resolution so it matches the predicted depth/K."""

    def __init__(self, model_path, *, device="cuda", **vggt_kw):
        from services.vggt import VggtStream

        self.stream = VggtStream(model_path, device=device, **vggt_kw)

    def frames(self, rgb_paths):
        paths = list(rgb_paths)
        for i, (p, g) in enumerate(zip(paths, self.stream.run_stream(paths))):
            h, w = g["depth"].shape
            rgb = np.asarray(Image.open(p).convert("RGB").resize((w, h)))
            yield i, rgb, g["K"], g["c2w"], g["depth"]
