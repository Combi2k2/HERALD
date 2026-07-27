"""RGB frames -> camera intrinsics, poses, and depth via VGGT-Omega.

Service-side home of the monocular-geometry model wrapper. `VggtStream` buffers
frame paths into fixed-size chunks and runs one VGGT-Omega forward pass per
chunk; consecutive chunks share `chunk_overlap` frames and a Sim3 estimated from the
shared poses maps each new chunk into the first chunk's frame (depth rescaled
accordingly), so drift accumulates but the stream stays globally consistent.
Each emitted dict holds K (3,3), c2w (4,4), depth (H,W), conf (H,W).

Heavy CUDA deps (torch + vggt_omega, `uv sync --group recon`). Imported only by
recon code that has opted into geometry, so importing `services` stays cheap.
Depends on herald for the Sim3 helper and w2c->c2w conversion.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from herald.scene.common.geometry import Sim3
from herald.scene.recon.utils import w2c_to_c2w


class VggtStream:
    """Push RGB frame paths, receive per-frame geometry dicts one chunk at a time.

    Frames buffer into chunks and each chunk runs one VGGT forward pass.
    Consecutive chunks share `chunk_overlap` frames; a Sim3 estimated from the shared
    camera poses maps each new chunk into the first chunk's frame (depth rescaled
    accordingly). Drift accumulates across chunks. Each emitted dict holds K
    (3,3), c2w (4,4), depth (H,W), conf (H,W).
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        chunk_size: int = 32,
        chunk_overlap: int = 4,
        image_size: int = 512,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available; pass device='cpu' to run on CPU.")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        model = VGGTOmega().eval()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = model.to(device)
        self.chunk_size = max(2, chunk_size)
        self.chunk_overlap = max(2, min(chunk_overlap, self.chunk_size - 1))
        self.image_size = image_size
        self.reset()

    def reset(self) -> None:
        """Drop buffered + Sim3-tail state so the same loaded model can start a
        fresh stream. Clearing the tail is what keeps the next stream's first
        chunk from being Sim3-aligned to the previous stream (weights stay
        resident -- no reload)."""
        self._buffer: list[str] = []
        self._tail_paths: list[str] = []
        self._tail_c2w: np.ndarray | None = None

    def _infer(self, image_paths: Sequence[str]) -> dict[str, np.ndarray]:
        """One VGGT-Omega forward pass over an RGB frame sequence.

        Returns {"K": (S,3,3), "c2w": (S,4,4), "depth": (S,H,W), "conf": (S,H,W)}.
        H/W are the model's working resolution, not the input resolution; K is
        in those pixels and c2w is camera-to-world.
        """
        device = next(self.model.parameters()).device
        images = load_and_preprocess_images(
            [str(p) for p in image_paths], image_resolution=self.image_size
        ).to(device)

        with torch.inference_mode():
            predictions = self.model(images)

        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        w2c = extrinsic.float().cpu().numpy()  # (S,3,4) world-to-camera
        w2c = w2c[0] if w2c.ndim == 4 else w2c
        K = intrinsic.float().cpu().numpy()
        K = K[0] if K.ndim == 4 else K
        depth = predictions["depth"].float().cpu().numpy()
        depth = depth[0] if depth.ndim == 5 else depth
        depth = depth[..., 0] if depth.ndim == 4 else depth
        conf_t = predictions.get("depth_conf")
        if conf_t is not None:
            conf = conf_t.float().cpu().numpy()
            conf = conf[0] if conf.ndim == 4 else conf
        else:
            conf = np.ones_like(depth)

        return {"K": K, "c2w": w2c_to_c2w(w2c), "depth": depth, "conf": conf}

    def run_chunk(self, image_paths: Sequence) -> list[dict[str, np.ndarray]]:
        """One forward pass over `image_paths`, Sim3-aligned to prior chunks."""
        paths = self._tail_paths + [str(p) for p in image_paths]
        geo = self._infer(paths)
        keep = len(self._tail_paths)
        if keep:
            sim3 = Sim3.from_poses(self._tail_c2w, geo["c2w"][:keep])
            geo["c2w"] = sim3.apply_pose(geo["c2w"])
            geo["depth"] = geo["depth"] * sim3.s
        self._tail_paths = paths[-self.chunk_overlap:]
        self._tail_c2w = geo["c2w"][-self.chunk_overlap:]
        return [{k: geo[k][i] for k in geo} for i in range(keep, len(paths))]

    def push(self, image_path) -> list[dict[str, np.ndarray]]:
        self._buffer.append(str(image_path))
        if len(self._tail_paths) + len(self._buffer) >= self.chunk_size:
            new, self._buffer = self._buffer, []
            return self.run_chunk(new)
        return []

    def finish(self) -> list[dict[str, np.ndarray]]:
        new, self._buffer = self._buffer, []
        return self.run_chunk(new) if new else []

    def run_stream(self, image_paths: Iterable) -> Iterator[dict[str, np.ndarray]]:
        """Lazily yield one geometry dict per frame path, a chunk at a time."""
        for p in image_paths:
            yield from self.push(p)
        yield from self.finish()

    def run_video(self, image_paths: Sequence) -> dict[str, np.ndarray]:
        """Whole-sequence convenience: returns stacked {"K","c2w","depth","conf"}."""
        frames = list(self.run_stream(image_paths))
        if not frames:
            return {"K": np.empty((0, 3, 3)), "c2w": np.empty((0, 4, 4)),
                    "depth": np.empty((0, 1, 1), np.float32), "conf": np.empty((0, 1, 1), np.float32)}
        return {k: np.stack([f[k] for f in frames]) for k in ("K", "c2w", "depth", "conf")}
