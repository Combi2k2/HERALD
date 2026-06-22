"""RGB stream ingestion + optional provided geometry.

Turns an image directory or a video file into an ordered list of frames, with
``frame_stride`` / ``max_frames`` controls mirroring R3's ``infer.py``. Camera
trajectory and per-frame depth are *optional*: when the caller supplies poses +
depth + intrinsics they are passed straight through as :class:`FrameGeometry`
(no VGGT inference); otherwise the backend infers them. Video decoding uses a
lazy ``imageio`` import so the module stays importable in the torch-free env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from services.reconstruction.base import FrameGeometry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class StreamFrame:
    """One RGB frame: a file path and/or an in-memory image."""

    index: int
    path: str | None = None
    image: np.ndarray | None = None

    def load(self) -> np.ndarray:
        """Return the frame as an (H, W, 3) uint8 array (loads from path if needed)."""
        if self.image is not None:
            return np.asarray(self.image)
        if self.path is None:
            raise ValueError(f"frame {self.index} has neither image nor path")
        from PIL import Image

        return np.asarray(Image.open(self.path).convert("RGB"))


@dataclass
class Stream:
    """Ordered RGB frames plus any caller-provided geometry/reference trajectory."""

    frames: list[StreamFrame] = field(default_factory=list)
    intrinsics: np.ndarray | None = None  # (3, 3) shared or (N, 3, 3) per-frame
    poses: list[np.ndarray] | None = None  # per-frame c2w (4, 4)
    depths: list[np.ndarray] | None = None  # per-frame depth (H, W)
    reference_positions: np.ndarray | None = None  # (N, 3) ENU, for registration

    def __len__(self) -> int:
        return len(self.frames)

    def image_paths(self) -> list[str]:
        paths = [f.path for f in self.frames]
        if any(p is None for p in paths):
            raise ValueError("stream frames lack file paths; a path-based backend needs them")
        return [str(p) for p in paths]

    def load_images(self) -> list[np.ndarray]:
        return [f.load() for f in self.frames]

    def has_provided_geometry(self) -> bool:
        return self.poses is not None and self.depths is not None and self.intrinsics is not None

    def provided_geometry(self) -> list[FrameGeometry]:
        """Build FrameGeometry from caller-provided poses + depth + intrinsics."""
        if not self.has_provided_geometry():
            raise ValueError("provided_geometry() requires poses, depths, and intrinsics")
        k = np.asarray(self.intrinsics, dtype=np.float64)
        shared = k.shape == (3, 3)
        out: list[FrameGeometry] = []
        for i, f in enumerate(self.frames):
            out.append(
                FrameGeometry(
                    index=i,
                    K=k if shared else k[i],
                    c2w=self.poses[i],
                    depth=self.depths[i],
                    conf=None,
                    rgb_ref=f.path,
                )
            )
        return out

    @classmethod
    def from_arrays(cls, images: Sequence[np.ndarray], **kwargs) -> "Stream":
        frames = [StreamFrame(index=i, path=None, image=np.asarray(img)) for i, img in enumerate(images)]
        return cls(frames=frames, **kwargs)


def _extract_video_frames(path: Path, out_dir: Path, *, frame_stride: int, max_frames: int) -> list[str]:
    import imageio.v2 as iio

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    reader = iio.get_reader(str(path))
    try:
        kept = 0
        for i, frame in enumerate(reader):
            if i % frame_stride != 0:
                continue
            fp = out_dir / f"{kept:06d}.png"
            iio.imwrite(str(fp), frame)
            paths.append(str(fp))
            kept += 1
            if max_frames and kept >= max_frames:
                break
    finally:
        reader.close()
    return paths


def load_stream(
    source: str | Path,
    *,
    frame_stride: int = 1,
    max_frames: int = 0,
    intrinsics: np.ndarray | None = None,
    poses: list[np.ndarray] | None = None,
    depths: list[np.ndarray] | None = None,
    reference_positions: np.ndarray | None = None,
    video_cache: str | Path | None = None,
) -> Stream:
    """Load an image directory or a video file into a :class:`Stream`."""
    src = Path(source)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        paths = [str(p) for p in files][:: max(1, frame_stride)]
        if max_frames:
            paths = paths[:max_frames]
    elif src.suffix.lower() in VIDEO_EXTS:
        cache = Path(video_cache) if video_cache else src.with_suffix("").parent / f"{src.stem}_frames"
        paths = _extract_video_frames(
            src, cache, frame_stride=max(1, frame_stride), max_frames=max_frames
        )
    else:
        raise ValueError(f"unsupported stream source: {source!r}")

    frames = [StreamFrame(index=i, path=p) for i, p in enumerate(paths)]
    return Stream(
        frames=frames,
        intrinsics=intrinsics,
        poses=poses,
        depths=depths,
        reference_positions=reference_positions,
    )
