from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from herald.scene.common.geometry import FrameGeometry

SKY_SENTINEL_M = 1000.0

IMG_HW = (640, 640)
FOV_DEG = 90.0

NED_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def decode_depth(path: str | Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path))
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected RGBA depth PNG, got shape {rgba.shape} for {path}")
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])
    return bgra.view("<f4").squeeze(-1)

def decode_labels(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.int32)

def default_K(hw: tuple[int, int] = IMG_HW, fov_deg: float = FOV_DEG) -> np.ndarray:
    h, w = hw
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

def pose_to_c2w(row: np.ndarray) -> np.ndarray:
    c2w = np.eye(4)
    c2w[:3, :3] = _quat_to_R(np.asarray(row[3:7], float)) @ NED_FROM_OPTICAL
    c2w[:3, 3] = row[:3]
    return c2w

@dataclass
class TartanGroundTraj:

    root: Path
    env: str
    version: str = "omni"
    traj: str = "P0000"
    camera: str = "lcam_front"
    frame_ids: list[int] = field(default_factory=list)
    poses: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        img_dir = self._dir("image")
        if not img_dir.is_dir():
            raise FileNotFoundError(f"no image dir: {img_dir}")
        self.frame_ids = sorted(int(p.stem.split("_")[0]) for p in img_dir.glob(f"*_{self.camera}.png"))
        pose_file = self.base / f"pose_{self.camera}.txt"
        self.poses = np.loadtxt(pose_file) if pose_file.is_file() else np.empty((0, 7))

    @property
    def base(self) -> Path:
        return self.root / self.env / f"Data_{self.version}" / self.traj

    def _dir(self, modality: str) -> Path:
        return self.base / f"{modality}_{self.camera}"

    def __len__(self) -> int:
        return len(self.frame_ids)

    def rgb_path(self, i: int) -> Path:
        return self._dir("image") / f"{self.frame_ids[i]:06d}_{self.camera}.png"

    def depth_path(self, i: int) -> Path:
        return self._dir("depth") / f"{self.frame_ids[i]:06d}_{self.camera}_depth.png"

    def seg_path(self, i: int) -> Path:
        return self._dir("seg") / f"{self.frame_ids[i]:06d}_{self.camera}_seg.png"

    def load_rgb(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.rgb_path(i)).convert("RGB"))

class TartanGroundGeometry:

    name = "tartanground-gt"

    def __init__(self, traj: TartanGroundTraj, K: np.ndarray | None = None) -> None:
        self._traj = traj
        self.K = default_K() if K is None else np.asarray(K, float)

    def __len__(self) -> int:
        return len(self._traj)

    def geometry(self, index: int) -> FrameGeometry:
        depth = decode_depth(self._traj.depth_path(index))
        c2w = pose_to_c2w(self._traj.poses[index])
        return FrameGeometry(
            index=index, K=self.K, c2w=c2w, depth=depth,
            rgb_ref=str(self._traj.rgb_path(index)),
        )

class TartanGroundSeg:

    name = "tartanground-gt"

    def __init__(self, traj: TartanGroundTraj) -> None:
        self._traj = traj

    def __len__(self) -> int:
        return len(self._traj)

    def labels(self, index: int) -> np.ndarray:
        return decode_labels(self._traj.seg_path(index))

def seg_filename(frame_id: int, camera: str) -> str:
    return f"{frame_id:06d}_{camera}_seg.png"

def save_seg(out_dir: Path, frame_id: int, camera: str, labels: np.ndarray) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / seg_filename(frame_id, camera)
    Image.fromarray(np.asarray(labels).astype(np.uint16)).save(path)
    return path

class TartanGroundStoredSeg:

    def __init__(self, traj: TartanGroundTraj, modality: str = "seg_sam2") -> None:
        self._traj = traj
        self.name = modality
        self.dir = traj.base / f"{modality}_{traj.camera}"

    def __len__(self) -> int:
        return len(self._traj)

    def path(self, index: int) -> Path:
        return self.dir / seg_filename(self._traj.frame_ids[index], self._traj.camera)

    def labels(self, index: int) -> np.ndarray:
        return decode_labels(self.path(index))
