"""Read a TartanGround camera-pose file  (experiments rule 4 — numpy, no torch chain).

Each line: tx ty tz qx qy qz qw. Returns positions (N,3) and quaternions (N,4, xyzw).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def pose_path(root: Path, env: str, traj: str, camera: str, version: str = "omni") -> Path:
    return root / env / f"Data_{version}" / traj / f"pose_{camera}.txt"


def load_poses(path: Path):
    a = np.loadtxt(path, dtype=np.float64)
    return a[:, :3].astype(np.float32), a[:, 3:7].astype(np.float32)
