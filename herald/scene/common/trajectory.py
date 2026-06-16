"""Ego trajectory: the SE(3) sensor poses the nav graph is constructed from."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Pose:
    t: float
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "pos": [self.pos[0], self.pos[1], self.pos[2]],
            "rot": [self.rot[0], self.rot[1], self.rot[2], self.rot[3]],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pose:
        pos = data["pos"]
        rot = data["rot"]
        return cls(
            t=float(data["t"]),
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            rot=(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
        )


@dataclass
class Trajectory:
    crs: str = "wgs84"
    poses: list[Pose] = field(default_factory=list)

    def add_pose(self, pose: Pose) -> None:
        self.poses.append(pose)

    def to_dict(self) -> dict[str, Any]:
        return {"crs": self.crs, "poses": [p.to_dict() for p in self.poses]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            crs=data.get("crs", "wgs84"),
            poses=[Pose.from_dict(p) for p in data.get("poses", [])],
        )

    def to_json(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> Trajectory:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
