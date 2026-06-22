"""Unified site scene representation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph
from herald.scene.common.nav import NavGraph


@dataclass
class SceneRepr:
    """Shared site frame, semantic scene graph, and navigation graph."""

    frame: Frame
    graph: SceneGraph
    nav: NavGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "graph": self.graph.to_dict(),
            "nav": self.nav.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneRepr:
        return cls(
            frame=Frame.from_dict(data["frame"]),
            graph=SceneGraph.from_dict(data["graph"]),
            nav=NavGraph.from_dict(data["nav"]),
        )

    def to_json(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> SceneRepr:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
