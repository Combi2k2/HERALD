"""Persistence for Phase-2 (refine) artifacts.

Writes the refined scene graph plus three sidecars under ``data/<run_id>/refine/``:
the registration transform, a render-only point cloud, and a metadata file that
holds the ``point_cloud_uri``. The point cloud is **only** referenced by URI here
— it is never embedded in ``scene_graph.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from herald.data.paths import RunPaths
from herald.scene.common.graph import SceneGraph
from herald.scene.refine.fusion import PointCloud
from herald.scene.refine.register import Sim3

EMBEDDINGS_NPZ = "embeddings.npz"


@dataclass
class RefineMetadata:
    """Summary + sidecar pointers for one refine run."""

    backend_id: str
    frame_count: int
    object_count: int
    registered: bool
    point_cloud_uri: str | None = None
    point_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "frame_count": self.frame_count,
            "object_count": self.object_count,
            "registered": self.registered,
            "point_cloud_uri": self.point_cloud_uri,
            "point_count": self.point_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefineMetadata":
        return cls(
            backend_id=str(data.get("backend_id", "")),
            frame_count=int(data.get("frame_count", 0)),
            object_count=int(data.get("object_count", 0)),
            registered=bool(data.get("registered", False)),
            point_cloud_uri=data.get("point_cloud_uri"),
            point_count=int(data.get("point_count", 0)),
        )


def save_refine(
    paths: RunPaths,
    graph: SceneGraph,
    cloud: PointCloud,
    sim3: Sim3,
    *,
    backend_id: str,
    frame_count: int,
    object_count: int,
    embeddings: dict[str, np.ndarray] | None = None,
) -> RefineMetadata:
    """Write graph + registration + point cloud + metadata (+ optional embeddings)."""
    paths.refine.mkdir(parents=True, exist_ok=True)

    graph.to_json(paths.refine_scene_graph)
    cloud.write_ply(paths.point_cloud)
    paths.registration.write_text(json.dumps(sim3.to_dict(), indent=2), encoding="utf-8")

    if embeddings:
        np.savez(paths.refine / EMBEDDINGS_NPZ, **embeddings)

    meta = RefineMetadata(
        backend_id=backend_id,
        frame_count=frame_count,
        object_count=object_count,
        registered=sim3.registered,
        point_cloud_uri=str(paths.point_cloud),
        point_count=len(cloud),
    )
    paths.refine_metadata.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return meta


def load_refine_metadata(paths: RunPaths) -> RefineMetadata:
    with paths.refine_metadata.open(encoding="utf-8") as f:
        return RefineMetadata.from_dict(json.load(f))
