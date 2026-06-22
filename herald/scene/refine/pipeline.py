"""Phase-2 orchestrator: RGB stream -> object nodes on the scene graph.

Pipeline (see ``herald/scene/refine`` module docs): ingest -> geometry (VGGT or
provided passthrough) -> per-frame perception -> multi-view association ->
register recon frame into site ENU -> fuse a render-only point cloud -> integrate
``object`` nodes under their nearest ``region``. Persistence is optional.

Registration contract (M1): the recon frame is aligned to ENU only when
``stream.reference_positions`` (per-frame ENU positions) is supplied; otherwise
objects stay recon-local and the run is flagged ``registered=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from herald.data.paths import RunPaths
from herald.scene.common.graph import SceneGraph, SceneNode
from herald.scene.common.repr import SceneRepr
from herald.scene.refine.associate import associate_objects
from herald.scene.refine.fusion import PointCloud, fuse_geometries
from herald.scene.refine.integrate import integrate_objects
from herald.scene.refine.io import RefineMetadata, save_refine
from herald.scene.refine.perceive import ObjectSource, StubObjectSource
from herald.scene.refine.reconstruct import reconstruct
from herald.scene.refine.register import Sim3, register_to_enu
from herald.scene.refine.stream import Stream, load_stream
from services.embeddings import Encoder
from services.reconstruction.base import GeometryBackend


@dataclass
class RefineResult:
    """Augmented graph plus the new object nodes, cloud, transform, and metadata."""

    graph: SceneGraph
    object_nodes: list[SceneNode] = field(default_factory=list)
    cloud: PointCloud | None = None
    sim3: Sim3 | None = None
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: RefineMetadata | None = None


def refine_scene_graph(
    scene: SceneRepr | SceneGraph,
    stream: Stream | str | Path,
    *,
    backend: GeometryBackend | None = None,
    object_source: ObjectSource | None = None,
    encoder: Encoder | None = None,
    frame_stride: int = 1,
    max_frames: int = 0,
    fuse_stride: int = 4,
    voxel: float = 0.05,
    conf_thresh: float = 0.0,
    assoc_dist: float = 0.5,
    min_observations: int = 1,
    paths: RunPaths | None = None,
    save: bool = False,
) -> RefineResult:
    """Enrich ``scene``'s graph with object nodes detected from ``stream``."""
    graph = scene.graph if isinstance(scene, SceneRepr) else scene
    if not isinstance(stream, Stream):
        stream = load_stream(stream, frame_stride=frame_stride, max_frames=max_frames)

    geometries = reconstruct(stream, backend)
    if not geometries:
        return RefineResult(graph=graph, sim3=Sim3.identity(), cloud=PointCloud(np.empty((0, 3))))

    images = stream.load_images()
    source = object_source or StubObjectSource()
    detections = [source.detect(images[i], geometries[i]) for i in range(len(geometries))]

    instances = associate_objects(
        geometries, detections,
        assoc_dist=assoc_dist, conf_thresh=conf_thresh, min_observations=min_observations,
    )

    sim3 = register_to_enu(geometries, stream.reference_positions)

    cloud = fuse_geometries(
        geometries, images, stride=fuse_stride, voxel=voxel, conf_thresh=conf_thresh
    )
    if sim3.registered and len(cloud):
        cloud = PointCloud(sim3.apply(cloud.points), cloud.colors)

    integrated = integrate_objects(graph, instances, sim3, encoder=encoder)

    backend_id = getattr(backend, "model_id", None) or (
        "provided" if stream.has_provided_geometry() else "vggt-1b"
    )

    metadata = None
    if save:
        if paths is None:
            paths = RunPaths()
        metadata = save_refine(
            paths, graph, cloud, sim3,
            backend_id=backend_id,
            frame_count=len(geometries),
            object_count=len(integrated.object_nodes),
            embeddings=integrated.embeddings,
        )

    return RefineResult(
        graph=graph,
        object_nodes=integrated.object_nodes,
        cloud=cloud,
        sim3=sim3,
        embeddings=integrated.embeddings,
        metadata=metadata,
    )
