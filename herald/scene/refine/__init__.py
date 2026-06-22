"""Phase 2: scene graph refinement from an RGB stream.

Enriches the Phase-1 graph with fine-grained ``object`` nodes detected from an
RGB walkthrough. Camera trajectory and depth are optional inputs; absent, they
are inferred by VGGT (:mod:`services.reconstruction`). A fused point cloud is
produced for rendering only and stored as a sidecar — never in the graph schema.
"""

from herald.scene.refine.associate import ObjectInstance, associate_objects
from herald.scene.refine.fusion import PointCloud, fuse_geometries
from herald.scene.refine.integrate import IntegrateResult, integrate_objects
from herald.scene.refine.io import RefineMetadata, load_refine_metadata, save_refine
from herald.scene.refine.perceive import Detection, ObjectSource, StubObjectSource
from herald.scene.refine.pipeline import RefineResult, refine_scene_graph
from herald.scene.refine.reconstruct import reconstruct
from herald.scene.refine.register import Sim3, register_to_enu, umeyama
from herald.scene.refine.stream import Stream, StreamFrame, load_stream

__all__ = [
    "refine_scene_graph",
    "RefineResult",
    "Stream",
    "StreamFrame",
    "load_stream",
    "reconstruct",
    "Detection",
    "ObjectSource",
    "StubObjectSource",
    "ObjectInstance",
    "associate_objects",
    "Sim3",
    "register_to_enu",
    "umeyama",
    "PointCloud",
    "fuse_geometries",
    "IntegrateResult",
    "integrate_objects",
    "RefineMetadata",
    "save_refine",
    "load_refine_metadata",
]
