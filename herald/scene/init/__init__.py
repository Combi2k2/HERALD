"""Phase 1: offline scene graph initialization from OSM."""

from herald.scene.common.repr import SceneRepr
from herald.scene.init.io import raw_polygons_from_geojson, save_annotations
from herald.scene.init.pathways import PathConfig, build_path
from herald.scene.init.pipeline import build_scene_graph

BuildResult = SceneRepr

__all__ = [
    "BuildResult",
    "PathConfig",
    "SceneRepr",
    "build_path",
    "build_scene_graph",
    "raw_polygons_from_geojson",
    "save_annotations",
]
