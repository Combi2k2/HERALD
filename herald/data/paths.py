"""On-disk paths for one scene-init run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RUN_ID_FORMAT = "%d%m%y_%H%M%S"
RUN_ID = datetime.now().strftime(RUN_ID_FORMAT)
RUN_ROOT = Path("data") / RUN_ID
RAW_ROOT = RUN_ROOT / "raw"
PHASE1_ROOT = RUN_ROOT / "phase1"

ROI_GEOJSON = "roi.geojson"
OSM_GEOJSON = "osm.geojson"
MAP_OVERLAY_HTML = "map_overlay.html"
AERIAL_PNG = "aerial.png"
AERIAL_META_JSON = "aerial_meta.json"

SCENE_GRAPH_JSON = "scene_graph.json"
PATHWAYS_GEOJSON = "pathways.geojson"
ANNOTATIONS_JSON = "annotations.json"
METADATA_JSON = "metadata.json"
FRAME_JSON_NAME = "frame.json"

SITE_DATA_ROOT = Path("data")
FRAME_JSON = SITE_DATA_ROOT / FRAME_JSON_NAME


@dataclass(frozen=True)
class RunPaths:
    run_id: str = RUN_ID

    @property
    def root(self) -> Path:
        return Path("data") / self.run_id

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def phase1(self) -> Path:
        return self.root / "phase1"

    @property
    def roi(self) -> Path:
        return self.raw / ROI_GEOJSON

    @property
    def osm(self) -> Path:
        return self.raw / OSM_GEOJSON

    @property
    def map_overlay(self) -> Path:
        return self.raw / MAP_OVERLAY_HTML

    @property
    def aerial(self) -> Path:
        return self.raw / AERIAL_PNG

    @property
    def aerial_meta(self) -> Path:
        return self.raw / AERIAL_META_JSON

    @property
    def scene_graph(self) -> Path:
        return self.phase1 / SCENE_GRAPH_JSON

    @property
    def pathways(self) -> Path:
        return self.phase1 / PATHWAYS_GEOJSON

    @property
    def annotations(self) -> Path:
        return self.phase1 / ANNOTATIONS_JSON

    @property
    def metadata(self) -> Path:
        return self.phase1 / METADATA_JSON
