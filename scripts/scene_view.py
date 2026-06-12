#!/usr/bin/env python3
"""Replay a saved scene graph in the Rerun viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from herald.browser import suppress_gtk_atk_bridge_warning
from herald.data import FRAME_JSON, RunPaths
from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph
from herald.scene.init import pathways_from_geojson, raw_polygons_from_geojson
from herald.viewer.rerun_view import RerunSceneViewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run id printed by scene_init (ddmmyy_hhmmss)",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    suppress_gtk_atk_bridge_warning()

    paths = RunPaths(run_id=args.run_id)
    graph_path = args.graph or paths.scene_graph

    if not graph_path.is_file():
        print(f"Error: graph not found: {graph_path}", file=sys.stderr)
        raise SystemExit(1)

    frame = Frame.load(FRAME_JSON)
    if frame is None:
        print(f"Error: site frame not found: {FRAME_JSON}", file=sys.stderr)
        raise SystemExit(1)

    graph = SceneGraph.from_json(graph_path)

    pathways = None
    if paths.pathways.is_file():
        with paths.pathways.open(encoding="utf-8") as f:
            pathways = pathways_from_geojson(json.load(f))

    raw_polygons = None
    if paths.osm.is_file():
        with paths.osm.open(encoding="utf-8") as f:
            raw_polygons = raw_polygons_from_geojson(json.load(f))
    else:
        print(f"Warning: raw OSM not found: {paths.osm}", file=sys.stderr)

    viewer = RerunSceneViewer(spawn=True)
    viewer.set_frame(frame)
    if raw_polygons:
        viewer.render_raw_osm_polygons(raw_polygons)
    viewer.render_graph(graph, frame=frame, pathways=pathways)
    print(f"Run: {paths.run_id}")
    print(f"Loaded {len(graph.nodes)} nodes from {graph_path}")
    print("Rerun viewer open — close the viewer window to exit.")
    input()


if __name__ == "__main__":
    main()
