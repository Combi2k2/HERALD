#!/usr/bin/env python3
"""Replay a saved scene graph in the Rerun viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from herald.browser import suppress_gtk_atk_bridge_warning
from herald.scene.graph import SceneGraph
from herald.scene.init import pathways_from_geojson, raw_polygons_from_geojson
from herald.viewer.rerun_view import RerunSceneViewer

DEFAULT_GRAPH = Path("data/scene/current/scene_graph.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    suppress_gtk_atk_bridge_warning()

    graph_path = args.graph
    if args.run_id is not None:
        graph_path = Path("data/scene") / args.run_id / "scene_graph.json"

    if not graph_path.is_file():
        print(f"Error: graph not found: {graph_path}", file=sys.stderr)
        raise SystemExit(1)

    graph = SceneGraph.from_json(graph_path)
    pathways_path = graph_path.parent / "pathways.geojson"
    pathways = None
    if pathways_path.is_file():
        with pathways_path.open(encoding="utf-8") as f:
            pathways = pathways_from_geojson(json.load(f))

    raw_polygons_path = graph_path.parent / "osm_polygons.geojson"
    raw_polygons = None
    if raw_polygons_path.is_file():
        with raw_polygons_path.open(encoding="utf-8") as f:
            raw_polygons = raw_polygons_from_geojson(json.load(f))

    viewer = RerunSceneViewer(spawn=True)
    viewer.set_frame(graph.frame)
    if raw_polygons:
        viewer.render_raw_osm_polygons(raw_polygons)
    viewer.render_graph(graph, pathways=pathways)
    print(f"Loaded {len(graph.nodes)} nodes from {graph_path}")
    print("Rerun viewer open — close the viewer window to exit.")
    input()


if __name__ == "__main__":
    main()
