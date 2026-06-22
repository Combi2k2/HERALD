#!/usr/bin/env python3
"""Replay a saved scene in the Rerun viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from herald.browser import suppress_gtk_atk_bridge_warning
from herald.data import FRAME_JSON, RunPaths
from herald.ui import render_scene
from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph
from herald.scene.common.nav import NavGraph
from herald.scene.common.repr import SceneRepr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run id from scene_init")
    parser.add_argument("--graph", type=Path, default=None)
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
    nav = NavGraph.from_json(paths.nav_graph) if paths.nav_graph.is_file() else NavGraph()
    render_scene(SceneRepr(frame=frame, graph=graph, nav=nav), spawn=True)

    print(f"Run: {paths.run_id}  nodes: {len(graph.nodes)}  nav: {len(nav.nodes)}")
    print("Toggle semantic / hierarchy / nav in the entity tree. Close Rerun to exit.")
    input()


if __name__ == "__main__":
    main()
