#!/usr/bin/env python3
"""Phase 2: refine an existing scene graph with objects from an RGB stream.

Loads the Phase-1 graph for a run, runs the RGB-stream refinement (VGGT geometry
by default, stub perception in M1), and writes the augmented graph + render-only
point cloud sidecar under ``data/<run_id>/refine/``.

Example:
    python scripts/scene_refine.py --run 220626_145041 \\
        --stream ../LitReview/R3/examples/indoor --frame-stride 2 --max-frames 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from herald.data import RunPaths
from herald.scene.common.graph import SceneGraph
from herald.scene.refine import refine_scene_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Existing run id under data/ (uses its phase1 graph)")
    parser.add_argument("--stream", required=True, type=Path, help="Image directory or video file")
    parser.add_argument("--backend", default="vggt", choices=["vggt"], help="Geometry backend")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap number of frames (0=all)")
    parser.add_argument("--fuse-stride", type=int, default=4, help="Pixel stride for cloud fusion")
    parser.add_argument("--voxel", type=float, default=0.05, help="Voxel size (m) for cloud downsample")
    parser.add_argument("--conf-thresh", type=float, default=0.0, help="Min depth confidence to keep a pixel")
    parser.add_argument("--assoc-dist", type=float, default=0.5, help="Max 3D distance (m) to merge detections")
    parser.add_argument("--min-observations", type=int, default=1, help="Drop instances seen fewer times")
    parser.add_argument("--no-save", action="store_true", help="Run without writing artifacts")
    args = parser.parse_args()

    paths = RunPaths(run_id=args.run)
    if not paths.scene_graph.is_file():
        raise SystemExit(f"no phase1 scene graph at {paths.scene_graph}")

    graph = SceneGraph.from_json(paths.scene_graph)
    n_before = len(graph.nodes)
    print(f"Loaded {n_before} nodes from {paths.scene_graph}", flush=True)
    print(f"Refining from stream: {args.stream}", flush=True)

    backend = None
    if args.backend == "vggt":
        from services.reconstruction.vggt import VGGTBackend

        backend = VGGTBackend()

    result = refine_scene_graph(
        graph,
        args.stream,
        backend=backend,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        fuse_stride=args.fuse_stride,
        voxel=args.voxel,
        conf_thresh=args.conf_thresh,
        assoc_dist=args.assoc_dist,
        min_observations=args.min_observations,
        paths=paths,
        save=not args.no_save,
    )

    print(f"  + {len(result.object_nodes)} object node(s)", flush=True)
    print(f"  registered to ENU: {result.sim3.registered}", flush=True)
    if result.cloud is not None:
        print(f"  point cloud: {len(result.cloud)} points", flush=True)
    if result.metadata is not None:
        print(f"  wrote refined graph -> {paths.refine_scene_graph}", flush=True)
        print(f"  point cloud sidecar -> {result.metadata.point_cloud_uri}", flush=True)


if __name__ == "__main__":
    main()
