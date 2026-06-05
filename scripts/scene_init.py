#!/usr/bin/env python3
"""Build offline scene graph from ROI and save artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from herald.browser import suppress_gtk_atk_bridge_warning, open_browser as launch_browser
from herald.data import RunPaths
from services.osm.client import OSMClient
from services.osm.location import resolve_location
from services.aerial.fetch import fetch_aerial_for_bbox, save_aerial
from services.embeddings.encoder import StubEncoder
from herald.scene.common.graph import SceneGraph
from herald.scene.init import (
    BuildResult,
    build_scene_graph,
    pathways_to_geojson,
    save_classifications,
)
from herald.scene.common.roi import ROI
from herald.ui import SceneOverlayServer
from herald.ui.map_overlay import _write_overlay_map_html
from herald.ui.roi_picker import ROIPickerError


def save_raw_artifacts(
    paths: RunPaths,
    roi: ROI,
    osm_polygons_geojson: dict,
    *,
    aerial_saved: bool = False,
) -> None:
    paths.raw.mkdir(parents=True, exist_ok=True)
    paths.roi.write_text(json.dumps(roi.to_geojson(), indent=2), encoding="utf-8")
    paths.osm.write_text(json.dumps(osm_polygons_geojson, indent=2), encoding="utf-8")
    if not aerial_saved:
        return


def save_phase1_artifacts(
    paths: RunPaths,
    build: BuildResult,
    roi: ROI,
    pathways_geojson: dict,
    *,
    elapsed_s: float,
    osm_polygon_counts: dict[str, int] | None = None,
    osm_polygon_total: int | None = None,
) -> None:
    graph = build.graph
    paths.phase1.mkdir(parents=True, exist_ok=True)
    graph.to_json(paths.scene_graph)
    paths.pathways.write_text(json.dumps(pathways_geojson, indent=2), encoding="utf-8")
    save_classifications(paths.annotations, build.classifications)
    outdoor = sum(1 for n in graph.nodes if n.level == "outdoor_region")
    buildings = sum(1 for n in graph.nodes if n.level == "building")
    meta = {
        "run_id": paths.run_id,
        "schema_version": graph.schema_version,
        "site_id": graph.site_id,
        "embedding_model_id": graph.embedding_model_id,
        "roi_area_m2": roi.area_m2(),
        "raw_dir": str(paths.raw),
        "counts": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "outdoor_zones": outdoor,
            "buildings": buildings,
            "hierarchy_nodes": len(build.forest.nodes),
            "site_children": len(build.forest.site_children),
            "other_polygons": len(build.forest.others),
        },
        "elapsed_s": round(elapsed_s, 2),
    }
    if osm_polygon_counts is not None:
        meta["osm_polygon_counts"] = osm_polygon_counts
    if osm_polygon_total is not None:
        meta["osm_polygon_total"] = osm_polygon_total
    paths.metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _map_center_fallback(args: argparse.Namespace) -> tuple[float, float] | None:
    """Optional fallback map center if browser geolocation is unavailable."""
    if args.center is not None:
        parts = [p.strip() for p in args.center.split(",")]
        if len(parts) != 2:
            raise ROIPickerError("--center requires lat,lon")
        return float(parts[0]), float(parts[1])
    try:
        loc = resolve_location()
        return loc.lat, loc.lon
    except ValueError:
        return None


def resolve_roi(
    args: argparse.Namespace,
    *,
    overlay_server: SceneOverlayServer | None = None,
) -> tuple[ROI, bool]:
    """Return (roi, used_picker)."""
    if args.roi is not None:
        return ROI.from_geojson(args.roi), False
    if args.bbox is not None:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ROIPickerError("--bbox requires south,west,north,east")
        return ROI.from_bbox(*parts), False
    if overlay_server is None:
        raise ROIPickerError(
            "ROI picker requires the scene UI server; omit --no-map or pass --roi / --bbox."
        )
    fallback = _map_center_fallback(args)
    fb_lat, fb_lon = (fallback if fallback else (None, None))
    return (
        overlay_server.prompt_roi_rectangle(
            fallback_lat=fb_lat,
            fallback_lon=fb_lon,
            open_browser=not args.no_browser,
        ),
        True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, help="GeoJSON polygon file")
    parser.add_argument(
        "--bbox",
        help="south,west,north,east bounding box (skips ROI picker)",
    )
    parser.add_argument(
        "--center",
        help="lat,lon fallback if browser geolocation is denied (optional)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Open Rerun viewer and stream graph construction",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser windows (ROI picker or map overlay)",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip Folium map overlay after processing",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Keep map HTTP server open until Ctrl+C (same tab: picker → overlay)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="HTTP port for ROI picker and map overlay (default: 3000)",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Classify polygons with a vision LLM (aerial + OSM map per polygon)",
    )
    parser.add_argument(
        "--vlm-model",
        default="ollama:qwen2.5vl:3b",
        metavar="SPEC",
        help=(
            "LangChain model spec for --vlm (default: ollama:qwen2.5vl:3b). "
            "Examples: openai:gpt-4o, anthropic:claude-sonnet-4-6"
        ),
    )
    parser.add_argument(
        "--no-aerial",
        action="store_true",
        help="Skip aerial tile fetch (VLM requires aerial imagery)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--overpass-timeout",
        type=float,
        default=180.0,
        metavar="SECONDS",
        help="HTTP read timeout for Overpass queries (default: 180)",
    )
    args = parser.parse_args()

    show_progress = not args.no_progress

    suppress_gtk_atk_bridge_warning()

    paths = RunPaths()
    print(f"Run id: {paths.run_id}")

    needs_picker = args.roi is None and args.bbox is None
    want_live_map = not args.no_map
    overlay_server: SceneOverlayServer | None = None

    if needs_picker or want_live_map:
        overlay_server = SceneOverlayServer(paths.raw, args.port)
        try:
            overlay_server.start()
        except OSError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    viewer = None
    on_event = None
    if args.rerun:
        from services.osm.client import LatLon
        from herald.viewer.rerun_view import RerunSceneViewer

        viewer = RerunSceneViewer(spawn=True)
        on_event = viewer.publish

    try:
        roi, used_picker = resolve_roi(
            args,
            overlay_server=overlay_server if needs_picker else None,
        )
    except ROIPickerError as exc:
        if overlay_server is not None:
            overlay_server.shutdown()
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if overlay_server is not None and not want_live_map:
        overlay_server.shutdown()
        overlay_server = None

    print(f"ROI area: {roi.area_m2() / 1e6:.3f} km²")

    if overlay_server is not None:
        overlay_server.set_status("Fetching OSM polygons…")

    encoder = StubEncoder()
    client = OSMClient(timeout_s=args.overpass_timeout)
    print("Fetching all OSM polygons (unfiltered Overpass query) …")
    raw_polygons = client.query_raw_polygons_in_polygon(roi.latlon_vertices())
    raw_counts = raw_polygons.counts_by_tag()
    print(f"  OSM polygons: {len(raw_polygons.polygons)} ({raw_counts})")

    if viewer is not None:
        lat_c, lon_c = roi.centroid_latlon()
        from herald.scene.common.frame import LocalFrame

        viewer.set_frame(LocalFrame(origin=LatLon(lat=lat_c, lon=lon_c)))
        viewer.render_raw_osm_polygons(raw_polygons.polygons)

    if overlay_server is not None:
        overlay_server.set_status("Fetching aerial imagery…")

    aerial_image = None
    aerial_meta = None
    aerial_saved = False
    if args.vlm and not args.no_aerial:
        try:
            image, meta = fetch_aerial_for_bbox(
                roi.bbox(),
                show_progress=show_progress,
            )
            save_aerial(image, meta, paths.aerial, paths.aerial_meta)
            aerial_image = image
            aerial_meta = meta
            aerial_saved = True
            print(f"Saved aerial mosaic -> {paths.aerial}")
        except Exception as exc:
            print(f"Warning: aerial fetch failed ({exc}); continuing without VLM imagery.")
    elif args.vlm and args.no_aerial:
        print("Warning: --vlm set but --no-aerial skips imagery; using OSM templates only.")

    if overlay_server is not None:
        overlay_server.set_status("Building scene graph…")

    paths.phase1.mkdir(parents=True, exist_ok=True)

    print(
        "Building scene graph (containment → pathways → classification → nodes)…",
        flush=True,
    )

    def checkpoint(partial_graph: SceneGraph, classifications: dict) -> None:
        partial_graph.to_json(paths.scene_graph)
        save_classifications(paths.annotations, classifications)

    t0 = time.monotonic()
    build = build_scene_graph(
        roi,
        raw_polygons.polygons,
        client=client,
        highways=raw_polygons.highways,
        encoder=encoder,
        on_event=on_event,
        on_checkpoint=checkpoint,
        aerial_image=aerial_image,
        aerial_meta=aerial_meta,
        use_vlm=args.vlm,
        vlm_model=args.vlm_model,
        show_progress=show_progress,
    )
    graph = build.graph
    elapsed = time.monotonic() - t0

    pathways_geojson = pathways_to_geojson(build.pathways)
    osm_polygons_geojson = raw_polygons.to_geojson()

    save_raw_artifacts(
        paths,
        roi,
        osm_polygons_geojson,
        aerial_saved=aerial_saved,
    )
    save_phase1_artifacts(
        paths,
        build,
        roi,
        pathways_geojson,
        elapsed_s=elapsed,
        osm_polygon_counts=osm_polygons_geojson.get("properties", {}).get("counts"),
        osm_polygon_total=osm_polygons_geojson.get("properties", {}).get("total"),
    )

    if viewer is not None:
        viewer.log_pathways(build.pathways)

    if not args.no_map:
        if overlay_server is not None:
            overlay_server.set_status("Writing map overlay…")
        _write_overlay_map_html(
            paths.map_overlay,
            roi=roi,
            osm_polygons_geojson=osm_polygons_geojson,
            graph=graph,
            pathways_geojson=pathways_geojson,
        )
        print(f"Saved map overlay -> {paths.map_overlay}")
        if overlay_server is not None:
            overlay_server.set_overlay_from_scene(
                roi=roi,
                osm_polygons_geojson=osm_polygons_geojson,
                graph=graph,
                pathways_geojson=pathways_geojson,
            )
            if not used_picker and not args.no_browser:
                launch_browser(f"{overlay_server.url}/")
            if args.serve:
                overlay_server.block_until_interrupt()
            elif not args.no_browser and not used_picker:
                print("Tip: pass --serve to keep the map server running.")
        else:
            print(f"Open {paths.map_overlay} in a browser to inspect the layout.")

    print(f"Nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")
    print(f"Saved run -> {paths.root}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
