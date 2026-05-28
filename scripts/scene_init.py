#!/usr/bin/env python3
"""Build offline scene graph from ROI and save artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from herald.browser import suppress_gtk_atk_bridge_warning, open_browser as launch_browser
from herald.osm.client import OSMClient
from herald.osm.location import resolve_location
from herald.scene.embedding import StubEncoder
from herald.scene.graph import SceneGraph
from herald.scene.init import build_scene_graph, pathways_to_geojson
from herald.scene.roi import ROI
from herald.ui import SceneOverlayServer
from herald.ui.map_overlay import _write_overlay_map_html
from herald.ui.roi_picker import ROIPickerError

DEFAULT_OUT_DIR = Path("data/scene/current")


def save_artifacts(
    graph: SceneGraph,
    roi: ROI,
    pathways_geojson: dict,
    out_dir: Path,
    *,
    elapsed_s: float,
    osm_polygons_geojson: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph.to_json(out_dir / "scene_graph.json")
    (out_dir / "roi.geojson").write_text(
        json.dumps(roi.to_geojson(), indent=2), encoding="utf-8"
    )
    (out_dir / "pathways.geojson").write_text(
        json.dumps(pathways_geojson, indent=2), encoding="utf-8"
    )
    if osm_polygons_geojson is not None:
        (out_dir / "osm_polygons.geojson").write_text(
            json.dumps(osm_polygons_geojson, indent=2), encoding="utf-8"
        )
    outdoor = sum(1 for n in graph.nodes if n.level == "outdoor_region")
    buildings = sum(1 for n in graph.nodes if n.level == "building")
    meta = {
        "site_id": graph.site_id,
        "embedding_model_id": graph.embedding_model_id,
        "roi_area_m2": roi.area_m2(),
        "counts": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "outdoor_zones": outdoor,
            "buildings": buildings,
        },
        "elapsed_s": round(elapsed_s, 2),
    }
    if osm_polygons_geojson is not None:
        meta["osm_polygon_counts"] = osm_polygons_geojson.get("properties", {}).get(
            "counts", {}
        )
        meta["osm_polygon_total"] = osm_polygons_geojson.get("properties", {}).get(
            "total", 0
        )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
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
    args = parser.parse_args()

    suppress_gtk_atk_bridge_warning()

    out_dir = Path("data/scene") / args.run_id if args.run_id else args.out_dir

    needs_picker = args.roi is None and args.bbox is None
    want_live_map = not args.no_map
    overlay_server: SceneOverlayServer | None = None

    if needs_picker or want_live_map:
        overlay_server = SceneOverlayServer(out_dir, args.port)
        try:
            overlay_server.start()
        except OSError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    viewer = None
    on_event = None
    if args.rerun:
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

    print(f"ROI area: {roi.area_m2() / 1e6:.3f} km², vertices: {len(roi.latlon_vertices())}")

    if overlay_server is not None:
        overlay_server.set_status("Fetching OSM polygons…")

    encoder = StubEncoder()
    client = OSMClient()
    print("Fetching all OSM polygons (unfiltered Overpass query) …")
    raw_polygons = client.query_raw_polygons_in_polygon(roi.latlon_vertices())
    raw_counts = raw_polygons.counts_by_tag()
    print(f"  OSM polygons: {len(raw_polygons.polygons)} ({raw_counts})")

    if overlay_server is not None:
        overlay_server.set_status("Building scene graph…")

    t0 = time.monotonic()
    build = build_scene_graph(
        roi, client=client, encoder=encoder, on_event=on_event
    )
    graph = build.graph
    elapsed = time.monotonic() - t0

    pathways_geojson = pathways_to_geojson(build.pathways)
    osm_polygons_geojson = raw_polygons.to_geojson()

    save_artifacts(
        graph,
        roi,
        pathways_geojson,
        out_dir,
        elapsed_s=elapsed,
        osm_polygons_geojson=osm_polygons_geojson,
    )

    if viewer is not None:
        viewer.set_frame(graph.frame)
        viewer.render_raw_osm_polygons(raw_polygons.polygons)
        viewer.render_graph(graph, pathways=build.pathways)

    if not args.no_map:
        if overlay_server is not None:
            overlay_server.set_status("Writing map overlay…")
        map_path = out_dir / "map_overlay.html"
        _write_overlay_map_html(
            map_path,
            roi=roi,
            osm_polygons_geojson=osm_polygons_geojson,
            graph=graph,
            pathways_geojson=pathways_geojson,
        )
        print(f"Saved map overlay -> {map_path}")
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
            print(f"Open {map_path} in a browser to inspect the layout.")

    print(f"Nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")
    print(f"Saved scene graph -> {out_dir / 'scene_graph.json'}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
