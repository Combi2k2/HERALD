#!/usr/bin/env python3
"""Query OSM, save results, and build an interactive map for inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from herald.osm.client import OSMQueryResult, query_nearby, save_geojson
from herald.osm.location import resolve_location

DEFAULT_RADIUS_M = 250.0
DEFAULT_OUT_DIR = Path("data/osm")


def save_metadata(
    result: OSMQueryResult,
    path: Path,
) -> None:
    meta: dict = {
        "center": {"lat": result.center.lat, "lon": result.center.lon},
        "radius_m": result.radius_m,
        "counts": {
            "buildings": len(result.buildings),
            "highways": len(result.highways),
        },
    }

    meta["buildings"] = [
        {
            "osm_id": f.osm_id,
            "osm_type": f.osm_type,
            "name": f.name,
            "kind": f.kind,
            "tags": f.tags,
        }
        for f in result.buildings
    ]
    meta["highways"] = [
        {
            "osm_id": f.osm_id,
            "osm_type": f.osm_type,
            "name": f.name,
            "highway": f.tags.get("highway"),
            "kind": f.kind,
        }
        for f in result.highways
    ]

    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_geojson(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_map(geojson: dict, out_html: Path) -> None:
    import folium
    from folium import plugins

    center = geojson["properties"]["center"]
    lat, lon = center["lat"], center["lon"]

    m = folium.Map(location=[lat, lon], zoom_start=17, tiles="OpenStreetMap")

    folium.Marker(
        [lat, lon],
        popup=folium.Popup(f"<b>Query center</b><br>{lat:.6f}, {lon:.6f}", max_width=280),
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=geojson["properties"]["radius_m"],
        color="#3388ff",
        fill=False,
        weight=2,
        dash_array="6",
        popup=f"Search radius ({geojson['properties']['radius_m']:.0f} m)",
    ).add_to(m)

    for feature in geojson["features"]:
        category = feature["properties"].get("category", "unknown")
        name = feature["properties"].get("name") or feature["id"]
        geom = feature["geometry"]
        props = feature["properties"]

        if geom["type"] == "Polygon":
            coords = [(c[1], c[0]) for c in geom["coordinates"][0]]
            color = "#2563eb" if category == "building" else "#ea580c"
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.35,
                popup=folium.Popup(
                    f"<b>{category}</b><br>{name}<br><pre>{json.dumps(props, indent=2)[:500]}</pre>",
                    max_width=320,
                ),
            ).add_to(m)
        elif geom["type"] == "LineString":
            coords = [(c[1], c[0]) for c in geom["coordinates"]]
            folium.PolyLine(
                locations=coords,
                color="#ea580c",
                weight=3,
                opacity=0.85,
                popup=folium.Popup(
                    f"<b>{category}</b><br>{name}<br>highway={props.get('highway', '?')}",
                    max_width=280,
                ),
            ).add_to(m)

    plugins.MeasureControl(position="topleft").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=None, help="Query center latitude")
    parser.add_argument("--lon", type=float, default=None, help="Query center longitude")
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = out_dir / "features.geojson"
    meta_path = out_dir / "metadata.json"
    map_path = out_dir / "map.html"

    try:
        loc = resolve_location(lat=args.lat, lon=args.lon)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    lat, lon = loc.lat, loc.lon
    print(f"Querying OSM around ({lat:.6f}, {lon:.6f}), radius={args.radius_m} m ...")
    result = query_nearby(lat, lon, radius_m=args.radius_m)
    print(f"  buildings={len(result.buildings)}, highways={len(result.highways)}")

    save_geojson(result, str(geojson_path))
    save_metadata(result, meta_path)
    print(f"Saved GeoJSON -> {geojson_path}")
    print(f"Saved metadata -> {meta_path}")

    geojson = load_geojson(geojson_path)
    assert len(geojson["features"]) == len(result.all_features)
    print(f"Loaded {len(geojson['features'])} features from disk")

    build_map(geojson, map_path)
    print(f"Wrote interactive map -> {map_path.resolve()}")
    print("Open map.html in a browser to inspect polygons and roads.")


if __name__ == "__main__":
    main()
