# HERALD

Site-scale semantic navigation framework (see `PLAN.md`).

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for virtualenvs and dependencies.

```bash
# Install uv (if needed): https://docs.astral.sh/uv/getting-started/installation/

cd HERALD
uv sync              # creates .venv/, installs herald + dev deps (pytest)
```

`uv sync` creates `.venv` automatically on first run. You do not need `python -m venv` or `pip install`. Commit `uv.lock` for reproducible installs.

## OSM module

Query OpenStreetMap for **building polygons** and **highway geometries** around a location via the [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API).

### Setup

From the repo root (see **Getting started** above):

```bash
uv sync
```

Activate the virtual environment (optional; `uv run` uses `.venv` automatically):

```bash
source .venv/bin/activate
```

Run Python or tests:

```bash
uv run python -c "from herald.osm import query_nearby; print(query_nearby(48.7128, 2.2060, radius_m=100))"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest   # avoids stray system pytest plugins (e.g. ROS)
```

### Usage

```python
from herald.osm import query_nearby

# IP Paris campus example
result = query_nearby(48.7128, 2.2060, radius_m=300)

print(f"Buildings: {len(result.buildings)}, highways: {len(result.highways)}")
for b in result.buildings:
    print(b.name, b.kind, len(b.geometry), "vertices")

# GeoJSON export
from herald.osm.client import save_geojson
save_geojson(result, "campus_osm.geojson")
```

With environment variables for “current” location:

```bash
export HERALD_LAT=48.7128
export HERALD_LON=2.2060
```

```python
from herald.osm import OSMClient
from herald.osm.location import resolve_location

loc = resolve_location()
client = OSMClient()
result = client.query_nearby(loc.lat, loc.lon, radius_m=200)
```

### Inspect query results (save + map)

Set the query center with `--lat` / `--lon` or `HERALD_LAT` / `HERALD_LON`:

```bash
export HERALD_LAT=48.7128
export HERALD_LON=2.2060
uv run python scripts/osm_inspect.py
# manual override:  --lat 48.71 --lon 2.21
# radius / output:   --radius-m 300 --out-dir data/osm
```

Writes to `data/osm/`:

| File | Contents |
|------|----------|
| `features.geojson` | All geometries (reloadable) |
| `metadata.json` | Summary + per-feature tags |
| `map.html` | Interactive Folium map (open in a browser) |

### Notes

- **Buildings** are returned as closed polygons when OSM ways form rings.
- **Highways** are usually centerline polylines; mapped areas (`area:highway`) are classified as polygons.
- Respect [OSM tile usage / API etiquette](https://operations.osmfoundation.org/policies/tiles/); the default Overpass endpoint is shared—avoid tight polling loops.

## Scene graph (offline initialization)

Build a hierarchical scene graph from OpenStreetMap: **site → outdoor zones → building zones**, with pathway-based outdoor partitioning.

```bash
uv sync --group dev --group viz --group viewer
export HERALD_LAT=48.7128 HERALD_LON=2.2060   # optional fallback if GPS denied
uv run python scripts/scene_init.py --rerun --serve   # browser GPS → picker → overlay
uv run python scripts/scene_init.py --port 8080 --serve   # default is 3000
uv run python scripts/scene_init.py --center 48.7128,2.2060 --serve
uv run python scripts/scene_init.py --bbox 48.710,2.200,48.713,2.203 --serve  # skip picker
uv run python scripts/scene_view.py             # replay saved graph in Rerun
```

Writes to `data/scene/current/` by default:

| File | Contents |
|------|----------|
| `scene_graph.json` | Site/outdoor/building nodes, captions, stub embeddings |
| `roi.geojson` | Selected region of interest |
| `pathways.geojson` | Walkable highways used for outdoor partitioning |
| `osm_polygons.geojson` | All raw OSM polygon footprints (landuse, leisure, amenity, …) |
| `map_overlay.html` | Folium map: OSM tiles + polygon overlays (layout check) |
| `metadata.json` | Node counts, OSM polygon counts by tag, timing, model id |

Use `--run-id <name>` to snapshot under `data/scene/<name>/` instead of overwriting `current/`.
