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

### Inspect query results (Python)

```python
from herald.data import RunPaths

paths = RunPaths()
print(paths.run_id)   # e.g. 010626_135959
print(paths.root)     # data/010626_135959
print(paths.raw)      # data/010626_135959/raw
print(paths.phase1)   # data/010626_135959/phase1
```

### Notes

- **Buildings** are returned as closed polygons when OSM ways form rings.
- **Highways** are usually centerline polylines; mapped areas (`area:highway`) are classified as polygons.
- Respect [OSM tile usage / API etiquette](https://operations.osmfoundation.org/policies/tiles/); the default Overpass endpoint is shared—avoid tight polling loops.

## Scene graph (offline initialization)

Build a hierarchical scene graph from OpenStreetMap: **site → outdoor zones → building zones**, with pathway-based outdoor partitioning.

```bash
uv sync --group dev --group viz --group vlm --group viewer
export HERALD_LAT=48.7128 HERALD_LON=2.2060   # optional fallback if GPS denied
uv run python scripts/scene_init.py --rerun --serve   # browser GPS → picker → overlay
uv run python scripts/scene_init.py --vlm          # vision LLM per polygon (aerial + OSM map)
uv run python scripts/scene_init.py --vlm --vlm-model openai:gpt-4o  # proprietary model via LangChain
uv run python scripts/scene_init.py --port 8080 --serve   # default is 3000
uv run python scripts/scene_init.py --center 48.7128,2.2060 --serve
uv run python scripts/scene_init.py --bbox 48.710,2.200,48.713,2.203 --serve  # skip picker
uv run python scripts/scene_view.py --run-id 010626_135959
```

Writes one run folder (timestamp id ``ddmmyy_hhmmss``)::

```
data/{run_id}/
├── raw/
│   ├── roi.geojson
│   ├── osm.geojson
│   └── map_overlay.html
└── phase1/
    ├── scene_graph.json
    ├── pathways.geojson
    └── metadata.json
```
