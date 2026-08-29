# HERALD

Site-scale semantic navigation framework. HERALD builds and refines one hierarchical scene
graph — `site → region/zone → area → object → ego` — in two phases (see `PLAN.md` for the full
roadmap and `CLAUDE.md` for architecture detail):

- **Phase 1 — offline scene-graph init:** OpenStreetMap polygons → a `SceneRepr` (graph + nav graph).
- **Phase 2 — online RGB(-D) reconstruction:** a session's frames → static object OBBs + a colored
  scene cloud, then multiple sessions reconciled into one persistent map.

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (never `pip`/`venv` directly).
`.venv/` is created on first `uv sync`.

```bash
cd HERALD
uv sync                          # default groups: dev, viz, vlm
uv sync --group recon            # + VGGT-Omega geometry + torch/opencv (heavy, CUDA)
uv sync --group seg              # + SAM2/SigLIP via transformers (heavy, CUDA)
uv sync --group viewer           # + rerun-sdk (Rerun viewer)
```

**Boxer engine (Phase-2 recon).** The detector/tracker lives in the `third_party/boxer` git
submodule; its checkpoints (~1.1 GB) are not in git. On a fresh clone:

```bash
git submodule update --init third_party/boxer     # or clone with --recurse-submodules
bash third_party/boxer/scripts/download_ckpts.sh   # BoxerNet + DINOv3 + OWLv2 from HuggingFace
```

**Dataset (Phase-2 demos).** Reconstruction runs on TartanGround:

```bash
uv run python scripts/download_tartanground.py     # writes data/tartanground/<Env>/...
```

## Phase 1 — Scene-graph init (OSM)

Build a hierarchical scene graph from OpenStreetMap: **site → outdoor zones → building zones**,
with pathway-based outdoor partitioning, and validate it in Rerun/Folium.

```bash
uv sync --group dev --group viz --group vlm --group viewer
export HERALD_LAT=48.7128 HERALD_LON=2.2060            # optional fallback if GPS denied
uv run python scripts/scene_init.py --serve            # browser GPS → ROI picker → build
uv run python scripts/scene_init.py --bbox 48.710,2.200,48.713,2.203 --serve   # skip picker
uv run python scripts/scene_init.py --vlm              # vision-LLM classifies each polygon
uv run python scripts/scene_view.py --run-id 010626_135959    # view a finished run
```

Each run writes `data/{run_id}/` (timestamp id `ddmmyy_hhmmss`):

```
data/{run_id}/
├── raw/     roi.geojson · osm.geojson · map_overlay.html
└── phase1/  scene_graph.json · pathways.geojson · metadata.json
```

## Phase 2 — RGB(-D) reconstruction → persistent map

Each session's frames become a `SessionResult` (Tier 1: static object OBBs + scene cloud); the
sessions are then folded one-by-one into a single **persistent `SceneMap` on disk** (Tier 2) and
objects are clustered into **portal-bounded areas**. Tier 2 consumes only the saved
`SessionResult` dumps — never raw frames. Each stage is its own `scene_*` script and only writes
files to disk; **rendering is a separate step**.

The persistent representation lives under `data/tartanground/<Env>/scene/`:

```
persistent.npz     accumulating SceneMap (cloud + object OBBs)
route.json         persistent route graph (nav-node lattice)
attach.json        object → observing-nav-node attachment
scene_graph.json   area/object SceneGraph
sim3_<T>.json      per-session Sim3
```

### Run it

Per session `T`: recon (GPU) → align → track_obj → merge_route (CPU); then area-cluster once.

```bash
REC=data/tartanground/Hospital/recon ; S=data/tartanground/Hospital/scene

# Tier 1 (GPU): one video -> SessionResult
uv run python scripts/scene_recon.py --env Hospital --traj P0000 --out $REC/session_P0000.npz

# Tier 2 (CPU): fold the session into the persistent map
uv run python scripts/scene_align.py     --session $REC/session_P0000.npz --persistent $S/persistent.npz --out $S/sim3_P0000.json --no-align
uv run python scripts/scene_track_obj.py --session $REC/session_P0000.npz --persistent $S/persistent.npz --sim3 $S/sim3_P0000.json
uv run python scripts/scene_merge_route.py --objects $S/persistent.npz --route $S/route.json --attach $S/attach.json --session P0000 --sim3 $S/sim3_P0000.json

# after ALL sessions: portal-bounded area clustering (+ optional VLM captions)
uv run python scripts/scene_area_cluster.py  --objects $S/persistent.npz --route $S/route.json --attach $S/attach.json --out $S/scene_graph.json
uv run python scripts/scene_area_classify.py --graph $S/scene_graph.json --out $S/scene_graph.json

# render the persistent representation
uv run python scripts/scene_render.py --objects $S/persistent.npz --graph $S/scene_graph.json --route $S/route.json --out $S/scene.rrd
```

The whole flow (recon on GPU per trajectory → integration on CPU → area-cluster) runs via one job:

```bash
sbatch --export=ALL,ENV_NAME=Hospital,TRAJS="P0000 P0001 P0002" scripts/scene_pipeline.slurm
```

`--no-align` uses an identity Sim3 — TartanGround global poses are already co-registered; real,
independently-originated sessions drop it and let `scene_align` recover the Sim3.

Key knobs — recon: `--conf-thr2d`/`--conf-thr3d` (OWLv2/BoxerNet floors) · `--min-obs` (track
support) · `--max-range`/`--min-visible` (range + box-visibility gates) · `--corridor-step`
(dynamic filter). Route: `--spacing` (nav-node spacing) · `--r-connect` (proximity radius).
Areas: `--d-max` (area diameter cap) · `--linkage` · `--mirror-adj`/`--gate-margin` (portal
disambiguation). Render: portals are magenta bboxes, doors solid magenta panels, the route graph
grey with portal-cut edges in red; click any object/door/portal for its crops.

## Cluster / GPU (SLURM)

Heavy jobs run via `scripts/*.slurm` (`sbatch`, override config with `--export=ALL,KEY=val,...`).
Hard constraints (see `CLAUDE.md`): use `--partition=3090` (not P100), and clear `PYTHONPATH`
(`unset PYTHONPATH` / prefix `PYTHONPATH= uv run ...`) so system site-packages don't shadow the
uv venv. Login nodes have no usable GPU. Job outputs go to `runs/{jobid}/`.

## Repository layout

```
herald/scene/init/     Phase-1 scene-graph builder
herald/scene/recon/    Tier-1 per-session recon (SessionRecon, corridor, types, utils)
herald/scene/refine/   Tier-2 persistent map: align, reconcile, route, portals
herald/scene/common/   shared domain types (SceneGraph/SceneNode, RouteGraph, SourceRef, Frame, ...)
herald/scene/semantic.py   object text-label embeddings
services/              model/integration wrappers (boxer, vggt, osm, vlm, sam2, embeddings)
scripts/               scene_* entry points + SLURM templates
third_party/boxer/     Boxer detector/tracker (pinned git submodule)
```

Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` to run diagnostics without stray system
plugins (e.g. ROS).
