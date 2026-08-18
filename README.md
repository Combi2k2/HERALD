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

## Phase 2 — RGB(-D) reconstruction (two-tier)

Tier 1 turns one video into a `SessionResult` (static object OBBs + scene cloud); Tier 2
reconciles many `SessionResult`s into one persistent map. Tier 2 consumes only the saved
`SessionResult` dumps — never raw frames.

### Tier 1 — per-session recon

`scripts/session_recon.py` streams frames through the Boxer engine (OWLv2 detect → BoxerNet
3D lift → online 3D tracker), accumulates the scene cloud, drops in-video movers (corridor
filter), and dumps a `SessionResult`.

```bash
# local (GPU):
uv run python scripts/session_recon.py --env Hospital --traj P0000 \
    --dump data/tartanground/Hospital/recon/session_P0000.npz \
    --out runs/session_P0000.rrd
```

On SLURM, use the `scripts/boxer.slurm` template (pick the entry with `SCRIPT`):

```bash
sbatch --export=ALL,SCRIPT=session_recon.py,ENV_NAME=Hospital,TRAJ=P0000,\
DUMP=data/tartanground/Hospital/recon/session_P0000.npz scripts/boxer.slurm
```

Key knobs: `--conf-thr2d` (OWLv2 2D floor, 0.40) · `--conf-thr3d` (BoxerNet 3D floor, 0.50) ·
`--min-obs` (track support, 4) · `--corridor-step`/`--max-range` (dynamic filter).

### Tier 2 — multi-session merge + persistent embeddings

`merge_multi.py` aligns each session's cloud onto the first (energy-based Sim3), reconciles
objects (two-pass gate + union-find), and embeds each merged object's text label into a vector
(vote-weighted mean via a text-only sentence model). Use `merge_objects.py` for a pairwise merge.

```bash
uv run python scripts/merge_multi.py \
    --sessions data/tartanground/Hospital/recon/session_P000{0,1,2}.npz \
    --out runs/merge/merged.npz                        # + merged.ply / merged.rrd

uv run python scripts/merge_objects.py --a session_P0000.npz --b session_P0001.npz \
    --out runs/merge/pair.npz                          # pairwise

uv run python scripts/embed_map.py --map runs/merge/merged.npz   # (re)fill embeddings in place
```

Merged objects render color-coded by source (per session, blended for fused). A merged map is
the same `SceneMap` structure as a single-session recon, so it can be merged again.

### Diagnostics

```bash
uv run python scripts/test_recon.py  --env Hospital --traj P0000 --max-frames 60   # layered 3D/2D panel
uv run python scripts/test_align.py  --p1 A.npz --p2 B.npz   # Sim3 recovery on a known perturbation
uv run python scripts/test_refine.py                        # alignment inspection (synthetic if no --p1/--p2)
```

## Cluster / GPU (SLURM)

Heavy jobs run via `scripts/*.slurm` (`sbatch`, override config with `--export=ALL,KEY=val,...`).
Hard constraints (see `CLAUDE.md`): use `--partition=3090` (not P100), and clear `PYTHONPATH`
(`unset PYTHONPATH` / prefix `PYTHONPATH= uv run ...`) so system site-packages don't shadow the
uv venv. Login nodes have no usable GPU. Job outputs go to `runs/{jobid}/`.

## Repository layout

```
herald/scene/init/     Phase-1 scene-graph builder
herald/scene/recon/    Tier-1 per-session recon (SessionRecon, corridor, types, utils)
herald/scene/refine/   Tier-2 multi-session align + reconcile
herald/scene/common/   shared domain types (SceneGraph, Frame, NavGraph, ...)
herald/scene/semantic.py   object text-label embeddings
services/              model/integration wrappers (boxer, vggt, osm, vlm, sam2, embeddings)
scripts/               entry points + SLURM templates
third_party/boxer/     Boxer detector/tracker (pinned git submodule)
```

Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` to run diagnostics without stray system
plugins (e.g. ROS).
