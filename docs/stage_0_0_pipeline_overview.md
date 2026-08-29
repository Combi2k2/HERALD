# Stage 0.0 — Pipeline overview

HERALD builds and refines one hierarchical scene graph — `site → region/zone → area → object →
ego` — in two phases. Each stage below has its own `stage_x_y_name.md` design note; `x` is the
main phase, `y` the minor stage in run order.

## Phase 1 — Offline scene-graph init (`herald/scene/init/`, `scripts/scene_init.py`)

Turns OpenStreetMap polygons into a `SceneRepr` (semantic graph + route graph) before any robot
exploration. One script (`scene_init.py`) runs `build_scene_graph()`, whose internal steps are:

| stage | name | module |
|-------|------|--------|
| 1.1 | OSM fetch | `services/osm.py`, `scene/common/roi.py` |
| 1.2 | Seed graph (site + region nodes) | `init/pipeline.py:seed_graph` |
| 1.3 | Containment hierarchy | `init/hierarchy.py:build_tree` |
| 1.4 | Classification | `init/classify.py:build_feat` |
| 1.5 | Pathway route graph | `init/pathways.py:build_path` |

## Phase 2 — Online RGB(-D) reconstruction → persistent map (`herald/scene/recon/` + `refine/`)

Each session's frames become a `SessionResult` (Tier 1), then sessions are folded one-by-one into
a single **persistent `SceneMap` on disk** (Tier 2) and objects are clustered into portal-bounded
areas. Tier 2 consumes only the saved `SessionResult` dumps — never raw frames. Each stage is its
own `scene_*` script that only writes files; rendering is separate.

| stage | name | script |
|-------|------|--------|
| 2.1 | Per-session reconstruction | `scripts/scene_recon.py` |
| 2.2 | Session alignment (Sim3) | `scripts/scene_align.py` |
| 2.3 | Object reconciliation | `scripts/scene_track_obj.py` |
| 2.4 | Route-graph merge + attachment | `scripts/scene_merge_route.py` |
| 2.5 | Area clustering | `scripts/scene_area_cluster.py` |
| 2.6 | Area captioning | `scripts/scene_area_classify.py` |
| 2.7 | Rendering | `scripts/scene_render.py` |

Per session `T`, stages 2.1→2.4 run in order; 2.5→2.6 run once after all sessions; 2.7 renders on
demand. The whole flow is `scripts/scene_pipeline.slurm`.

### Persistent representation on disk (`data/tartanground/<Env>/scene/`)

```
persistent.npz     accumulating SceneMap (cloud + object OBBs)   [2.3 writes]
route.json         persistent route graph (nav-node lattice)     [2.4 writes]
attach.json        object -> observing-nav-node attachment        [2.4 writes]
scene_graph.json   area/object SceneGraph                        [2.5, 2.6 write]
sim3_<T>.json      per-session Sim3                              [2.2 writes]
scene.rrd          Rerun recording                               [2.7 writes]
```

## Shared data model (`herald/scene/common/`)

- `SceneNode` — int `uid`, `level`, `parent`/`children`, a `Geometry` (polygon/point/OBB), label
  `votes`/`name`/`desc`, evidence `supp`/`conf`, `refs` (`SourceRef` per contributing source),
  embedding refs, and free-form `attrs`. `SceneObject` (recon) is a thin `SceneNode` subclass.
- `RouteGraph` — persistent node+edge lattice; `RouteNode` carries a `Vec3` position and a
  `SourceRef` (session + sequence index). See stage 2.4.
- `SourceRef` — one provenance record: `assigned_by`, `assigned_id`, `supp`, `conf`, `metadata`.
