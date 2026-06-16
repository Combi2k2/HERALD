# Discussion: Persistence and Graph Representation

Notes on open design problems for how HERALD stores offline scene data,
separates semantic hierarchy from navigability, and aligns with HOV-SG conventions.

Branch: `SG-offline-init` (as of offline init implementation).

---

## Current persistence layout

`scripts/scene_init.py` writes per run id (default `current`):

**`data/raw/{run_id}/`** — Overpass fetch + ROI

**`data/phase1/{run_id}/`** — scene graph, pathways, metadata, map overlay

| Artifact | Role |
|----------|------|
| `scene_graph.json` | Semantic hierarchy: nodes, containment edges, captions, **inline embeddings** |
| `pathways.geojson` | Walkable OSM highway linestrings |
| `osm_polygons.geojson` | All raw OSM polygons (unfiltered); not wired into graph nodes |
| `roi.geojson` | Query region |
| `metadata.json` | Counts, timing, tag histogram |

There is **no** persisted navigation graph (`nav_graph.graphml`, node-link JSON, etc.).

---

## Problem 1 — Semantic graph and nav graph are not co-designed

**Today:** The scene graph (`SceneGraph`) and pathways live in separate files with no
shared node IDs or cross-references.

- `SceneGraph` is a containment tree: `site → outdoor_region → building`.
- `pathways.geojson` is raw OSM geometry used for (a) outdoor partition barriers and
  (b) map/Rerun visualization only.
- Pathways are buffered and **subtracted** from free space in `partition_outdoor_zones()`,
  so they define outdoor zone *topology* but are not stored as graph edges.

**Gap:** The global planner needs `G_nav = (W, P)` — waypoints plus traversable edges —
with Dijkstra (or similar) over edge weights. That structure is planned in `PLAN.md`
(`herald/planner/nav_graph.py`) but not implemented. Consumers (`scene_view.py`, map
overlay) load pathways as drawables, not as a routable graph.

**Open question:** Should nav waypoints be (1) derived only from pathway polylines,
(2) augmented with Voronoi samples in open outdoor zones, or (3) both with explicit
provenance on each edge?

---

## Problem 2 — Monolithic `scene_graph.json` with inline embeddings

**Today:** Every `SceneNode` carries a full embedding vector in JSON. On a real campus
ROI (~237 nodes) the file is already large; with CLIP at 512–1024 dims and future
`functional_area` / `object` / `ego` levels it will not scale.

**HOV-SG contrast:** Hierarchy metadata in per-node `.json` files; features in separate
`.pt` tensors (`full_feats.pt`, `mask_feats.pt`). Graph JSON stays small.

**Gap:** No sidecar format (`embeddings.npz`, FAISS index path, or similar) and no
stable row index from `node.id → embedding row`.

**Open question:** Split into `scene_graph.json` (topology + captions + geometry refs)
plus `embeddings.npz` keyed by node id, or follow HOV-SG's directory-per-level layout?
For campus-scale outdoor, per-node `.ply` files are likely unnecessary; lat/lon rings
in JSON or GeoJSON sidecars are enough.

---

## Problem 3 — Duplicate OSM data with no single source of truth

**Today:** Three representations of overlapping OSM content:

1. **Filtered** buildings/highways → scene graph nodes + pathways GeoJSON.
2. **Unfiltered** polygons → `osm_polygons.geojson` (478 polygons in IP Paris run).
3. **Partition geometry** → outdoor zone polygons derived algorithmically, not copied
   from OSM `landuse` / `leisure` / `natural` tags.

Tag-rich polygons in (2) are not linked to scene nodes in (1). Outdoor zones are named
`outdoor_003` with area-only captions; building captions use raw tag dumps including
misleading `primary_tag` ordering (`addr:*`, `source`, etc.).

**Gap:** Priority 1b (`herald/scene/osm_tags.py`) is designed but not implemented.
Until then, persisted graph semantics under-use data already on disk.

**Open question:** Should outdoor zones be persisted as (a) derived partition polygons
with semantic labels overlaid, or (b) direct OSM polygon nodes where tags exist, with
partition only filling gaps?

---

## Problem 4 — Schema ahead of usage

**Defined but unused in offline init:**

- `EdgeType`: `"spatial"` and `"traj"` — only `"contains"` is written.
- `SceneGraph.point_cloud_uri` — always null offline.
- `SceneNode` lacks planned fields: `osm_tags`, `semantic_label`, `confidence`.

**Defined in PLAN but absent from persistence:**

- Levels: `functional_area`, `object`, `ego` (online refinement).
- Experience-weighted nav edges (ego trajectory history).

**Risk:** Early clients may assume the JSON schema is stable; adding levels and edge
types later requires versioned migration or breaking changes.

**Open question:** Add a `schema_version` field to `scene_graph.json` and artifact
metadata now, before planner/refinement land?

---

## Problem 5 — Coordinate frames split across artifacts

**Today:**

- Node geometry: WGS84 `(lat, lon)` rings in `scene_graph.json`.
- `LocalFrame`: aeqd ENU origin at ROI centroid (`frame_origin` in graph header).
- Pathways GeoJSON: `[lon, lat]` per RFC 7946.
- Partition: internal UTM for metric ops; converted back to lat/lon for nodes.

Nav graph construction will need a **single metric frame** for edge lengths and
nearest-waypoint queries. Rerun viewer already maps via `LocalFrame`.

**Gap:** No persisted transform manifest beyond `frame_origin`; no convention for
whether nav graph nodes store ENU, WGS84, or both.

**Open question:** Persist nav graph in local ENU (meters, planner-native) with optional
WGS84 denormalization for map UI, or stay GeoJSON-first for community tooling?

---

## Problem 6 — Pathways serve two masters

Walkable highways currently:

1. **Structural:** 3 m buffer (`DEFAULT_PATHWAY_BUFFER_M`) splits outdoor free space.
2. **Nav intent:** Intended backbone for `G_nav` (per `PLAN.md` / osmnx analogy).

Changing pathway filter sets, buffer width, or merging rules **mutates outdoor zone
topology** and therefore the scene graph, not just the nav graph. There is no way to
rebuild nav without re-running partition, or to experiment with nav topology independently.

**Open question:** Decouple partition barriers from nav edges — e.g. persist partition
parameters and allow nav graph rebuild from `pathways.geojson` + ROI without changing
zone IDs, or accept that nav and semantic zone graphs co-evolve?

---

## Comparison with HOV-SG (reference baseline)

| Aspect | HOV-SG | HERALD (current) |
|--------|--------|------------------|
| Semantic levels | site → floor → room → object | site → outdoor_region → building |
| Semantic storage | Directory of `{id}.json` + `.ply` per node | Single `scene_graph.json` |
| Embeddings | In node JSON / `.pt` sidecars | Inline in JSON (stub) |
| Semantic graph in memory | `networkx.Graph` of Python objects | Dataclass lists + string IDs |
| Nav graph | Separate Voronoi `nx` graphs, `node_link_data` JSON | Not persisted; pathways GeoJSON only |
| Nav node attrs | `pos` (x,y,z), `floor_id`; edge `dist` | N/A |
| Query API | `query_hierarchy()` on loaded `Graph` | Not implemented |

HOV-SG keeps **semantic hierarchy** and **actionable nav graph** as two graphs linked
at planning time (target node → nearest nav waypoint → Dijkstra). HERALD should adopt
that separation explicitly rather than overloading `pathways.geojson` as both map layer
and implicit nav graph.

---

## Recommended persistence target (proposal)

Not implemented — for discussion only:

```
data/raw/{run_id}/
data/phase1/{run_id}/
├── scene_graph.json      # topology, captions, geometry, schema_version; no embeddings
├── embeddings.npz        # node_id → vector (or FAISS index path)
├── pathways.geojson      # OSM backbone (immutable OSM snapshot)
├── nav_graph.graphml      # networkx-exported G_nav: pos, length, optional zone_id
├── osm_polygons.geojson  # raw reference layer
├── roi.geojson
└── metadata.json
```

Cross-links:

- Scene node `id` referenced on nav waypoint properties where a waypoint lies inside a zone.
- Nav graph built from pathways + optional Voronoi augment; **versioned independently**
  of outdoor partition when possible.

---

## Relation to `PLAN.md`

| PLAN item | Persistence impact |
|-----------|-------------------|
| Priority 1b (tag-driven init) | Richer node captions + optional `osm_tags` on `SceneNode` |
| Priority 2 (refinement) | New levels + embedding sidecar; checkpoint format |
| Priority 3 (FAISS) | Index file alongside `embeddings.npz` |
| Priority 4 (Voronoi) | Extra edges/nodes in `nav_graph.*` |
| Priority 5 (planner) | Reads scene + nav artifacts; no third graph format |

---

## Open decisions (summary)

1. **Split embeddings** out of `scene_graph.json` before CLIP is wired.
2. **Introduce `nav_graph` artifact** from `pathways.geojson` (Planner P2) with a
   documented node/edge schema (likely networkx → GraphML or node-link JSON).
3. **Add `schema_version`** to graph and metadata.
4. **Resolve partition vs nav coupling** — document whether zone IDs are stable across
   nav-only rebuilds.
5. **Align outdoor semantics** with `osm_polygons.geojson` (Priority 1b) so persisted
   graph nodes are not anonymous while raw tagged data sits unused beside them.
