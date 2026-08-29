# Stage 1.2 — Seed graph (site + region nodes)

**Module:** `herald/scene/init/pipeline.py:seed_graph`.
**Input:** raw OSM polygons, the ROI, the `Frame`.
**Output:** a `SceneGraph` with one `site` root and one `region` node per admissible polygon.

## Mechanism

1. **Site root.** A single `SceneNode(uid=0, level="site")` whose geometry is the ROI ring (WGS).
   `SITE_UID = 0` is the fixed root; every other node's `uid` counts up from 1.
2. **Region candidates.** For each raw polygon:
   - close the WGS ring and skip degenerate rings (< 4 points);
   - compute its **UTM area** (`Frame.wgs2utm` on each vertex → `shapely` polygon area);
   - keep it only if `utils.osm_filter.polygon_disposition(tags, area, min_area, max_area)` returns
     `"hierarchy"` — i.e. it is a real structural footprint, not too small (noise/cadastre) or too
     large (the whole ROI), and not a tag class that should be dropped.
3. **Region node.** Each kept polygon becomes `SceneNode(level="region", parent=SITE_UID)` with its
   WGS ring geometry and a `SourceRef(assigned_by="osm", assigned_id="<type>/<id>", metadata={tags,
   area})`.

## Design rationale

- Seeding is deliberately **flat** (site → regions) — the parent/child containment is not yet
  known; stage 1.3 discovers it. Keeping seeding and hierarchy separate makes each testable.
- Area gating at seed time removes the dominant OSM noise (tiny cadastre slivers, ROI-sized blobs)
  before the O(n²) containment test, so stage 1.3 works on a clean candidate set.
- Provenance (`SourceRef`) is attached immediately, so a node can always be traced to its OSM id
  and tags regardless of later relabelling.

## Knobs / notes

- `MIN_NODE_AREA` / `MAX_NODE_AREA` (in `init/hierarchy.py`) bound admissible footprints.
- `polygon_disposition` is the single decision point for "is this polygon a structural node?"
