# Stage 1.3 — Containment hierarchy

**Module:** `herald/scene/init/hierarchy.py:build_tree`.
**Input:** the flat seed graph (site + region nodes) and the `Frame`.
**Output:** the same graph with `parent` links + `contains` edges reflecting spatial nesting.

## Mechanism

1. **Project to metric.** Each region ring is converted to a UTM `shapely` polygon (areas and
   intersections must be metric, not degrees).
2. **Best-parent search.** For every polygon, scan the candidate parents and pick the **smallest
   polygon that geometrically contains it**: a parent qualifies when
   `parent ∩ poly.area ≥ containment_ratio · poly.area`, and among qualifiers the one with the
   smallest area wins (the tightest enclosing footprint).
3. **Rewire.** The chosen parent becomes `node.parent`; a `contains` `SceneEdge(parent → child)` is
   added. Polygons with no qualifying parent stay children of the `site` root.

## Design rationale

- "Smallest containing polygon" yields the natural nesting (a room inside a wing inside a building)
  without needing OSM to declare hierarchy — OSM footprints are flat and overlapping.
- The `containment_ratio` (< 1) tolerates imperfect OSM geometry: a child slightly poking outside
  its parent still nests, instead of being orphaned to the site root.
- Metric projection avoids the latitude-dependent distortion of comparing areas in raw lat/lon.

## Knobs / notes

- `MIN_NODE_AREA = 30 m²`, `MAX_NODE_AREA = 20 000 m²` bound what stage 1.2 admitted.
- `containment_ratio` is the single tolerance for "is A inside B?".
- Deeper levels (`structure`/`floor`/`space`) are represented via `NodeLevel`; this stage builds
  the OSM-derivable outdoor nesting, and classification (1.4) refines the level/role labels.
