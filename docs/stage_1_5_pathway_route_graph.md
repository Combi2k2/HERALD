# Stage 1.5 — Pathway route graph

**Module:** `herald/scene/init/pathways.py:build_path`.
**Input:** the OSM highway ways in the ROI, the `Frame`.
**Output:** a `RouteGraph` (nav-node lattice) attached to the `SceneRepr`.

## Mechanism

1. **Keep walkable ways.** Filter highways to `WALKABLE_HIGHWAY_TYPES` (`_is_walkable`).
2. **Densify.** Each walkable linestring is resampled at ≤ `max_node_spacing` between consecutive
   vertices, so no edge spans a long gap; every sample becomes a `RouteNode` and consecutive
   samples form a `RouteEdge` (`source="osm"`).
3. **Snap junctions.** A `_SnapGrid` with cell size = `snap_radius` merges samples closer than
   `snap_radius` into one node, so linestrings that touch at an intersection share a node — this is
   what turns a bag of independent ways into one connected, traversable graph.

## Design rationale

- The Phase-1 route graph is the **outdoor** navigability backbone, in the same `RouteGraph` type
  the Phase-2 route stage (2.4) uses — one graph abstraction across phases.
- Grid-snapping (rather than all-pairs distance) makes junction merging O(n) and gives a stable,
  order-independent node identity.
- Node positions are ENU (metric, local), matching the finalized semantic-graph geometry so the two
  graphs live in one coordinate frame.

## Knobs / notes

- `max_node_spacing` — resample density along a way.
- `snap_radius` (default 2 m) — junction-merge radius; also the grid cell size.
- Contrast with stage 2.4: Phase-1 route nodes come from OSM ways and *are* snap-merged; Phase-2
  route nodes come from camera trajectories and are **append-only** (never merged), linked by
  proximity edges instead.
