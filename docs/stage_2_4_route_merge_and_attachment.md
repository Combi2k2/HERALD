# Stage 2.4 — Route-graph merge + attachment

**Script:** `scripts/scene_merge_route.py` · **Modules:** `herald/scene/refine/route.py`
(`build_route`), `herald/scene/common/route.py` (`RouteGraph`).
**Input:** the persistent map, its `route.json` + `attach.json`, the session's poses, `sim3_<T>.json`.
**Output:** `route.json` (persistent route graph) and `attach.json` (object → observing-node),
both updated in place.

## Mechanism

1. **Session route (`build_route`).** The session's camera trajectory (poses, Sim3-transformed into
   the canonical frame) is turned into a route sub-graph: **Taubin smooth** → **arc-length resample
   at 1 m** (`--spacing`) → one `RouteNode` per sample + **sequential edges**. Each node records its
   `session` + sequence index in a `SourceRef`.
2. **Absorb (append-only).** `RouteGraph.absorb` appends the new nodes and edges to the persistent
   graph — **no node is moved or merged**, and ids are stable, so object attachments never remap.
3. **Proximity edges.** `RouteGraph.add_proximity_edges(radius, min_seq_gap)` links any node pair
   within `--r-connect` over the **whole** graph: this adds both *intra-session loop closures* (a
   trajectory revisiting a place joins its two passes) and *cross-session fusion* (parallel
   trajectories link up). A same-session pair whose sequence indices differ by `< min_seq_gap` is
   skipped (already chained along the path); `min_seq_gap = round(r_connect / spacing) + 1`.
4. **Multi-anchor attachment.** For each object the session observed, every observing frame's pose
   maps to its **nearest route node**; that node is added to the object's anchor **set** in
   `attach.json`. An object thus anchors to the set of nav-nodes that actually saw it.

## Design rationale

- **Append-only, snap-free** (unlike Phase-1's junction snapping): merging camera nodes would move
  positions and break the stable-id contract that keeps attachments valid across sessions. Proximity
  edges give loop closure and cross-session connectivity *without* moving anything.
- Doing proximity over the **whole** graph each merge (not just new-vs-old) is what makes it
  idempotent and order-independent — the incremental accumulation lands on the same graph a single
  batch build would.
- A **set** of observing anchors (not one nearest node) is robust to a single bad frame and is the
  geometry stage 2.5's geodesic minimises over.

## Knobs

`--spacing` (1 m nav-node spacing) · `--r-connect` (2 m proximity radius ≈ corridor width).
