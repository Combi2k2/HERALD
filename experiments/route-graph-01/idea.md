# route-graph-01 — Route graph from the camera trajectory

**Stage:** Route Graph (nav-node backbone for routing *and* for the area-clustering geodesic)
**Status:** working (trajectory half, single session)

## Hypothesis

The camera only travels through navigable free space, so a graph sampled along its trajectory is
a wall-safe free-space skeleton. Snapping nearby samples turns the temporally-ordered path into a
spatial **connectivity** graph (loop closure), giving geodesic distances that respect walls and
behave under loops.

## Approach (a "simple feature")

Camera positions -> **Taubin smooth** (shrink-free; pure Laplacian would round corners into walls)
-> **arc-length resample** at fixed `spacing` (uniform edges) -> **voxel-snap** within `r_snap`
(merge co-located samples = loop closure) -> edges = **sequential-along-path + snap-merges** (no
unchecked kNN edges that could jump a wall). Distances via `scipy` Dijkstra.

Files: `traj_io.py` (pose-file reader, numpy), `route_types.py` (prototype **`RouteGraph`** —
`RouteNode`/`RouteEdge`, `geodesic`/`shortest_path`/`nearest`/`merge`/JSON io, per
`design-route-graph.md`), `route_graph.py` (construction pipeline → emits a `RouteGraph`), `run.py`
(entry + Rerun viz + JSON emit). Only OSM-derived nav-nodes and multi-session `merge` are left out
(see below).

`build()` now returns a `route_types.RouteGraph`: nodes `type="nav"`, `source="trajectory"`, 3D
`pos`, `refs=[{session}]`, `radius=r_snap`; serialised to `out/route_graph_01.json`. Verified
round-trip + `shortest_path` (t0→t200 = 32.8 m) + `nearest`.

## How to run

```bash
uv run python experiments/route-graph-01/run.py --env Hospital --traj P0000 \
    --spacing 1.0 --r-snap 0.8 --smooth-iters 10 \
    --cloud data/tartanground/Hospital/recon/session_P0000_v3.npz
```
Report to stdout; Rerun scene → `out/route_graph_01.rrd` (raw + smoothed trajectory, nav-nodes,
edges, nodes coloured by geodesic distance from node 0).

## Result

P0000 `pose_lcam_front.txt`: 2012 poses / 292 m path, defaults `spacing=1.0 r_snap=0.8`:
**261 nav-nodes, 274 edges, 1 connected component**, mean edge 0.99 m.
288 resampled points → 261 nodes: **27 loop/overlap closures**, and 14 edges beyond a pure chain
= real cycles → the graph reflects the building's connectivity, not a temporal line. Geodesic
(Dijkstra from node 0) colours smoothly along corridors.

## Next / notes

- **OSM half** (warm-start nav-nodes from path/road contours) not built — indoor Hospital has no
  OSM; add for outdoor/site scale, stitched to the trajectory at doorways.
- **Multi-session**: fuse P0001/P0002 trajectories via the Sim3 (snapping merges the overlap).
- **Consumer**: this graph is the geodesic backbone for area clustering — `d(obj_i,obj_j)` =
  Dijkstra between the objects' observing nav-nodes (`frame_id` refs), replacing Euclidean in the
  spatio-semantic metric.
- `r_snap ≈ spacing` keeps loops closing without merging along-corridor neighbours.
