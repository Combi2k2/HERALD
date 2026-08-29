# Stage 2.5 — Area clustering

**Script:** `scripts/scene_area_cluster.py` · **Module:** `herald/scene/refine/portals.py`
(`edge_crosses_gate`).
**Input:** the persistent map, `route.json`, `attach.json`.
**Output:** `scene_graph.json` — one `area` `SceneNode` per cluster, parenting its member objects.

Groups objects into **portal-bounded areas** on a route-graph geodesic proxy.

## Mechanism

1. **Mirror → portal disambiguation.** OWL can't geometrically separate portal / door / mirror (all
   thin flat panels). A `mirror` becomes a **portal** if the camera **walked through its gate** (a
   route edge crosses it — you can't walk through a mirror) **or** it sits within `--mirror-adj`
   (2 m) of a `door`. Native `entrance` objects are portals too. Portals are marked
   `attrs["is_portal"]` for the renderer.
2. **Cut portal-crossing route edges.** A route edge is cut if its segment passes through any
   portal's **gate**: it crosses the panel plane (normal = the OBB's thin axis) within the in-plane
   rectangle + `--gate-margin`. Cutting these splits the route graph into portal-bounded components.
3. **Object↔object distance (offset geodesic).** Each object is a node joined to each of its
   observing nav-nodes (from `attach.json`) by an **offset edge** of length = the centre→node
   distance (the viewing depth). The distance is the plain Dijkstra shortest path over
   `{objects ∪ nav-nodes}` on the **cut** graph:

   ```
   d(i,j) = min over anchors (a∈set_i, b∈set_j)  [ ‖C_i − pos_a‖ + geodesic(a,b) + ‖C_j − pos_b‖ ]
   ```

   An offset hop whose centre→node segment crosses a portal is **∞** (an object seen *through* a
   doorway can't anchor across it); a geodesic across portal components is **∞**. `∞ → BIG`.
   **No Euclidean floor** — the offset edges already give same-node objects a real
   `d(o₁,n)+d(o₂,n)` separation, so no special case is needed.
4. **Cluster.** `AgglomerativeClustering(metric="precomputed", distance_threshold=d_max,
   linkage=average)` on that matrix; `--d-max` (10 m) caps area size. Structural labels
   (`door,entrance,mirror`, `--drop-labels`) are excluded from the clustered set (still emitted as
   objects). Each cluster becomes an `area` `SceneNode` (children = member uids, geom = enclosing
   OBB, `attrs["diameter"]`); members get `parent = area.uid` and a `contains` edge.

## Design rationale

- **Geodesic, not Euclidean**, so a wall/portal detour separates objects that are close in straight
  line but far to walk between — the definition of a distinct area.
- Portals partition **weakly** by design (proximity edges can skirt a doorway through the adjacent
  wall, so the biggest route component stays large) — that's expected; area compactness is enforced
  by `d_max`, portals add the hard cross-doorway ∞ where a gate is actually detected.
- The offset edges are the **principled** object↔object model (a graph shortest path), replacing an
  earlier Euclidean-floor shortcut. They add viewing depth (~2.4 m/end), so `d_max` is set to 10 to
  keep areas at room scale.

## Knobs

`--d-max` (10 m) · `--linkage` (average) · `--mirror-adj` (2 m) · `--gate-margin` (0.3 m) ·
`--drop-labels` (`door,entrance,mirror`).
