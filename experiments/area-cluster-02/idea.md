# area-cluster-02 — spatial area nodes via route-graph geodesic + ratio separability

**Stage:** Graph Builder (area nodes) · **Status:** working (single session P0000)

## Hypothesis (the converged design)

An area is a **navigably-compact** group of objects. Cluster on the **route-graph geodesic** (not
Euclidean), drop the embedding term entirely, guarantee full coverage (no noise), and separate
across walls using the **geodesic/Euclidean ratio** (tortuosity).

## Method

- **Attach** each object to its **observing** camera node: among the frames that saw it, take the
  pose of closest approach, then its nearest nav-node (route graph from `route-graph-01`).
- **Object distance = graph geodesic between anchor nav-nodes.** (See correction 1.)
- **Separability** `sep(i,j) = geodesic/Euclidean`. ≈1 → open connected space; ≫1 → a wall detours
  the path.
- **Cluster** = connected components of `link iff Euclid < d_link AND sep < rho_max`. No noise —
  every object lands in an area; singletons allowed.
- **No embedding** in the distance or gate (kept only as a future descriptor).

Files: `run.py` (imports `route-graph-01` + `area-cluster-01` loaders; no core edits).

## Result (P0000, 121 objects, `d_link=4 rho_max=1.8`)

**42 areas, 18 singletons, largest 13 — full coverage.** The ratio adds **+21** areas over a
Euclidean-only baseline (21) by cutting wall-separated neighbours (`trash can~bench` 3.4 m Euclid →
**146 m** geodesic = different rooms). `rho_max` sweep is graceful: 1.3→5.0 gives 46→33 areas.

## Two design corrections the experiment forced

1. **Drop the offset term.** The user's original `d = off_i + graph(n_i,n_j) + off_j` **overcounts**:
   the object→node offset is *viewing depth* (perpendicular to the traversal path), so two touching
   objects seen from 7 m away read as ~14 m apart → ratio explodes → everything becomes a singleton
   (118 areas / 115 singletons). Fix: object distance = **anchor-node graph geodesic only**; offset
   is not on the path.
2. **Observing-pose attachment, not nearest-center.** Nearest-center attaches objects to the wrong
   trajectory branch (parallel corridors); the observing camera pose is a guaranteed-reachable
   anchor.

## Known limitation → next

**Attachment noise dominates the residual errors.** Objects sit a median **3.3 m** off the path
(viewing depth), so a single anchor node is displaced and some co-located objects get spuriously
split (`bench~trash can` 0.9 m → 13 m geodesic, ratio 14). Mitigations to try:
- attach to a **set** of observing nodes and take `min` over node pairs (robust to one bad frame),
- or project the object onto the nearest free-space nav-node rather than a single observing pose,
- or densify the route graph (smaller `spacing`).

Also pending: multi-session (needs Sim3-fused pose graph), and area descriptor/label (post-hoc).

## Update — diameter-bounded clustering + tuned recon

**Connected-components chains (single-linkage), so cluster diameter is unbounded** (a 40 m corridor
became one 33-object area, geodesic diameter 19 m). Fix: **complete-linkage on the geodesic with a
diameter cap `d_max`** — a cluster forms only while *all* pairwise geodesics stay < `d_max`, so every
area's geodesic diameter <= `d_max` by construction. Wall separation stays automatic (geodesic across
a wall is large), so the ratio is optional (`--ratio-cut`). One length-scale knob `d_max` replaces
`d_link`/`rho_max`.

**Tuned recon** (`session_P0000_frustum5.npz`: max-range 5 m, conf 0.30/0.40, min-obs 3) vs the old
15 m / 0.40 / 0.50 / 4 dump: objects 121->186 (recall up), attach-offset p90 6.46->4.16 m (proxy
error down 36% — the frustum cap tightens the distance proxy exactly as predicted), singletons down.

Complete-linkage on the tuned dump (186 objects), `d_max` sweep — diameter is capped exactly:

| method | areas | singletons | largest | max geo diameter |
|---|---|---|---|---|
| ratio-CC (chaining) | 26 | 7 | 33 | 19.0 m |
| complete d_max=6 | 35 | 3 | 14 | 6.0 m |
| **complete d_max=8** | 27 | 1 | 17 | **8.0 m** |
| complete d_max=12 | 21 | 0 | 25 | 10.1 m |

Current best: **complete-linkage geodesic, `d_max=8`** on the frustum5 recon -> 27 areas, 1 singleton,
diameter <= 8 m. `run.py --method complete --d-max 8` (rrd: `out/area_cluster_02_complete.rrd`).

Open: `d_max` is a single global scale (a big hall vs a small room want different caps) -> the
clearance/room-type idea would adapt it. Doors still cluster as objects (should become portals).
