# area-cluster-03 — portal-bounded areas

Areas are separated by **portals** (spatial transitions), not just by geodesic distance.
Builds on area-cluster-02 (route-graph geodesic + multi-anchor node-sets); adds a barrier layer.

## Vocabulary change (upstream)
`DEFAULT_VOCAB` renames `entrance` → `portal`. Rationale: a doorway is a *spatial transition*
that reads the same whether the camera is inside or outside, so "entrance" mislabels it when
the camera is on the far side; "door" is a *physical object* (the panel) and stays a normal
object. Needs a detection re-run (tag `portal5`).

## Mirror → portal disambiguation
OWL can't geometrically separate portal / mirror / door — all three lift to the same thin flat
vertical panel (~one axis 0.07 m, two axes ~0.8 m half). Context rule (user's call, "near door
→ portal"): a `mirror` detection whose OBB is adjacent to a `door` object is re-classified
`portal` — a flat panel next to a door is a real doorway, not a wall mirror. Isolated mirrors
stay mirrors (and, like doors, are dropped from the clustered object set).

## Portal barrier (this experiment's core)
A portal cuts the route graph. Concretely, a route edge (segment between two consecutive
nav-nodes = a stretch the camera walked) is **cut** if it passes through a portal's "gate":
the segment crosses the portal's plane (normal = the OBB's thin axis) *within* the panel's
in-plane rectangle (+margin). Cutting portal-crossing edges splits the route graph into
portal-bounded components; the node-set geodesic then returns **∞ between components**, so two
objects reachable only by crossing a portal get infinite pairwise distance and can never land in
the same area. Objects in the same portal-bounded region keep their normal geodesic. (This is
the "camera path crosses a portal" criterion. The second criterion the user named —
"line-of-sight camera→object crosses a portal", which fixes objects observed *through* a doorway
attaching to the wrong side — is a v2 refinement, noted below.)

Barrier set: **portal only** (user's call). Doors stay object-labeled and do NOT cut.

## Pipeline
recon `portal5` → merge (+in-merge overlap-dedup) → route-merge (node-sets) →
`portal_cluster.py`: mirror→portal disambiguation → cut portal-crossing route edges →
node-set geodesic on the cut graph (∞ across portals) → agglomerative areas (avg, d_max).

## Open / v2
- **Line-of-sight attachment**: an object seen through a doorway currently attaches to the
  observing camera's node (wrong side). Fix: drop an observing frame whose camera→object ray
  crosses a portal, so an object anchors only to nodes on its own side.
- **Gate margin & adjacency thresholds** are eyeballed; no ground truth yet.
- Portal *detection* recall now matters (portals are load-bearing). If OWL's "portal" prompt
  under-detects vs "entrance"/"doorway", decouple detection-prompt from output-label.
