# Design — unified `RouteGraph`

A single routing graph shared by **Phase-1 (OSM warm start)** and **Phase-2 (exploration)**, and
the backbone for the area-clustering geodesic. This extends the existing `herald/scene/common/nav.py`
(`NavGraph`) rather than replacing it — that type already anticipated both sources
(`NavEdge.source ∈ {"osm","trajectory"}`). Not yet promoted to core; pinned here first.

## Rename (from the current `nav.py`)

| current | unified |
|---|---|
| `NavGraph` | `RouteGraph` |
| `NavNode` | `RouteNode` |
| `NavEdge` | `RouteEdge` |
| `NavNodeType = "waypoint" \| "portal"` | `RouteNodeType = "nav" \| "poi"` |
| `NavEdgeSource = "osm" \| "trajectory"` | `RouteEdgeSource = "osm" \| "trajectory"` (unchanged) |

- **Drop `"portal"`.** It is declared but never constructed (only `pathways.py:136` makes nodes,
  as `waypoint`). Portal-ness is a *transition* property of an **edge** (an edge whose endpoints
  lie in different areas), and is derivable from the hierarchy — so it needs no node type. If a
  portal ever becomes first-class (lockable door, elevator with a floor set, named entrance with
  its own semantics), reintroduce it deliberately as a node/object with real fields.
- **`"nav"` vs `"poi"` = transit vs place.** A `nav` node is a point you pass through for
  path-following; a `poi` node anchors an actual place (an area) via `poi_ref`.

## Types

```python
RouteNodeType   = Literal["nav", "poi"]
RouteNodeSource = Literal["osm", "trajectory", "derived"]
RouteEdgeSource = Literal["osm", "trajectory"]

@dataclass
class RouteNode:
    id: str
    pos: tuple[float, float, float]        # 3D ENU, site frame (z=0 for planar OSM; z carries floor)
    type: RouteNodeType = "nav"
    source: RouteNodeSource = "osm"        # where this node came from (warm start vs exploration)
    refs: list[SourceRef] = field(default_factory=list)   # provenance: OSM way | (session, frame_ids)
    radius: float | None = None            # snap / coverage radius (from voxel-snap)
    poi_ref: str | None = None             # set iff type == "poi": the area SceneNode id it anchors

@dataclass
class RouteEdge:
    source_id: str
    target_id: str
    length: float                          # metric length (m)
    weight: float | None = None            # optional routing cost (defaults to length)
    source: RouteEdgeSource = "osm"

@dataclass
class RouteGraph:
    nodes: list[RouteNode]
    edges: list[RouteEdge]
    # existing: add_node/add_edge/get_node/to_dict/from_dict/to_json/from_json
    # ADD: nearest(pos) · geodesic(src) / shortest_path(a,b) [scipy Dijkstra] · merge(other)
```

### Design notes
- **3D `pos`** (was 2D) makes it multi-floor- and multi-session-ready; OSM nodes use `z=0`.
- **`type` is nearly derivable** from `poi_ref` (poi iff `poi_ref is not None`); keep it explicit
  for readability, enforce the invariant in a constructor.
- All `pos` live in **one frame** (site ENU of the `SceneRepr`); trajectory nodes are brought in
  via the Tier-2 Sim3 before insertion.

## Two feeders, one graph

- **OSM warm start** → `source="osm"` nodes/edges from path/road contours (planar, `z=0`),
  available immediately after Phase-1.
- **Exploration** → `source="trajectory"` nodes from the camera path (route-graph-01 pipeline:
  Taubin smooth → arc-length resample → voxel-snap → sequential+snap edges).
- **`merge(other)`** snaps a new graph's nodes into the existing one by `radius` (same voxel-snap
  as loop closure), unions `refs`, and stitches the two sources at doorways. This realises the
  "usable at warm start, denser as it explores" behaviour, and handles multi-session fusion.

## Attachment layer (the glue)

1. **Area ↔ POI.** Each area `SceneNode` (`graph.py`, `type="area"`) registers a
   `RouteNode(type="poi", poi_ref=<area id>)` at its centroid, edged to its nearest `nav` nodes.
   "Area = POI node in the route graph" is then literal, and routing search runs over one graph.
2. **Object ↔ node.** `SceneObject` gains one field — `route_node: str | None` — the id of the
   nav node it anchors to (nearest observing pose, from its `frame_id` refs). The area-clustering
   geodesic becomes `route_graph.shortest_path(obj_i.route_node, obj_j.route_node)`.

## Backward compatibility & migration

- Fully compatible with Phase-1 JSON: new `RouteNode` fields (`source`, `radius`, `poi_ref`) and
  the 3rd `pos` component default; loaders tolerate 2D `pos` (append `z=0`).
- Core touch points to update on promotion: `herald/scene/common/nav.py` (types + module
  docstring), `herald/scene/init/pathways.py:136` (constructor call), and one field on
  `SceneObject` (`route_node`). Small, no data migration.

## Open decisions (defaults chosen)

- Transit type name: **`"nav"`** (user pick) — `"waypoint"`/`"transit"` are alternatives that
  don't echo the dropped "nav" graph name.
- Object anchor: **one `route_node` id on `SceneObject`** (vs a separate object→node map).
- 3D `pos` (vs 2D + optional z): **3D**, for multi-floor.

## Status

Design pinned. Not in core. Next: refactor `route-graph-01`'s ad-hoc numpy graph to emit a real
`RouteGraph` (via a wrapper, no core edits) to validate the structure, then promote to
`herald/scene/common/route.py`.
