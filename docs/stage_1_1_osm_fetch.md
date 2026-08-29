# Stage 1.1 — OSM fetch

**Module:** `services/osm.py` (`OSMClient`), `herald/scene/common/roi.py` (`ROI`),
`herald/scene/common/geometry.py` (`Frame`).
**Input:** a region-of-interest (GPS bbox or browser-picked polygon).
**Output:** raw OSM polygons + highway ways within the ROI, and the local `Frame`.

## Mechanism

1. **ROI → query.** The ROI is a lat/lon polygon (from the browser GPS/ROI picker, or a `--bbox`).
   Its vertices define the Overpass query area.
2. **Overpass fetch.** `OSMClient.query_in_polygon(roi)` pulls two things:
   - **polygons** — every closed way / relation with a footprint (buildings, landuse, natural,
     leisure, amenity, …), each as an `OSMRawPolygon` (WGS ring + tag dict + `osm_type/osm_id`).
   - **highways** — the walkable/road way network, kept for stage 1.5's route graph.
3. **Local frame.** A `Frame` anchored at the ROI centroid provides the WGS↔UTM↔ENU transforms
   used downstream (areas in stage 1.2 are measured in UTM; final geometry is ENU).

## Design rationale

- Fetching once, up front, gives Phase 1 a structurally meaningful scaffold **before** any robot
  data — the offline counterpart to Phase 2's online reconstruction.
- Polygons and highways are fetched together so the semantic graph (stages 1.2–1.4) and the route
  graph (stage 1.5) share exactly one ROI and one coordinate frame.
- Tags are carried verbatim on each polygon's `SourceRef.metadata` so later stages (and audits)
  can trace every node back to its OSM origin.

## Knobs / notes

- The picker and validation UI live in `scripts/scene_init.py` (`--serve`, `--bbox`, `--vlm`).
- Network access is required; a fetched ROI can be replayed from `data/{run_id}/raw/osm.geojson`.
