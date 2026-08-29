# Stage 1.4 — Classification

**Module:** `herald/scene/init/classify.py:build_feat`.
**Input:** the hierarchical graph, the `Frame`, and (optionally) a `VLMClient` + aerial/OSM map
imagery.
**Output:** each non-site node gets semantic `attrs` (`role`, `category`, `function`), a `name`,
and a `desc`.

## Mechanism

Each polygon is classified into a structured `EntityClassification(role, category, function, name,
desc)` by one of two paths:

1. **Heuristic (default, `_fallback`).** Rule-based mapping from OSM tags to a role:
   - `building=*` → `role=structure`, `category=building`;
   - `amenity=parking` → `role=facility`, `category=parking`;
   - sports/leisure → `role=facility`, `category=recreation`;
   - landcover / `natural` / `landuse` → `role=region_use` (grass, forest, open ground);
   - fences/barriers → `role=obstacle`; roads/utilities/tanks → `role=infrastructure`;
   - otherwise → `role=unclassified`.
2. **VLM (`--vlm`).** When a `VLMClient` is supplied, each polygon is cropped from the aerial/OSM
   mosaic and sent (image + tag context) to a vision-LLM under a fixed system prompt (`_VLM_SYSTEM`)
   that returns the same structured fields. The heuristic result is the fallback if the VLM fails.

The role/category/function land in `node.attrs` (they are heuristic hard-coded classes, not core
schema fields); `name`/`desc` populate the node directly.

## Design rationale

- Role/category/function are kept in `attrs` deliberately: they drive heuristic downstream logic
  and are OSM-vocabulary-specific, so they don't belong in the generic `SceneNode` schema.
- The heuristic path guarantees a usable label offline and with no model; the VLM path is an
  optional accuracy upgrade that reuses the exact same output contract (`EntityClassification`).
- Classification is separated from geometry finalization: after this stage, `build_scene_graph`
  reprojects every ring to ENU and refreshes `SourceRef.metadata` with filtered context tags.

## Knobs / notes

- `use_vlm` / `vlm_model` (default `ollama:qwen2.5vl:3b`).
- Aerial/OSM mosaic + bbox/zoom feed the VLM crops; without imagery the VLM path is skipped.
