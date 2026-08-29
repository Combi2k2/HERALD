# Stage 2.7 — Rendering

**Script:** `scripts/scene_render.py` · **Output:** a Rerun recording (`scene.rrd`).
**Input:** the persistent map (`persistent.npz`), the area/object `SceneGraph` (`scene_graph.json`),
and optionally the route graph (`route.json`).

This is the **only** place rendering happens — every pipeline stage just writes data to disk, so
the visualization is decoupled from the mechanisms and can be regenerated at any time.

## What it draws

- **Scene cloud** — the persistent coloured point cloud (subsampled to `--max-points`).
- **Objects** — each object OBB as a wireframe box **coloured by its area** (golden-ratio hue from
  the area uid); objects with no area (structural/free) are grey.
- **Portals** — magenta bounding boxes with **thicker edge lines** (`radii`), identified from the
  graph's `attrs["is_portal"]`.
- **Doors** — solid magenta panels (`name == "door"`).
- **Route graph** — grey line strips for kept edges; **portal-cut edges in red**. Cut edges are
  recomputed from the portal gates via the shared `edge_crosses_gate`, so the picture agrees exactly
  with stage 2.5's clustering.
- **Crops + labels** — every object, door, and portal logs its label and up to 3 image crops, shown
  in the selection panel on click.

## Design rationale

- **Data/rendering separation:** pipeline scripts stay headless (cluster-friendly, no viewer
  dependency); the render reads the finished on-disk representation. This is why 2.7 is a stage, not
  a flag on the others.
- Rendering **re-derives** portal cuts from the same `refine/portals.edge_crosses_gate` used in 2.5,
  rather than trusting a stored edge list, guaranteeing the visual and the metric never drift.
- Portals-as-thick-boxes / doors-as-solid-panels make the two kinds of doorway detection visually
  separable at a glance while both reading as "magenta = transition".

## Knobs

`--objects` / `--graph` / `--route` (inputs) · `--out` (`.rrd`) · `--max-points` (cloud subsample).

## View

```bash
uv run --group viewer rerun data/tartanground/<Env>/scene/scene.rrd
```
