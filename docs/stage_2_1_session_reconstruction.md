# Stage 2.1 — Per-session reconstruction

**Script:** `scripts/scene_recon.py` · **Module:** `herald/scene/recon/` (`SessionRecon`),
`services/boxer.py` (`BoxerPipeline`). **GPU.**
**Input:** one TartanGround video (RGB + depth + poses).
**Output:** a `SessionResult` dump (`session_<T>.npz`): static object OBBs + a coloured scene cloud.

## Mechanism

`SessionRecon.add_frame` streams each frame through the Boxer engine and accumulates a cloud;
`finalize()` returns the `SessionResult`.

1. **Boxer engine (per frame).** OWLv2 open-vocab **detect** → BoxerNet **3D lift** (labels stamped
   so the tracker's semantic merge can fire; `conf = (score2d + score3d)/2`) → Boxer's online
   `BoundingBox3DTracker` (Hungarian 3D-IoU match on *visible* tracks + occlusion-aware aging +
   semantic-gated duplicate merge). Returns per-frame track association `{match_tids, match_boxes,
   match_centers}`; `objects()` returns fused OBBs in NED.
2. **Scene cloud.** Frame depth is unprojected and voxel-accumulated (`utils.SceneCloud`). Two gates
   keep the cloud static and clean: the **corridor filter** cuts a moving track's 2D box from the
   unprojection (see below), and `--max-range` drops far, low-accuracy depth.
3. **Visibility gate (`--min-visible`).** A detection whose 3D OBB reprojects with too little of its
   area inside the frame (truncated at the image edge) is dropped — a precision guard against
   half-seen boxes with unreliable lifts.
4. **Existence gating.** An object survives only if it clears the per-frame `--conf-thr2d` /
   `--conf-thr3d` floors and accrues `--min-obs` track observations. There is **no** lifetime
   confidence filter — the fused `conf` is carried only as an informational attribute.
5. **Corridor filter (dynamic removal).** A Boxer track *is* a spatio-temporal corridor;
   `corridor.span` is the AABB diagonal of its world centroids. `corridor_span` drives two effects:
   *online* (once a track's span crosses it, its box is cut from the cloud so a mover never smears
   the static map — span only grows, no flip-back) and *finalize* (`classify_static` drops movers,
   gated by `dyn_min_support`). Catches **in-video** movers only.
6. **Finalize order.** `suppress_contained` (drop OBBs nested ≥ `contain_thr` inside a larger one —
   part-of-a-whole boxes) → `classify_static` (corridor split into static objects + dynamic).

## Design rationale

- The `SessionResult` is a **firewall**: every later stage consumes it, never raw frames, so Tier-2
  logic stays dataset-agnostic and cheap to re-run.
- Boxer stays a **generic** engine — no dynamic logic inside it; all dynamic/precision reasoning
  (corridor, visibility, containment) lives in `recon`, where it can be tuned per dataset.
- Per-session recon and a merged multi-session map are the **same** `SceneMap` structure, so a
  session can be merged, and a merged map re-merged, with identical code.

## Knobs

`--conf-thr2d` (0.35–0.40) · `--conf-thr3d` (0.45–0.50) · `--min-obs` (4) · `--max-range` ·
`--min-visible` (box-visibility fraction) · `--corridor-step` · `--scene-voxel` / `--scene-stride`.
