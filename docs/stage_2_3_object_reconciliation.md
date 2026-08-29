# Stage 2.3 — Object reconciliation

**Script:** `scripts/scene_track_obj.py` · **Module:** `herald/scene/refine/reconcile.py`.
**Input:** the new session's `SessionResult`, the persistent map, `sim3_<T>.json`.
**Output:** the persistent `SceneMap` (`persistent.npz`) **updated in place** — objects + cloud.

## Mechanism

1. **Transform.** Apply the stage-2.2 Sim3 to every new object (`transform_object`) and to the
   session cloud, bringing them into the canonical frame.
2. **Reconcile (`reconcile`, two-pass gate).** Each canonical–new object pair is tested for fusion:
   - **pass 1** — `IoS ≥ ios_trust` (`--ios-trust`, default 0.9): strong 3D overlap fuses
     regardless of label;
   - **pass 2** — `cdiag < cdiag_gate` (`--cdiag-gate`, centre-distance / diagonal co-location)
     **and** `labels_agree`.
   A **union-find** collapses each connected component of mergeable objects into one via
   `_fuse_many`: centre is confidence-averaged; **orientation + extent are taken from the lead box**
   (never averaged — averaging extents inflates thin objects); support is summed; provenance is
   unioned. Fused/only-canonical objects keep the canonical `uid`; only-new objects get fresh uids.
3. **Final dedup (`overlap_merge`).** A label-agnostic pass (`--overlap-ios`) collapses any
   remaining heavily-overlapping duplicates within the merged set.
4. **Cloud accumulation.** The transformed session cloud is voxel-merged (`--voxel`) into the
   persistent cloud (`utils.SceneCloud`).
5. **Persist.** `save_session` writes the updated `SceneMap` back to `persistent.npz` in place.

## Design rationale

- **Persistent map = one accumulating `SceneMap` on disk.** Each session's run mutates it; canonical
  uids stay stable across visits (so anything keyed on a `uid` — e.g. attachments in 2.4 — never
  remaps), and new objects get fresh uids.
- The **two-pass gate** separates "obviously the same thing" (geometry-only, pass 1) from "same
  place *and* same kind" (pass 2), so a high-overlap re-detection fuses even if labels disagree,
  while merely-nearby different objects don't.
- `labels_agree` is an **isolated swap point** — currently label-vote overlap, intended to become
  embedding similarity — without touching the gate structure.
- **Evidence-gated retire** (dropping a canonical object a new visit proves is gone, via frustum +
  free-space, not a timeout — the fix for Boxer's ghost boxes) is **not yet built**.

## Knobs

`--ios-trust` (0.9) · `--cdiag-gate` (0.5) · `--overlap-ios` (0.5, 0 disables) · `--voxel` (0.1 m).
