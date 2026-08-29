# Stage 2.2 — Session alignment (Sim3)

**Script:** `scripts/scene_align.py` · **Module:** `herald/scene/refine/align.py`.
**Input:** the new session's `SessionResult`, the persistent map (if any).
**Output:** `sim3_<T>.json` — the similarity transform that puts the session in the canonical frame.

## Mechanism

Registers the session cloud onto the canonical (persistent) cloud by an **energy-based Sim3**:

1. **Ground plane (RANSAC).** `fit_ground_plane` finds the dominant horizontal plane in each cloud;
   aligning the two grounds fixes roll, pitch, and vertical offset cheaply and robustly.
2. **Scale / yaw / translation.** With the grounds levelled, the remaining degrees of freedom are
   in-plane: a yaw sweep (`yaw_seeds`) × translation, scored by a point-to-point energy field over
   subsampled points, refined for `iters` steps → the best `Sim3(s, R, t)`.
3. **Output.** The transform plus its inlier fraction is written to `sim3_<T>.json`.

**`--no-align`** short-circuits to an **identity** Sim3. TartanGround provides global,
per-environment poses, so its sessions are already co-registered (measured Sim3 ≈ identity: < 0.1°,
< 11 cm); running RANSAC would only add noise and cost. Real, independently-originated sessions drop
the flag and let alignment recover the true Sim3.

## Design rationale

- **Ground-first** decomposition turns a 7-DoF Sim3 search into a small in-plane search — far more
  robust than raw ICP on partial, differently-scaled clouds.
- Alignment is its own stage (not folded into reconciliation) so the transform is an inspectable
  artifact and so `--no-align` can bypass it wholesale for pre-registered datasets.
- Only the **cloud** is used for registration; object reconciliation (2.3) then applies the frozen
  Sim3 — geometry drives the gauge, semantics never leak into it.

## Knobs

`--no-align` (identity for pre-registered data) · `--persistent` (canonical target; identity if
absent, i.e. the first session defines the frame) · alignment internals (`yaw_seeds`, `iters`) live
in `align()`.
