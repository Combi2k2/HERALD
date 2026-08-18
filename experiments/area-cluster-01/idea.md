# area-cluster-01 — Area nodes from spatio-semantic clustering

**Stage:** Graph Builder (`site → region/zone → area → object → ego`)
**Status:** planned — not started

## Hypothesis

Object nodes of a persistent `SceneMap` carry a dual signature `(p, e)` = (`center`,
`embedding`). Clustering them in a **normalized spatio-semantic metric** yields meaningful
**area nodes** (e.g. a seating area, a bay), which we then describe and embed.

## Approach (to be refined before coding)

- Distance (scale-normalized so `alpha` is meaningful):
  `D(i,j) = alpha · ‖p_i−p_j‖ / r_max + (1−alpha) · ½(1−cos(e_i,e_j))`.
- Cluster with HDBSCAN on the precomputed `D` matrix.
- Pre-partition objects by OSM region (point-in-polygon) so clusters never span regions.
- Acceptance gate: promote a cluster only if `diam(p) < r_max` **and** embedding incoherence
  `1 − ‖mean(e)‖ < eps`.
- Once a cluster is accepted: LLM description → store text + its embedding on the area node.

## Wrappers (rule 4)

Read `SceneMap` via the existing `refine.load_session`; area nodes are assembled in this folder
(no edits to `herald/scene/common.SceneNode` until validated).

## How to run

_TBD._

## Result / verdict

_TBD._
