# area-cluster-01 — Area nodes from spatio-semantic clustering

**Stage:** Graph Builder (`site → region/zone → area → object → ego`)
**Status:** first result — promising, not yet promoted

## Hypothesis

Object nodes of a persistent `SceneMap` carry a dual signature `(p, e)` = (`center`,
`embedding`). Clustering them in a **spatio-semantic metric** yields meaningful **area nodes**
(a seating area, a bay), which we then describe and embed.

## Approach

- Distance (`--raw` = slide baseline; `--normalize` on by default):
  `D(i,j) = alpha·‖p_i−p_j‖[/r_max] + (1−alpha)·(1−cos(e_i,e_j))`.
- Cluster the precomputed `D` with **HDBSCAN** (sklearn 1.9, `metric="precomputed"`); DBSCAN
  also available.
- **Acceptance gate:** promote a cluster iff `diam(p) < r_max` **and** incoherence
  `1 − ‖mean(unit e)‖ < eps_coh`.
- Area summary: support-weighted mean embedding + merged label votes (LLM description = hook,
  not built).
- OSM region pre-partition = **no-op here** (TartanGround is synthetic, no OSM contours);
  documented hook for real Phase-1 data.

Files: `scene_io.py` (numpy `.npz` loader — wrapper to avoid the torch chain, rule 4),
`cluster.py` (metric + gate), `run.py` (entry + Rerun viz).

## How to run

```bash
uv run python experiments/area-cluster-01/run.py --map runs/merge_multi3/merged.npz \
    --alpha 0.6 --r-max 4.0 --eps-coh 0.25 --method hdbscan          # --raw for the baseline
```
Report to stdout; Rerun scene → `out/area_cluster_01.rrd` (objects coloured by area).

## Result / verdict

On `runs/merge_multi3/merged.npz` (307 objects, 384-d embeddings), default params:
**86 clusters → 57 accepted areas, 29 rejected, 41 noise.**

Accepted areas are semantically coherent: `tv+couch+sofa` (lounge), `desk+chair+table`
(workstation), `laptop+monitor+computer` (a computer desk), `bed+curtain` (patient bay),
`entrance+door`.

The gate earns its place — rejections are exactly the right ones:
- **`diam ≥ r_max`** catches HDBSCAN chaining *same-category* objects across the building
  (`cand 0`: 16 objs, diam **17.2 m**, trash can/entrance/mirror). This is the flagged failure
  mode: with the semantic term near-binary (label-mean embeddings → same-label cos≈1), objects
  of one category link over long distance. The spatial gate cleans it up post-hoc.
- **incoherence** catches semantically mixed blobs (`chair+tv+curtain`, `trolley+bed+curtain`).

### Takeaways for promotion
1. The **diam gate is essential**, not optional — it compensates for HDBSCAN over-linking a
   category across space. Better still: **region pre-partition** (real Phase-1 data) and/or a
   tighter `r_max`, so those giant clusters never form.
2. `alpha` should stay spatially-leaning while `e` is a label-mean (semantic term is near-binary).
   Re-run with a **richer (appearance) embedding** to let semantics carry more weight.
3. Many 2-object areas — consider `min_cluster ≥ 3` for the promoted default.
4. Next: wire the **LLM description + its embedding** onto accepted areas, and compare
   `--raw` vs `--normalize` head-to-head to quantify the alpha-scaling effect.
