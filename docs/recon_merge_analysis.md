# Recon + Merge — Quantitative Analysis

**Sessions:** Hospital P0000 / P0001 / P0002 (`_v3` dumps) · **Merged map:** `runs/merge_multi3/merged.npz` (3-way)
Sources: object/cloud stats read directly from the `.npz`; Sim3 + reconcile counts from the merge/align job logs.
Generated 2026-08-17.

## 1. Per-session recon (`_v3` dumps that fed the 3-way merge)

| | P0000 | P0001 | P0002 |
|---|---|---|---|
| objects | 121 | 140 | 152 |
| cloud points | 373 k | 398 k | 519 k |
| extent X×Y×Z (m) | 61.5×60.4×18.0 | 61.6×68.8×14.5 | 74.1×74.3×18.4 |
| cloud diagonal | 88.0 m | 93.5 m | 106.6 m |
| support med / mean / max | 9 / 13.4 / 81 | 11 / 14.5 / 84 | 10 / 13.3 / 64 |
| support ≥8 / ≥16 / ≥32 | 70 / 34 / 7 | 99 / 50 / 11 | 96 / 43 / 10 |
| conf med / mean / max | 0.655 / 0.666 / 0.884 | 0.664 / 0.669 / 0.872 | 0.660 / 0.661 / 0.872 |
| OBB diag med / max (m) | 1.92 / 4.12 | 1.64 / 4.48 | 1.96 / 4.39 |
| OBB vol med / max (m³) | 0.42 / 6.1 | 0.46 / 7.5 | 0.43 / 7.3 |
| distinct labels | 26 | 25 | 25 |
| embeddings | 0/121 | 0/140 | 0/152 |

Conf is tightly clustered (0.51 floor from the `conf_thr` gates → ~0.66 median, few above 0.85), consistent across all three sessions — no outlier run. Embeddings are empty in the per-session dumps by design (only the merged map is backfilled).

Top labels per session:
- **P0000:** bench(15), door(15), entrance(14), trash can(12), couch(8), chair(7), tv(7), mirror(6), clock(6), bed(6)
- **P0001:** bench(21), entrance(13), couch(12), trash can(10), tv(7), mirror(7), ventilator(7), door(6), toilet(6), sofa(6)
- **P0002:** door(21), trash can(15), couch(14), entrance(12), clock(10), mirror(9), bench(8), tv(8), sofa(7), chair(6)

## 2. Merged map (`runs/merge_multi3/merged.npz`)

- **307 objects**, **915 k voxels** (0.1 m), extent 79.1×74.3×18.4 m.
- **Provenance:** 226 single-source · 68 two-source · **13 all-three**. Per-source presence: P0000 in 118, P0001 in 134, P0002 in 149 merged objects.
- Support med / mean / max = 13 / 18.5 / **130** (fusion sums support, so shared objects climb).
- **Embeddings: 307/307, dim 384** (`all-MiniLM-L6-v2`, backfilled).
- Top labels: door(33), bench(33), trash can(26), entrance(23), couch(22), mirror(18), chair(15), desk(14), clock(13), ventilator(13), tv(10), bed(8).

## 3. Alignment

### Production (inside the 3-way merge, real session-to-session)

| step | scale | yaw | inliers | fused |
|---|---|---|---|---|
| P0001 → P0000 | 1.0000 | −0.17° | 64.3% | 33 |
| P0002 → P0000 | 1.0000 | 0.04° | 52.1% | 61 |

Scale ≈ 1 and near-zero yaw are expected — the same building from GT-depth recon, so clouds are already gauge-consistent; alignment is mostly cleanup. Inlier fraction ~52–64%.

### Stress-test validation (`test_align`, synthetic known perturbation → recovery error)

A known Sim3 is applied to a cloud, then recovered — this is the real accuracy measurement of the aligner.

| job | perturbation | scale (true 0.9091) | rot err | RMSE | median | p95 | inliers |
|---|---|---|---|---|---|---|---|
| 925195 | yaw 35°, tilt 6/−4°, s1.1 | 0.9090 | 0.83° | 0.175 m | 0.135 m | 0.361 m | 62.1% |
| 925208/11 | yaw 160°, tilt 6/−4° | 0.9045 | 0.86° | 0.230 m | 0.214 m | 0.374 m | 61.9% |
| 925206 | yaw 35°, **tilt 20/−15°** | 0.9125 | 1.08° | 0.257 m | 0.208 m | 0.447 m | 59.5% |

The aligner recovers **scale to <0.5%, rotation to ~1°, cloud RMSE ~0.18–0.26 m** even from large yaw (160°) and heavy tilt, degrading gracefully as tilt grows. (These ran on the earlier P0000/P0001 dumps but validate the same Sim3 algorithm.)

## 4. Reconcile config sweep (2-way P0000 ↔ P0001, older dumps)

| gate | fused | only-A | only-B | persistent |
|---|---|---|---|---|
| centre-dist `d` | 37 | 126 | 147 | 310 |
| IoU > 0.10 | **46** | 117 | 138 | 301 |
| IoU > 0.15 | 35 | 128 | 149 | 312 |
| IoS > 0.2 / cdiag < 0.3 (thr40_50) | 33 | 90 | 113 | 236 |
| IoS > 0.9 / cdiag < 0.5 (unified) | 38 | 91 | 99 | 228 |

Fusion count is fairly flat (33–46) across gates — the ceiling is **detection recall**, not the gate: most objects have no counterpart in the other session (~120–150 only-X each way). Loosening the gate mostly trades precision for a few more fuses.

## Notable observations

- **Label confusion visible in the fuse logs:** repeated `entrance~mirror` and one `toilet~trolley` / `sofa~bench` fuse — OWLv2 open-vocab mislabels. Geometry says they are the same box; labels disagree. The embedding-based `labels_agree` swap (or the deferred vocab cleanup) would tighten this.
- **Per-session conf/support distributions are near-identical** across P0000/1/2 → the recon pipeline is stable run-to-run.
- **Only 13/307 objects seen in all three sessions** — persistent-map coverage is still recall-limited, matching the sweep finding.
