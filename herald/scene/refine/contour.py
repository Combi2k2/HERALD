"""The "anchoring map" for Tier-2 alignment: a building-footprint contour derived from the
persistent point cloud itself. TartanGround (and most reconstruction pipelines without a real
OSM feed) has no native floor-plan GT, so this flattens the cloud to a BEV density grid and
traces the single largest-connected-component's outer boundary -- a synthetic OSM substitute,
not a claim that this IS an OSM contour. A real deployment with an actual OSM/floor-plan feed
would skip straight to `resample_contour()`-shaped input and never call `extract_outline()` at
all.

Ported from experiments/bev-contour-01/{extract_bev_contour.py,extract_outline.py} --
`herald.scene.refine.align`'s `anchored_align()`/`reversed_align()` are this module's consumer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def flatten_to_grid(pts_xy: np.ndarray, cell_size: float, *, sigma_cells: float = 1.5,
                     z_min: float | None = None, z_max: float | None = None,
                     pts_z: np.ndarray | None = None,
                     norm_percentile: float = 100.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(N,2) world-XY points -> (likelihood grid in [0,1], raw counts grid, origin_x, origin_y).
    Optional z_min/z_max restricts to a height band (pts_z required if either is set).
    norm_percentile < 100 normalizes by a percentile of the smoothed density rather than its max,
    so sparse-but-real structure (e.g. a wall seen from a grazing angle, or the building's outer
    boundary getting less overlapping trajectory coverage than the interior) isn't compressed
    toward 0 by a handful of very dense hotspot cells."""
    from scipy.ndimage import gaussian_filter

    if z_min is not None or z_max is not None:
        assert pts_z is not None
        mask = np.ones(len(pts_xy), dtype=bool)
        if z_min is not None:
            mask &= pts_z >= z_min
        if z_max is not None:
            mask &= pts_z <= z_max
        pts_xy = pts_xy[mask]

    x0, y0 = pts_xy[:, 0].min(), pts_xy[:, 1].min()
    x1, y1 = pts_xy[:, 0].max(), pts_xy[:, 1].max()
    nx = max(int(np.ceil((x1 - x0) / cell_size)) + 1, 1)
    ny = max(int(np.ceil((y1 - y0) / cell_size)) + 1, 1)

    counts, _, _ = np.histogram2d(pts_xy[:, 0], pts_xy[:, 1], bins=[nx, ny],
                                   range=[[x0, x0 + nx * cell_size], [y0, y0 + ny * cell_size]])
    smoothed = gaussian_filter(counts, sigma=sigma_cells)
    denom = np.percentile(smoothed[smoothed > 0], norm_percentile) if norm_percentile < 100.0 else smoothed.max()
    likelihood = np.clip(smoothed / (denom + 1e-9), 0.0, 1.0)
    return likelihood.astype(np.float32), counts.astype(np.float32), x0, y0


def extract_outline(density: np.ndarray, cell_size: float, origin_x: float, origin_y: float,
                     *, density_thresh: float = 0.10, close_m: float = 1.0,
                     approx_eps_cells: float = 2.0, keep_top_n: int = 1) -> list[dict]:
    """density: the smoothed/normalized likelihood grid from `flatten_to_grid` (a low threshold,
    e.g. 0.02-0.12, on this grid is far more robust to sparse far-away noise points than raw
    counts>0 would be -- smoothing dilutes an isolated stray point below threshold while real,
    spatially-contiguous structure stays above it). Validated on Hospital: 0.02 let a faint
    disconnected noise blob merge into the outline via closing; 0.12 started cutting into a real
    but sparsely-scanned wing; 0.10 traced the true footprint cleanly with neither failure.

    `close_m` (morphological closing kernel, meters) must exceed the largest real gap you want
    bridged (wing junctions, doorways, sensor dropouts) -- small real gaps otherwise fragment one
    building into several disconnected components. `keep_top_n` keeps only the N largest
    connected components (area-ranked), dropping distant noise blobs; >1 only if the site
    genuinely has multiple separate buildings."""
    import cv2

    any_mask = (density > density_thresh).astype(np.uint8) * 255
    close_cells = max(int(round(close_m / cell_size)), 1)
    kernel = np.ones((close_cells, close_cells), np.uint8)
    closed = cv2.morphologyEx(any_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    fg_labels = sorted(range(1, n_labels), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:keep_top_n]

    outlines = []
    for rank, lbl in enumerate(fg_labels):
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        contours_px, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_px:
            continue
        c = max(contours_px, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(c, approx_eps_cells, closed=True)
        px = simplified.reshape(-1, 2).astype(np.float64)
        # cv2 contour points are (col, row); col indexes the Y-bin axis, row the X-bin axis
        # (numpy (nx, ny) array read as a cv2 (height, width) = (rows=nx, cols=ny) image).
        world = np.stack([origin_x + px[:, 1] * cell_size, origin_y + px[:, 0] * cell_size], axis=1)
        outlines.append({
            "rank": rank, "area_m2": float(cv2.contourArea(simplified) * cell_size ** 2),
            "n_vertices": len(world), "vertices": world.tolist(),
        })
    return outlines


def extract_outline_from_cloud(pts: np.ndarray, *, cell_size: float = 0.1, sigma_cells: float = 1.5,
                                norm_percentile: float = 98.0, **outline_kw) -> dict:
    """One-call convenience: (N,3) persistent-cloud points -> the outline JSON payload
    `resample_contour()` consumes (`scene_extract_contour.py`'s own `--out` content)."""
    pts = np.asarray(pts, np.float64)
    likelihood, _, ox, oy = flatten_to_grid(pts[:, :2], cell_size, sigma_cells=sigma_cells,
                                             pts_z=pts[:, 2], norm_percentile=norm_percentile)
    outlines = extract_outline(likelihood, cell_size, ox, oy, **outline_kw)
    return {"cell_size": cell_size, "origin_xy": [float(ox), float(oy)], "outlines": outlines}


def resample_contour(outline: dict | Path, step: float = 0.2) -> np.ndarray:
    """Outline JSON (dict, or a path to one) -> densely resampled (M,3) points, z=0 -- "sample
    points on the contour, let them emit heat" (the alignment target `reversed_align()`/
    `anchored_align()` register a session's cloud against)."""
    data = json.loads(Path(outline).read_text()) if isinstance(outline, (str, Path)) else outline
    pts = []
    for o in data["outlines"]:
        v = np.asarray(o["vertices"], np.float64)
        ring = np.vstack([v, v[:1]])
        for a, b in zip(ring[:-1], ring[1:]):
            seg_len = float(np.linalg.norm(b - a))
            n = max(int(seg_len / step), 1)
            for i in range(n):
                pts.append(a + (b - a) * (i / n))
    xy = np.asarray(pts, np.float64)
    return np.concatenate([xy, np.zeros((len(xy), 1))], axis=1).astype(np.float32)
