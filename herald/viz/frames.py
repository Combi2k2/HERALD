from __future__ import annotations

import numpy as np

from herald.datasets.tartanground import SKY_SENTINEL_M

def colorize_depth(
    depth: np.ndarray,
    *,
    cmap: str = "jet",
    vmax: float | None = None,
    mode: str = "inverse",
    sky_rgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    from matplotlib import colormaps

    valid = depth < SKY_SENTINEL_M
    finite = depth[valid]
    vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0

    if mode == "inverse":
        d = np.clip(depth, vmin, vmax)
        inv, inv_lo, inv_hi = 1.0 / d, 1.0 / vmax, 1.0 / vmin
        norm = (inv - inv_lo) / max(inv_hi - inv_lo, 1e-6)
    else:
        norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)

    rgb = (colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = sky_rgb
    return rgb

def label_palette(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = rng.integers(40, 256, size=(n, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)
    return pal

def colorize_labels(labels: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    if palette is None:
        palette = label_palette()
    idx = np.mod(labels, palette.shape[0])
    return palette[idx]
