#!/usr/bin/env python
"""scene_extract_contour: derive the "anchoring map" (a building-footprint contour) from the
persistent point cloud, for scene_align.py's --mode anchored.

TartanGround (and most reconstruction setups without a real OSM/floor-plan feed) has no native
contour ground truth, so this is a synthetic substitute: flatten the cloud to a BEV density
grid, trace the largest-connected-component's outer boundary. A real deployment with an actual
OSM/floor-plan feed would build its own contour.json in this exact schema (one `outlines` entry
per polygon, `vertices`: list of [x, y]) and skip this script entirely.

Re-run as the persistent map grows (more sessions -> better-covered footprint) -- this is cheap
(CPU, seconds), not a one-time step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERALD_ROOT = Path(__file__).resolve().parents[1]
if str(_HERALD_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERALD_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--persistent", type=Path, required=True, help="persistent map (.npz, needs 'points')")
    p.add_argument("--out", type=Path, required=True, help="output contour (.json)")
    p.add_argument("--cell-size", type=float, default=0.1)
    p.add_argument("--sigma-cells", type=float, default=1.5)
    p.add_argument("--norm-percentile", type=float, default=98.0)
    p.add_argument("--density-thresh", type=float, default=0.10,
                    help="threshold on the smoothed likelihood grid -- see extract_outline()'s docstring")
    p.add_argument("--close-m", type=float, default=1.0,
                    help="morphological closing kernel, meters -- must exceed the largest real "
                         "gap to bridge (wing junctions, doorways, sensor dropouts)")
    p.add_argument("--approx-eps-cells", type=float, default=2.0)
    p.add_argument("--keep-top-n", type=int, default=1,
                    help=">1 only if the site genuinely has multiple separate buildings")
    args = p.parse_args()

    from herald.scene.refine.contour import extract_outline_from_cloud

    d = np.load(args.persistent, allow_pickle=True)
    pts = d["points"]
    print(f"loaded {len(pts):,} points", flush=True)

    payload = extract_outline_from_cloud(
        pts, cell_size=args.cell_size, sigma_cells=args.sigma_cells,
        norm_percentile=args.norm_percentile, density_thresh=args.density_thresh,
        close_m=args.close_m, approx_eps_cells=args.approx_eps_cells, keep_top_n=args.keep_top_n)
    for o in payload["outlines"]:
        print(f"outline rank={o['rank']} area={o['area_m2']:.1f}m2 n_vertices={o['n_vertices']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
