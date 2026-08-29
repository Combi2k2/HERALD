"""area-cluster-01 entry: cluster a persistent SceneMap's objects into area nodes.

    uv run python experiments/area-cluster-01/run.py --map runs/merge_multi3/merged.npz \
        --alpha 0.6 --r-max 4.0 --eps-coh 0.25 --method hdbscan --normalize

Prints a report (clusters, accepted areas, per-area diam/incoherence/labels) and optionally
renders a Rerun scene (cloud + objects coloured by area) to out/.

OSM region pre-partition (clusters must not span regions) is a no-op here: TartanGround is a
synthetic env with no OSM contours. On real Phase-1 data it would partition objects by region
before clustering. Left as a documented hook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cluster as C          # noqa: E402
import scene_io as sio       # noqa: E402


def _area_color(cid: int) -> tuple:
    rng = np.random.default_rng(cid * 2654435761 % (2**32))
    h = rng.random()
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=Path, default=Path("runs/merge_multi3/merged.npz"))
    p.add_argument("--alpha", type=float, default=0.6, help="spatial vs semantic weight")
    p.add_argument("--r-max", type=float, default=4.0, help="acceptance radius (m); also spatial normalizer")
    p.add_argument("--eps-coh", type=float, default=0.25, help="max embedding incoherence to accept")
    p.add_argument("--method", choices=["hdbscan", "dbscan"], default="hdbscan")
    p.add_argument("--min-cluster", type=int, default=2)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps in D units")
    p.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                   help="divide spatial term by r_max (default on)")
    p.add_argument("--raw", dest="normalize", action="store_false",
                   help="slide-baseline: un-normalized spatial term (alpha dominated by geometry)")
    p.add_argument("--rrd", type=Path, default=HERE / "out" / "area_cluster_01.rrd")
    p.add_argument("--no-viz", action="store_true")
    args = p.parse_args()

    objs = sio.load_objects(args.map)
    keep = [o for o in objs if o.embedding is not None]
    dropped = len(objs) - len(keep)
    objs = keep
    P = np.array([o.center for o in objs], np.float32)
    E = np.array([o.embedding for o in objs], np.float32)
    print(f"map: {args.map.name} | {len(objs)} objects with embeddings"
          + (f" ({dropped} dropped: no embedding)" if dropped else ""), flush=True)

    D = C.spatiosem_matrix(P, E, alpha=args.alpha, r_max=args.r_max, normalize=args.normalize)
    labels = C.cluster_labels(D, method=args.method, min_cluster=args.min_cluster,
                              min_samples=args.min_samples, eps=args.eps)
    areas = C.build_areas(objs, labels, r_max=args.r_max, eps_coh=args.eps_coh)

    n_noise = int((labels == -1).sum())
    accepted = [a for a in areas if a.accepted]
    rejected = [a for a in areas if not a.accepted]
    print(f"\nparams: method={args.method} alpha={args.alpha} r_max={args.r_max} "
          f"eps_coh={args.eps_coh} normalize={args.normalize}"
          + (f" min_cluster={args.min_cluster}" if args.method == "hdbscan" else f" eps={args.eps}"),
          flush=True)
    print(f"clusters found: {len(areas)}  |  accepted areas: {len(accepted)}  |  "
          f"rejected: {len(rejected)}  |  unclustered objects (noise): {n_noise}", flush=True)

    def _top(votes, k=3):
        return ", ".join(f"{l}({c})" for l, c in sorted(votes.items(), key=lambda x: -x[1])[:k])

    print("\nACCEPTED AREAS:", flush=True)
    for a in sorted(accepted, key=lambda a: -len(a.members)):
        print(f"  area {a.cid:2d} | {len(a.members):2d} objs | diam={a.diam:4.1f}m "
              f"incoh={a.incoherence:.3f} | {_top(a.label_votes)}", flush=True)
    if rejected:
        print("\nREJECTED CANDIDATES:", flush=True)
        for a in sorted(rejected, key=lambda a: -len(a.members)):
            print(f"  cand {a.cid:2d} | {len(a.members):2d} objs | diam={a.diam:4.1f}m "
                  f"incoh={a.incoherence:.3f} | {a.reject_reason:12s} | {_top(a.label_votes)}",
                  flush=True)

    if args.no_viz:
        return
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except Exception as e:
        print(f"\n(viz skipped: rerun unavailable — {type(e).__name__})", flush=True)
        return
    pts, cols = sio.load_cloud(args.map)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(pts), min(150000, len(pts)), replace=False)
    rr.init("area_cluster_01")
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world"),
                                    rrb.SelectionPanel(state="expanded"), auto_views=False))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log("world/cloud", rr.Points3D(pts[sub], colors=cols[sub], radii=0.02), static=True)
    # objects coloured by accepted-area membership (grey = noise/rejected)
    obj_area = {i: a for a in areas if a.accepted for i in a.members}
    ocent = np.array([o.center for o in objs], np.float32)
    ocol = np.array([_area_color(obj_area[i].cid) if i in obj_area else (120, 120, 120)
                     for i in range(len(objs))], np.uint8)
    rr.log("world/objects", rr.Points3D(ocent, colors=ocol, radii=0.25), static=True)
    for a in accepted:
        col = _area_color(a.cid)
        mp = ocent[a.members]
        half = (mp.max(0) - mp.min(0)) / 2 + 0.2
        rr.log(f"world/area/{a.cid}",
               rr.Boxes3D(centers=[a.center], half_sizes=[half], colors=[col],
                          fill_mode="majorwireframe",
                          labels=[f"area{a.cid}: {_top(a.label_votes,2)}"]), static=True)
    args.rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.rrd))
    print(f"\nrrd: {args.rrd}", flush=True)


if __name__ == "__main__":
    main()
