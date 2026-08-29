"""Overlap-merge dedup for a merged SceneMap  (experiment fix for "intersecting objects not fused").

reconcile / suppress_contained only drop *nested* boxes or fuse near-total (IoS>=0.9) or
label-agreeing overlaps -- so two similar-size boxes with high *mutual* overlap but disagreeing
labels survive as duplicates. This fuses any pair with obb_ios >= thr, LABEL-AGNOSTIC, via
union-find: geometry+orientation from the lead (max-support) box, centre support-weighted, support
summed, labels vote-merged, per-session provenance unioned, embedding from lead. Each fused object
inherits the lead's route-node attachment (fused boxes are co-located), so no re-align is needed.

    uv run python experiments/area-cluster-02/dedup_objects.py \
        --merged data/tartanground/Hospital/recon/merged_frustum5.npz \
        --art-dir experiments/area-cluster-02/out/frustum5 --ios 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from herald.scene.recon.utils import obb_ios, quat_to_R


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--art-dir", type=Path, required=True, help="dir with merged_route.json + object_attach.json")
    ap.add_argument("--ios", type=float, default=0.5, help="fuse objects with obb_ios (inter/smaller) >= this")
    ap.add_argument("--out", type=Path, default=None, help="deduped .npz (default: <merged>_dedup.npz)")
    ap.add_argument("--out-art", type=Path, default=None, help="deduped art dir (default: <art-dir>_dedup)")
    args = ap.parse_args()

    d = np.load(args.merged, allow_pickle=True)
    C = d["center"].astype(np.float64)
    H = d["half_size"].astype(np.float64)
    Q = d["quat_xyzw"].astype(np.float64)
    sup = d["support"].astype(np.int64)
    conf = d["conf"].astype(np.float64)
    label = [str(x) for x in d["label"]]
    votes = [dict(v) for v in d["labels"]]
    sess = [dict(s) for s in d["sessions"]]
    emb = d["embedding"]
    uid = [int(x) for x in d["uid"]]
    N = len(C)
    R = [quat_to_R(Q[i]) for i in range(N)]
    att = json.loads((args.art_dir / "object_attach.json").read_text())
    a_nodes = att.get("nodes") or [[n] for n in att["node"]]   # per-object observing node-set
    a_off = att["offset"]

    # union-find over high-IoS pairs (label-agnostic). Prune by centre distance for speed.
    diag = 2 * np.linalg.norm(H, axis=1)
    dsu = _DSU(N)
    npairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(C[i] - C[j]) > 0.5 * (diag[i] + diag[j]):
                continue                               # too far to overlap meaningfully
            if obb_ios(C[i], H[i], R[i], C[j], H[j], R[j]) >= args.ios:
                dsu.union(i, j)
                npairs += 1

    groups: dict = {}
    for i in range(N):
        groups.setdefault(dsu.find(i), []).append(i)

    # fuse each group
    def merge_sessions(idxs):
        out: dict = {}
        for i in idxs:
            for sid, prov in sess[i].items():
                if sid not in out:
                    out[sid] = {"support": 0, "conf": 0.0, "track_id": prov.get("track_id"), "crops": []}
                out[sid]["support"] += int(prov.get("support", 0))
                out[sid]["conf"] = max(out[sid]["conf"], float(prov.get("conf", 0.0)))
                out[sid]["crops"] = out[sid]["crops"] + list(prov.get("crops", []))
        return out

    nc, nh, nq, ns, ncf, nlab, nvotes, nsess, nemb, nuid, nnode, noff = ([] for _ in range(12))
    for _, idxs in sorted(groups.items()):
        lead = max(idxs, key=lambda i: sup[i])
        w = sup[idxs].astype(np.float64)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(idxs)) / len(idxs)
        nc.append((w[:, None] * C[idxs]).sum(0))
        nh.append(H[lead]); nq.append(Q[lead])                 # orientation+extent from lead
        ns.append(int(sup[idxs].sum())); ncf.append(float(conf[idxs].max()))
        v: dict = {}
        for i in idxs:
            for k, c in votes[i].items():
                v[k] = v.get(k, 0) + c
        nvotes.append(v); nlab.append(max(v, key=v.get) if v else label[lead])
        nsess.append(merge_sessions(idxs))
        nemb.append(emb[lead]); nuid.append(uid[lead])
        # fused boxes are co-located: union their observing node-sets, offset = closest approach
        nnode.append(sorted({nid for i in idxs for nid in a_nodes[i]}))
        noff.append(min(a_off[i] for i in idxs))

    M = len(nc)
    out = args.out or args.merged.with_name(args.merged.stem + "_dedup.npz")
    np.savez(out, points=d["points"], colors=d["colors"],
             uid=np.array(nuid, np.int64), center=np.array(nc, np.float32),
             half_size=np.array(nh, np.float32), quat_xyzw=np.array(nq, np.float32),
             conf=np.array(ncf, np.float32), support=np.array(ns, np.int64),
             label=np.array(nlab, object), labels=np.array(nvotes, object),
             sessions=np.array(nsess, object), embedding=np.array(nemb, object),
             meta=d["meta"])
    out_art = args.out_art or args.art_dir.with_name(args.art_dir.name + "_dedup")
    out_art.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.art_dir / "merged_route.json", out_art / "merged_route.json")
    (out_art / "object_attach.json").write_text(json.dumps({"uids": nuid, "nodes": nnode, "offset": noff}))

    fused_groups = sum(1 for g in groups.values() if len(g) > 1)
    print(f"{N} objects -> {M} after overlap-merge (ios>={args.ios}): "
          f"{npairs} overlapping pairs, {fused_groups} groups fused, {N-M} objects removed", flush=True)
    print(f"deduped map -> {out}", flush=True)
    print(f"deduped artifacts -> {out_art}/ (merged_route.json + object_attach.json)", flush=True)


if __name__ == "__main__":
    main()
