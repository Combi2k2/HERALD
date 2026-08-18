"""Tier-2 object reconciliation: fuse two sessions' object OBBs into one persistent set.

Once a session's cloud is aligned onto the canonical/other session (a `Sim3`, see
`align`), its objects are carried through the same transform and merged with the other
session's objects. Merging is a **two-pass gate + union-find** (no global assignment): a
cross-session pair fuses if it clears strong overlap (pass 1, label-agnostic) or, failing
that, close co-location with label agreement (pass 2); a union-find then collapses every
connected component into one object, so a knot of duplicates becomes a single object and
the scheme generalises to N sessions. Objects seen in only one session pass through (insert).

This is the "update / insert" half of Tier-2; evidence-gated *retire* (dropping a canonical
object a new visit proves is gone) is not built yet.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from herald.scene.common.geometry import Sim3
from herald.scene.recon.types import SceneObject
from herald.scene.recon.utils import obb_ios, quat_to_R


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1 + t) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:                                        # pick the largest diagonal for stability
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = np.zeros(3)
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    v = np.array([x, y, z, w], np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


# --------------------------------------------------------------------------- transform

def transform_object(o: SceneObject, T: Sim3) -> SceneObject:
    """Carry one OBB through a Sim3: centre by the full transform, orientation by the
    rotation, extent by the scale. Identity / labels / evidence / provenance / embedding /
    crops are carried unchanged."""
    center = T.apply(np.asarray(o.center, np.float64).reshape(1, 3))[0]
    quat = _mat_to_quat(T.R @ quat_to_R(o.quat_xyzw))
    return SceneObject(uid=o.uid, center=center.astype(np.float32),
                       half_size=(np.asarray(o.half_size) * T.s).astype(np.float32),
                       quat_xyzw=quat.astype(np.float32), label=o.label, labels=dict(o.labels),
                       conf=o.conf, support=o.support,
                       sessions={k: dict(v) for k, v in o.sessions.items()},  # crop refs ride here (image-space)
                       embedding=o.embedding)


# --------------------------------------------------------------------------- label agreement

def labels_agree(a: SceneObject, b: SceneObject) -> bool:
    """Semantic-agreement test for the *proximity* (pass-2) merge gate below.

    ISOLATED ON PURPOSE so it can be swapped without touching the fusion logic. Current rule
    is a **vote-distribution overlap**: the two agree if either object's top label appears
    anywhere in the other's label-vote dict. Looser than exact top-label equality -- it lets
    a `couch`/`sofa` pair fuse when their votes overlap -- without going fully label-agnostic.
    Intended replacement once `SceneObject.embedding` is populated: an embedding-similarity
    criterion (e.g. cosine >= thr). Pass 1 (strong overlap) never calls this -- it fuses
    regardless of label."""
    return a.label in b.labels or b.label in a.labels


# --------------------------------------------------------------------------- merge gate

def _mergeable(a: SceneObject, b: SceneObject, ios: float, cdiag: float, *,
               ios_trust: float, cdiag_gate: float) -> bool:
    """Two independent passes, no global assignment -- pass 2 only rescues what pass 1 misses:
        pass 1  IoS >= ios_trust                          -> definitely same (label-agnostic;
                                                             strong overlap is conclusive)
        pass 2  cdiag < cdiag_gate AND labels_agree(a, b) -> likely same (proximity fallback;
                                                             weaker evidence, so label-gated)"""
    if ios >= ios_trust:
        return True
    if cdiag < cdiag_gate and labels_agree(a, b):
        return True
    return False


# --------------------------------------------------------------------------- fuse a cluster

def _merge_sessions(objs: list[SceneObject]) -> dict:
    """Union the per-session provenance of a cluster: session_id -> {support, conf, track_id,
    crops}. Crop refs concatenate; support sums and conf maxes if the same session appears
    twice (rare, via A-A bridging)."""
    out: dict = {}
    for o in objs:
        for sid, rec in o.sessions.items():
            crops = list(rec.get("crops", []))
            if sid in out:
                out[sid] = {"support": out[sid]["support"] + int(rec["support"]),
                            "conf": max(out[sid]["conf"], float(rec["conf"])),
                            "track_id": out[sid]["track_id"],
                            "crops": out[sid]["crops"] + crops}
            else:
                out[sid] = {"support": int(rec["support"]), "conf": float(rec["conf"]),
                            "track_id": int(rec["track_id"]), "crops": crops}
    return out


def _fuse_many(objs: list[SceneObject], uid: int) -> SceneObject:
    """Collapse one union-find component (>=1 objects from any sessions) into a single object
    with the given persistent `uid`: centre is a confidence-weighted mean; **orientation AND
    extent are taken whole from the most-confident ("lead") box**; support summed, label votes
    merged, per-session provenance (incl. crop refs) unioned, embedding from the lead.

    Extent is NOT averaged on purpose: each box's `half_size` is expressed in that box's own
    rotated frame, so averaging extent components across boxes with different yaw mixes the
    thin/wide axes and inflates the box (a thin door reads as thick). Taking orientation and
    extent from the same (lead) box keeps them consistent."""
    lead = max(objs, key=lambda o: o.conf)
    w = np.array([max(o.conf, 1e-3) for o in objs], np.float64)
    centers = np.array([o.center for o in objs], np.float64).reshape(len(objs), 3)
    center = (w[:, None] * centers).sum(0) / w.sum()          # position: confidence-weighted mean
    labels: dict = {}
    for o in objs:
        for k, v in o.labels.items():
            labels[k] = labels.get(k, 0) + v
    return SceneObject(uid=uid, center=center.astype(np.float32),
                       half_size=np.asarray(lead.half_size, np.float32), quat_xyzw=lead.quat_xyzw,
                       label=max(labels, key=labels.get) if labels else lead.label, labels=labels,
                       conf=max(o.conf for o in objs), support=sum(o.support for o in objs),
                       sessions=_merge_sessions(objs), embedding=lead.embedding)


# --------------------------------------------------------------------------- union-find

class _DSU:
    """Minimal disjoint-set forest (path halving)."""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# --------------------------------------------------------------------------- reconcile

def reconcile(a: list[SceneObject], b: list[SceneObject], *,
              ios_trust: float = 0.5, cdiag_gate: float = 0.5) -> dict:
    """Reconcile object list `b` (already Sim3-transformed into `a`'s frame) against `a`
    into one persistent set -- no Hungarian, no one-to-one constraint.

    Every cross-session pair (Aᵢ, Bⱼ) that passes `_mergeable` adds an edge; a union-find
    collapses each connected component into one fused object (`_fuse_many`). So a knot of
    overlapping duplicates -- one A with two B's, or two A's bridged by a shared B -- becomes
    a single object, and the scheme generalises directly to N sessions (pool them all; one
    component == one persistent object). Only cross-session (A-B) edges are added; objects
    within a session are assumed already de-duplicated by that session's tracker.

    Two size-robust signals per pair (plain IoU is dragged below threshold by BoxerNet's
    noisy extents even for true duplicates):
      * IoS   = intersection / smaller-box volume  (small box nested in a big one ~1)
      * cdiag = centre distance / mean box diagonal (small => co-located)

    Persistent identity: a fused / only-A object keeps its canonical A `uid`; a new (only-B)
    object gets a fresh uid beyond A's range -- so A's ids stay stable across successive merges.

    Returns {merged, matched, only_a, only_b}. `merged` is ordered fused ++ only_a ++ only_b;
    `matched` is one representative (obj_a, obj_b, ios) per both-session component (viz/print)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        base = (max(o.uid for o in a) + 1) if a else 0
        out = [_fuse_many([o], uid=(o.uid if k < na else base + (k - na)))
               for k, o in enumerate(list(a) + list(b))]
        return {"merged": out, "matched": [], "only_a": out[:na], "only_b": out[na:]}

    Ra = [quat_to_R(o.quat_xyzw) for o in a]
    Rb = [quat_to_R(o.quat_xyzw) for o in b]
    ca = np.array([o.center for o in a], np.float64).reshape(na, 3)
    cb = np.array([o.center for o in b], np.float64).reshape(nb, 3)
    rad_a = np.linalg.norm(np.array([o.half_size for o in a], np.float64).reshape(na, 3), axis=1)
    rad_b = np.linalg.norm(np.array([o.half_size for o in b], np.float64).reshape(nb, 3), axis=1)

    dsu = _DSU(na + nb)                                               # 0..na-1 = A, na.. = B
    edge_ios: dict = {}                                              # (i, j) -> IoS for passing cross pairs
    for i in range(na):
        for j in range(nb):
            dist = float(np.linalg.norm(ca[i] - cb[j]))
            if dist > rad_a[i] + rad_b[j]:                           # cheap reject: too far to co-locate
                continue
            ios = obb_ios(ca[i], a[i].half_size, Ra[i], cb[j], b[j].half_size, Rb[j])
            cdiag = dist / (rad_a[i] + rad_b[j] + 1e-9)              # mean full diagonal = rad_a + rad_b
            if _mergeable(a[i], b[j], ios, cdiag, ios_trust=ios_trust, cdiag_gate=cdiag_gate):
                dsu.union(i, na + j)
                edge_ios[(i, j)] = float(ios)

    pool = list(a) + list(b)
    comps: dict = defaultdict(list)
    for idx in range(na + nb):
        comps[dsu.find(idx)].append(idx)

    next_uid = (max(o.uid for o in a) + 1) if a else 0               # fresh uids for new objects
    fused, only_a, only_b, matched = [], [], [], []
    for members in comps.values():
        a_idx = [m for m in members if m < na]
        b_idx = [m - na for m in members if m >= na]
        mobjs = [pool[m] for m in members]
        if a_idx and b_idx:                                          # seen in both -> keep canonical A uid
            lead_a = max(a_idx, key=lambda i: a[i].conf)
            fused.append(_fuse_many(mobjs, uid=a[lead_a].uid))
            best = max(((i, j) for i in a_idx for j in b_idx if (i, j) in edge_ios),
                       key=lambda ij: edge_ios[ij], default=None)    # representative cross edge
            matched.append((a[best[0]], b[best[1]], edge_ios[best]) if best is not None
                           else (a[a_idx[0]], b[b_idx[0]], 0.0))
        elif a_idx:                                                  # only A -> keep its uid
            only_a.append(_fuse_many(mobjs, uid=a[a_idx[0]].uid))
        else:                                                        # new (only B) -> fresh uid
            only_b.append(_fuse_many(mobjs, uid=next_uid))
            next_uid += 1
    return {"merged": fused + only_a + only_b, "matched": matched,
            "only_a": only_a, "only_b": only_b}


# --------------------------------------------------------------------------- session IO

def save_session(path, result) -> None:
    """Persist a SceneMap (cloud + objects) to a single .npz -- full fidelity, every field of
    every object round-trips (uid, OBB, labels, evidence, per-session provenance incl. crop
    refs, embedding) plus the map-level `meta` (session ids + source paths)."""
    from pathlib import Path
    o = result.objects
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=result.points.astype(np.float32), colors=result.colors.astype(np.uint8),
        uid=np.array([x.uid for x in o], np.int64),
        center=np.array([x.center for x in o], np.float32).reshape(-1, 3),
        half_size=np.array([x.half_size for x in o], np.float32).reshape(-1, 3),
        quat_xyzw=np.array([x.quat_xyzw for x in o], np.float32).reshape(-1, 4),
        conf=np.array([x.conf for x in o], np.float32),
        support=np.array([x.support for x in o], np.int64),
        label=np.array([x.label for x in o], object),
        labels=np.array([dict(x.labels) for x in o], object),
        sessions=np.array([dict(x.sessions) for x in o], object),
        embedding=np.array([x.embedding for x in o], object),
        meta=np.array(getattr(result, "meta", {}), object))


def load_meta(path) -> dict:
    """Map-level `meta` of a saved SceneMap (session roster + source paths). {} if absent."""
    d = np.load(path, allow_pickle=True)
    return dict(d["meta"].item()) if "meta" in d.files else {}


def load_session(path):
    """Load a saved SceneMap -> (list[SceneObject], points, colors). Backward-compatible with
    pre-unification dumps (old `track_id`/`motion` schema, no uid/sessions/embedding): missing
    fields fall back to sensible defaults (uid<-track_id, empty provenance)."""
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    n = len(d["center"])
    uid = d["uid"] if "uid" in keys else (d["track_id"] if "track_id" in keys else np.arange(n))
    objs = []
    for i in range(n):
        emb = d["embedding"][i] if "embedding" in keys else None
        emb = None if emb is None else np.asarray(emb, np.float32)   # object-array row -> float32
        objs.append(SceneObject(
            uid=int(uid[i]), center=d["center"][i], half_size=d["half_size"][i],
            quat_xyzw=d["quat_xyzw"][i], label=str(d["label"][i]), labels=dict(d["labels"][i]),
            conf=float(d["conf"][i]), support=int(d["support"][i]),
            sessions=dict(d["sessions"][i]) if "sessions" in keys else {},
            embedding=emb))
    return objs, d["points"], d["colors"]
