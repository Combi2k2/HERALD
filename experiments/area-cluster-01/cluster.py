"""Spatio-semantic area clustering  (experiment area-cluster-01).

Slide formulation:
    D(i,j) = alpha * ||p_i - p_j||_2  +  (1 - alpha) * (1 - cos(e_i, e_j))
`normalize=True` divides the spatial term by r_max so alpha is scale-meaningful (the design
issue we flagged: metres vs [0,2] are incommensurable, so raw alpha is dominated by geometry).

Cluster the precomputed D with HDBSCAN (or DBSCAN). A candidate cluster O_k is accepted iff it
is jointly compact:
    diam(p | O_k) < r_max        (spatial extent, always in metres)
    incoherence(e | O_k) < eps   incoherence = 1 - || mean(unit e) ||  (mean-resultant length;
                                 0 = identical directions, grows with embedding spread -> the
                                 Var(e) < eps gate)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


def spatiosem_matrix(P: np.ndarray, E: np.ndarray, *, alpha: float, r_max: float,
                     normalize: bool) -> np.ndarray:
    """(N,N) distance matrix D per the formula above. P:(N,3) metres, E:(N,d)."""
    diff = P[:, None, :] - P[None, :, :]
    Dp = np.sqrt((diff ** 2).sum(-1))                  # spatial, metres
    if normalize:
        Dp = Dp / float(r_max)
    U = _unit(E.astype(np.float64))
    cos = np.clip(U @ U.T, -1.0, 1.0)
    Ds = 1.0 - cos                                     # semantic, [0,2] (slide-faithful)
    return alpha * Dp + (1.0 - alpha) * Ds


def diam(P: np.ndarray) -> float:
    """Max pairwise Euclidean distance (metres) over a cluster's points."""
    if len(P) < 2:
        return 0.0
    diff = P[:, None, :] - P[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max())


def incoherence(E: np.ndarray) -> float:
    """1 - ||mean(unit e)||  in [0,1]; 0 = all embeddings point the same way."""
    if len(E) == 0:
        return 0.0
    return float(1.0 - np.linalg.norm(_unit(E.astype(np.float64)).mean(0)))


def cluster_labels(D: np.ndarray, *, method: str, min_cluster: int, min_samples: int | None,
                   eps: float) -> np.ndarray:
    from sklearn.cluster import DBSCAN, HDBSCAN
    if method == "hdbscan":
        m = HDBSCAN(metric="precomputed", min_cluster_size=max(2, min_cluster),
                    min_samples=min_samples)
        return m.fit_predict(D.astype(np.float64))
    if method == "dbscan":
        m = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples or min_cluster)
        return m.fit_predict(D.astype(np.float64))
    raise ValueError(f"unknown method {method!r}")


@dataclass
class Area:
    cid: int
    members: list[int]              # indices into the object list
    diam: float
    incoherence: float
    accepted: bool
    reject_reason: str = ""
    center: np.ndarray = field(default=None)     # (3,) mean position
    embedding: np.ndarray = field(default=None)  # (d,) support-weighted mean of member e (unit)
    label_votes: dict = field(default_factory=dict)


def build_areas(objs, labels: np.ndarray, *, r_max: float, eps_coh: float) -> list[Area]:
    """Group by cluster id (dropping noise = -1), run the acceptance gate, and summarise."""
    P = np.array([o.center for o in objs], np.float32)
    E = np.array([o.embedding for o in objs], np.float32)
    sup = np.array([o.support for o in objs], np.float32)
    areas = []
    for cid in sorted(set(labels) - {-1}):
        idx = [i for i, l in enumerate(labels) if l == cid]
        d = diam(P[idx])
        inc = incoherence(E[idx])
        ok = (d < r_max) and (inc < eps_coh)
        reason = "" if ok else (("diam>=r_max " if d >= r_max else "")
                                + ("incoherent" if inc >= eps_coh else "")).strip()
        w = sup[idx]
        e = (w[:, None] * _unit(E[idx])).sum(0)
        n = np.linalg.norm(e)
        votes: dict = {}
        for i in idx:
            for lab, c in objs[i].labels.items():
                votes[lab] = votes.get(lab, 0) + c
        areas.append(Area(int(cid), idx, d, inc, ok, reason,
                          center=P[idx].mean(0),
                          embedding=(e / n).astype(np.float32) if n > 0 else e.astype(np.float32),
                          label_votes=votes))
    return areas
