"""Throwaway numpy loader for a persistent SceneMap .npz  (experiments rule 4).

`herald.scene.refine.load_session` is the faithful loader, but importing it pulls the whole
`herald.scene` -> torch chain, which a pure-CPU clustering test does not need (and which OOMs a
login node). We read the same fields `refine.save_session` writes, directly with numpy. If this
experiment graduates, the promoted code should use `refine.load_session` proper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Obj:
    uid: int
    center: np.ndarray            # (3,) float32
    embedding: np.ndarray | None  # (d,) float32 or None
    label: str
    labels: dict                  # label -> vote count
    support: int


def load_objects(path) -> list[Obj]:
    d = np.load(path, allow_pickle=True)
    emb = d["embedding"]
    out = []
    for i in range(len(d["uid"])):
        e = emb[i]
        e = None if e is None else np.asarray(e, np.float32)
        out.append(Obj(int(d["uid"][i]), d["center"][i].astype(np.float32), e,
                       str(d["label"][i]), dict(d["labels"][i]), int(d["support"][i])))
    return out


def load_cloud(path):
    d = np.load(path, allow_pickle=True)
    return d["points"], d["colors"]
