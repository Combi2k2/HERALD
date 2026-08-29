"""Scene map for Tier-1 recon and Tier-2 merge: a coloured point cloud + object nodes.

An object is a `SceneNode` (level="object"): OBB geometry in `geom`, label votes in `votes`,
evidence in `supp`/`conf`, and one `SourceRef` per contributing session in `refs`. `SceneObject`
is a thin constructor/accessor over `SceneNode` that keeps the recon/merge code's field names
(center/half_size/quat_xyzw/label/labels/support/sessions/embedding)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from herald.scene.common.geometry import Geometry
from herald.scene.common.graph import SceneNode
from herald.scene.common.source import SourceRef


class SceneObject(SceneNode):
    """A SceneNode(level="object") built from recon fields; provenance goes to `refs` (one
    SourceRef per session: assigned_by=session id, assigned_id=track id, crops/frames in metadata)."""

    def __init__(self, uid, center, half_size, quat_xyzw, label, labels,
                 conf=0.0, support=0, sessions=None, embedding=None):
        refs = [SourceRef(assigned_by=str(sid), assigned_id=str(p.get("track_id")),
                          supp=int(p.get("support", 0)), conf=float(p.get("conf", 0.0)),
                          metadata={"crops": [[int(f), [float(v) for v in b]] for f, b in p.get("crops", [])],
                                    "frames": [int(x) for x in p.get("frames", [])]})
                for sid, p in (sessions or {}).items()]
        super().__init__(uid=int(uid), level="object",
                         geom=Geometry(type="obb", frame="NED", offset=center,
                                       half_size=half_size, quat_xyzw=quat_xyzw),
                         votes=dict(labels), name=label, supp=int(support), conf=float(conf), refs=refs)
        self.embedding = embedding

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.geom.offset, np.float32)

    @property
    def half_size(self) -> np.ndarray:
        return np.asarray(self.geom.half_size, np.float32)

    @property
    def quat_xyzw(self) -> np.ndarray:
        return np.asarray(self.geom.quat_xyzw, np.float32)

    @property
    def label(self) -> str:
        return self.name

    @property
    def labels(self) -> dict:
        return self.votes

    @property
    def support(self) -> int:
        return self.supp

    @property
    def sessions(self) -> dict:
        return {r.assigned_by: {"support": r.supp, "conf": r.conf,
                                "track_id": int(r.assigned_id) if r.assigned_id not in (None, "None") else None,
                                "crops": r.metadata.get("crops", []), "frames": r.metadata.get("frames", [])}
                for r in self.refs}


@dataclass
class SceneMap:
    points: np.ndarray
    colors: np.ndarray
    objects: list                                 # list[SceneObject]
    dynamic: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


SessionResult = SceneMap
