"""Provenance reference for a scene-graph node (OSM/VLM assignment or a session's contribution)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceRef:
    assigned_by: str
    assigned_id: str
    supp: int = 1
    conf: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assigned_by": self.assigned_by,
            "assigned_id": self.assigned_id,
            "supp": self.supp,
            "conf": self.conf,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        return cls(
            assigned_by=data["assigned_by"],
            assigned_id=data["assigned_id"],
            supp=int(data.get("supp", 1)),
            conf=float(data.get("conf", 1.0)),
            metadata=dict(data.get("metadata") or {}),
        )
