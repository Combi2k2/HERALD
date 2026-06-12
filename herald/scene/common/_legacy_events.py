"""Live pipeline event stream for incremental viewers (Rerun).

Kept out of the generic schema module (``graph.py``). Revise when rebuilding
progress/streaming observability for scene init.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

SceneEventKind = Literal[
    "roi_resolved",
    "osm_fetched",
    "partition_computed",
    "pipeline_status",
    "zone_node_created",
    "containment_inferred",
    "embedding_attached",
    "pathways_ready",
    "pipeline_complete",
]


@dataclass(frozen=True)
class SceneEvent:
    kind: SceneEventKind
    node_id: str | None = None
    pid: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[SceneEvent], None]
