"""Walkable OSM highway filtering and navigation graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SourceRef
from herald.scene.common.nav import NavGraph, NavNode
from services.osm.client import OSMFeature

WALKABLE_HIGHWAY_TYPES = frozenset(
    {"footway", "path", "pedestrian", "steps", "cycleway", "living_street"}
)


@dataclass(frozen=True)
class PathConfig:
    """Hyperparameters for nav-graph construction (distances in metres)."""

    edge_length: float = 10.0
    """Maximum spacing between consecutive NavNodes along a linestring."""

    snap_radius: float = 2.0
    """NavNodes closer than this are merged into one (links touching paths)."""


def _is_walkable(feature: OSMFeature) -> bool:
    """Whether an OSM feature is a pedestrian-routable linestring."""
    return (
        feature.kind == "linestring"
        and feature.tags.get("highway") in WALKABLE_HIGHWAY_TYPES
    )


def filter_walkable_highways(highways: list[OSMFeature]) -> list[OSMFeature]:
    """Keep linestring highways suitable for pedestrian routing."""
    return [h for h in highways if _is_walkable(h)]


def _densify_polyline(
    points: list[tuple[float, float]], max_spacing: float
) -> list[tuple[float, float]]:
    """Subdivide a polyline so no segment exceeds ``max_spacing``.

    Original vertices are preserved (the path shape is kept) and evenly spaced
    samples are inserted on any segment longer than ``max_spacing``.
    """
    if len(points) < 2:
        return list(points)
    dense = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist == 0.0:
            continue
        steps = max(1, math.ceil(dist / max_spacing))
        for i in range(1, steps + 1):
            t = i / steps
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return dense


class _SnapGrid:
    """Spatial hash that merges points within a radius into shared representatives.

    Cell size equals the snap radius, so every point within the radius of a
    representative falls in the same or an adjacent cell; a 3x3 neighbourhood
    search is therefore exhaustive.
    """

    def __init__(self, radius: float) -> None:
        self._radius = radius
        self._cell = max(radius, 1e-9)
        self._reps: list[tuple[float, float]] = []
        self._buckets: dict[tuple[int, int], list[int]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self._cell)), int(math.floor(y / self._cell)))

    def representative(self, x: float, y: float) -> int:
        """Index of an existing representative within the radius, else a new one."""
        cx, cy = self._key(x, y)
        r2 = self._radius * self._radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self._buckets.get((cx + dx, cy + dy), ()):
                    rx, ry = self._reps[idx]
                    if (rx - x) ** 2 + (ry - y) ** 2 <= r2:
                        return idx
        idx = len(self._reps)
        self._reps.append((x, y))
        self._buckets.setdefault((cx, cy), []).append(idx)
        return idx

    @property
    def positions(self) -> list[tuple[float, float]]:
        return self._reps


def _node_id(index: int) -> str:
    return f"nav_{index:04d}"


def build_path(
    highways: list[OSMFeature],
    frame: Frame,
    config: PathConfig = PathConfig(),
) -> NavGraph:
    """Build a navigation graph from walkable OSM highways.

    Pipeline:
    1. Keep only walkable linestrings (``_is_walkable``).
    2. Project each to ENU metres and densify it into samples no more than
       ``config.edge_length`` apart, so every endpoint and interior sample
       becomes a NavNode and consecutive samples form an edge.
    3. Merge samples within ``config.snap_radius`` so linestrings meeting at a
       junction share one node, yielding a connected routing graph.

    Each node records the OSM way(s) that pass through it in ``refs``. Node
    positions are ENU metres in ``frame``.
    """
    grid = _SnapGrid(config.snap_radius)
    edges: set[tuple[int, int]] = set()
    node_refs: dict[int, set[str]] = {}

    for feature in highways:
        if not _is_walkable(feature):
            continue
        ref = f"{feature.osm_type}/{feature.osm_id}"
        enu = [frame.wgs2enu(lat, lon) for lat, lon in feature.geometry]
        ids = [grid.representative(x, y) for x, y in _densify_polyline(enu, config.edge_length)]
        for idx in ids:
            node_refs.setdefault(idx, set()).add(ref)
        for a, b in zip(ids, ids[1:]):
            if a != b:
                edges.add((a, b) if a < b else (b, a))

    graph = NavGraph(frame=frame)
    for index, pos in enumerate(grid.positions):
        refs = [SourceRef(assigned_by="osm", assigned_id=r) for r in sorted(node_refs.get(index, ()))]
        graph.add_node(NavNode(id=_node_id(index), pos=pos, refs=refs))
    
    for a, b in sorted(edges):
        (ax, ay) = grid.positions[a]
        (bx, by) = grid.positions[b]
        graph.add_edge(
            _node_id(a),
            _node_id(b),
            length=math.hypot(bx - ax, by - ay),
            source="osm",
        )
    return graph
