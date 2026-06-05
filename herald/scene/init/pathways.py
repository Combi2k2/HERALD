"""Walkable OSM highway filtering for navigation pathways."""

from __future__ import annotations

from services.osm.client import OSMFeature

WALKABLE_HIGHWAY_TYPES = frozenset(
    {"footway", "path", "pedestrian", "steps", "cycleway", "living_street"}
)


def filter_walkable_highways(highways: list[OSMFeature]) -> list[OSMFeature]:
    """Keep linestring highways suitable for pedestrian routing."""
    return [
        h
        for h in highways
        if h.kind == "linestring" and h.tags.get("highway") in WALKABLE_HIGHWAY_TYPES
    ]
