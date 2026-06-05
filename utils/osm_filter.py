"""OSM polygon eligibility rules for the containment hierarchy.

Disposition rules follow OpenStreetMap Map Features:
https://wiki.openstreetmap.org/wiki/Map_features
"""

from __future__ import annotations

from typing import Literal

from utils.osm_helpers import (
    STRONG_CLASSIFICATION_KEYS,
    has_strong_classification_tag,
    is_metadata_key,
)

PolygonDisposition = Literal[
    "hierarchy",
    "boundary",
    "pathway",
    "infrastructure",
    "excluded",
]

HIERARCHY_KEYS = STRONG_CLASSIFICATION_KEYS | frozenset(
    {
        "man_made",
        "basin",
        "aeroway",
    }
)

PATHWAY_HIGHWAY_VALUES = frozenset(
    {
        "footway",
        "path",
        "pedestrian",
        "steps",
        "cycleway",
        "living_street",
        "service",
        "track",
        "bridleway",
    }
)


def polygon_disposition(
    tags: dict[str, str],
    *,
    area_m2: float,
    min_area_m2: float,
    max_area_m2: float,
) -> PolygonDisposition:
    """Decide whether a polygon participates in the containment hierarchy."""
    if area_m2 < min_area_m2:
        return "excluded"
    if area_m2 > max_area_m2:
        return "excluded"

    if tags.get("boundary"):
        return "boundary"

    highway = tags.get("highway")
    if highway:
        if highway in PATHWAY_HIGHWAY_VALUES or tags.get("area") == "yes":
            return "pathway"
        return "pathway"

    if tags.get("power") or tags.get("railway"):
        if has_strong_classification_tag(tags):
            return "hierarchy"
        return "infrastructure"

    if any(key in HIERARCHY_KEYS for key in tags):
        return "hierarchy"

    if not tags or all(is_metadata_key(k) for k in tags):
        return "hierarchy"

    return "hierarchy"
