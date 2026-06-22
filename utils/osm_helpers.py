"""OSM tag helpers for classification and VLM context."""

from __future__ import annotations

METADATA_KEYS = frozenset(
    {
        "source",
        "check_date",
        "created_by",
        "note",
        "fixme",
        "todo",
        "description",
        "start_date",
        "end_date",
        "opening_date",
        "ref",
        "ref:FR:cadastre",
        "ref:INSEE",
        "ref:NUM_ILOT",
        "ref:UAI",
        "local_ref",
        "length",
        "layer",
        "survey:date",
    }
)

METADATA_PREFIXES = ("addr:", "ref:", "source:", "contact:")

STRONG_CLASSIFICATION_KEYS = frozenset(
    {
        "building",
        "building:part",
        "amenity",
        "leisure",
        "landuse",
        "natural",
        "tourism",
        "shop",
        "office",
        "healthcare",
        "historic",
        "sport",
        "water",
    }
)


def is_metadata_key(key: str) -> bool:
    if key in METADATA_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in METADATA_PREFIXES)


def has_strong_classification_tag(tags: dict[str, str]) -> bool:
    return any(key in STRONG_CLASSIFICATION_KEYS for key in tags)


def context_tags(tags: dict[str, str]) -> dict[str, str]:
    """Tags sent to VLM / stored on nodes (no metadata noise)."""
    return {
        key: value
        for key, value in sorted(tags.items())
        if not is_metadata_key(key)
    }

def osm_tags(node) -> dict[str, str]:
    """OSM tag dict from a scene node's primary source ref."""
    for ref in node.refs:
        if ref.assigned_by == "osm":
            raw = ref.metadata.get("tags")
            if isinstance(raw, dict):
                return dict(raw)
    return {}

