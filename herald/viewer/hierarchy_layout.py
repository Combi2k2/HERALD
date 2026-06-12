"""Hierarchy layout: leaf nodes on ground, ancestors as elevated contours."""

from __future__ import annotations

# Vertical spacing between ancestor map layers (local ENU Z-up).
LAYER_HEIGHT = 225.0
GROUND_Z = 0.0
SITE_ID = "site_000"


def children_map_from_pairs(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for parent_id, child_id in edges:
        mapping.setdefault(parent_id, []).append(child_id)
    for child_ids in mapping.values():
        child_ids.sort()
    return mapping


def max_distance_to_leaf(
    node_id: str,
    children_map: dict[str, list[str]],
    *,
    cache: dict[str, int] | None = None,
) -> int:
    """Longest path from ``node_id`` down to any leaf in the containment tree."""
    if cache is not None and node_id in cache:
        return cache[node_id]
    children = children_map.get(node_id, [])
    if not children:
        result = 0
    else:
        result = 1 + max(
            max_distance_to_leaf(child, children_map, cache=cache) for child in children
        )
    if cache is not None:
        cache[node_id] = result
    return result


def is_containment_leaf(node_id: str, children_map: dict[str, list[str]]) -> bool:
    return node_id not in children_map or not children_map[node_id]


def _parent_map(children_map: dict[str, list[str]]) -> dict[str, str]:
    parent: dict[str, str] = {}
    for parent_id, child_ids in children_map.items():
        for child_id in child_ids:
            parent[child_id] = parent_id
    return parent


def depth_from_site(
    node_id: str,
    children_map: dict[str, list[str]],
    *,
    site_id: str = SITE_ID,
    cache: dict[str, int] | None = None,
) -> int:
    """Containment depth from the site root (site=0, direct child=1, …)."""
    if node_id == site_id:
        return 0
    if cache is not None and node_id in cache:
        return cache[node_id]
    parent_id = _parent_map(children_map).get(node_id)
    if parent_id is None:
        result = 1
    else:
        result = depth_from_site(
            parent_id, children_map, site_id=site_id, cache=cache
        ) + 1
    if cache is not None:
        cache[node_id] = result
    return result


def ancestor_layer_z(
    node_id: str,
    children_map: dict[str, list[str]],
    *,
    site_id: str = SITE_ID,
    cache: dict[str, int] | None = None,
) -> float | None:
    """Z for an ancestor footprint; ``None`` for site root and leaves (ground map only)."""
    if node_id == site_id or is_containment_leaf(node_id, children_map):
        return None
    depth = depth_from_site(node_id, children_map, site_id=site_id, cache=cache)
    return depth * LAYER_HEIGHT
