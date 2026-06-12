"""Tests for hierarchy-based Rerun layout."""

from herald.viewer.hierarchy_layout import (
    LAYER_HEIGHT,
    ancestor_layer_z,
    children_map_from_pairs,
    depth_from_site,
    is_containment_leaf,
)


def test_depth_from_site_chain():
    children = children_map_from_pairs([("site_000", "region"), ("region", "leaf")])
    assert depth_from_site("site_000", children) == 0
    assert depth_from_site("region", children) == 1
    assert depth_from_site("leaf", children) == 2


def test_ancestor_layer_z_only_for_parents():
    children = children_map_from_pairs([("site_000", "region"), ("region", "leaf")])
    assert ancestor_layer_z("site_000", children) is None
    assert ancestor_layer_z("leaf", children) is None
    assert ancestor_layer_z("region", children) == LAYER_HEIGHT


def test_is_containment_leaf():
    children = children_map_from_pairs([("site_000", "a"), ("a", "b")])
    assert is_containment_leaf("b", children)
    assert not is_containment_leaf("a", children)
