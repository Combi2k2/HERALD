"""Tests for hierarchy-based Rerun layout."""

from herald.viewer.hierarchy_layout import (
    LAYER_THICKNESS,
    ancestor_layer_z,
    children_map_from_pairs,
    depth_from_site,
    is_containment_leaf,
    max_distance_to_leaf,
)


def test_depth_from_site_chain():
    children = children_map_from_pairs([("site_000", "region"), ("region", "leaf")])
    assert depth_from_site("site_000", children) == 0
    assert depth_from_site("region", children) == 1
    assert depth_from_site("leaf", children) == 2


def test_ancestor_layer_z_uses_max_distance_to_leaf():
    children = children_map_from_pairs([("site_000", "region"), ("region", "leaf")])
    assert ancestor_layer_z("site_000", children) is None
    assert ancestor_layer_z("leaf", children) is None
    assert ancestor_layer_z("region", children) == LAYER_THICKNESS

    nested = children_map_from_pairs([
        ("site_000", "outer"),
        ("outer", "inner"),
        ("inner", "leaf"),
    ])
    assert max_distance_to_leaf("outer", nested) == 2
    assert max_distance_to_leaf("inner", nested) == 1
    assert ancestor_layer_z("outer", nested) == 2 * LAYER_THICKNESS
    assert ancestor_layer_z("inner", nested) == LAYER_THICKNESS


def test_is_containment_leaf():
    children = children_map_from_pairs([("site_000", "a"), ("a", "b")])
    assert is_containment_leaf("b", children)
    assert not is_containment_leaf("a", children)
