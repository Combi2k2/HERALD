"""Tests for Rerun Z epsilon helper."""

from herald.viewer.mesh import layer_z_with_epsilon


def test_layer_z_with_epsilon_offsets_by_entity():
    assert layer_z_with_epsilon(10.0, 1) != layer_z_with_epsilon(10.0, 2)
    assert layer_z_with_epsilon(10.0, "node_1") == layer_z_with_epsilon(10.0, "node_1")
