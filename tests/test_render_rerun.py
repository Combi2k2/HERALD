"""Tests for herald.ui.rerun."""

from unittest.mock import MagicMock, patch

import numpy as np

from herald.ui.rerun import (
    hierarchy_color,
    render_scene,
    semantic_color,
    _children,
    _dist_to_leaf,
)
from herald.scene.common.geometry import Frame, Geometry, Vec3
from herald.scene.common.graph import SceneGraph, SceneNode, SourceRef
from herald.scene.common.nav import NavGraph
from herald.scene.common.repr import SceneRepr


def test_dist_to_leaf():
    graph = SceneGraph(emb_model_id="stub", nodes=[])
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    tree = _children(graph)
    cache: dict[str, int] = {}
    assert _dist_to_leaf("c", tree, cache) == 0
    assert _dist_to_leaf("b", tree, cache) == 1
    assert _dist_to_leaf("a", tree, cache) == 2


def test_hierarchy_color_by_dist():
    assert hierarchy_color(0) != hierarchy_color(2)


def test_semantic_color_prefers_category():
    node = SceneNode(
        id="n1",
        type="structure",
        pid=None,
        geom=Geometry(type="point", coords=np.zeros((1, 3)), frame="ENU"),
        category="water",
        role="structure",
    )
    assert semantic_color(node) == [34, 211, 238]


def test_render_scene_logs_all_layers():
    frame = Frame.from_origin(48.71, 2.20)
    site_id = "site-uuid"
    ring = [(48.71, 2.20), (48.712, 2.20), (48.712, 2.202), (48.71, 2.202), (48.71, 2.20)]
    site = SceneNode(
        id=site_id,
        type="site",
        pid=None,
        geom=Geometry(
            type="polygon",
            coords=np.array([[*frame.wgs2enu(*p), 0.0] for p in ring]),
            frame="ENU",
        ),
    )
    building = SceneNode(
        id="b1",
        type="structure",
        pid="r1",
        geom=Geometry(
            type="polygon",
            coords=np.array(
                [[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in [
                    (48.711, 2.201), (48.711, 2.2015), (48.7115, 2.2015),
                    (48.7115, 2.201), (48.711, 2.201),
                ]]
            ),
            frame="ENU",
            offset=Vec3((0.0, 0.0, 12.0)),
        ),
        category="building",
    )
    region = SceneNode(
        id="r1",
        type="region",
        pid=site_id,
        geom=Geometry(
            type="polygon",
            coords=np.array(
                [[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in [
                    (48.710, 2.200), (48.710, 2.202), (48.712, 2.202),
                    (48.712, 2.200), (48.710, 2.200),
                ]]
            ),
            frame="ENU",
        ),
        category="vegetation",
    )
    graph = SceneGraph(emb_model_id="stub", nodes=[site, region, building])
    graph.add_edge(site_id, "r1")
    graph.add_edge("r1", "b1")
    scene = SceneRepr(frame=frame, graph=graph, nav=NavGraph())

    mock_rr = MagicMock()
    with patch.dict("sys.modules", {"rerun": mock_rr}):
        render_scene(scene, spawn=False)

    paths = [c.args[0] for c in mock_rr.log.call_args_list if c.args]
    assert any(str(p).startswith("semantic/") for p in paths)
    assert any(str(p).startswith("hierarchy/") for p in paths)
