"""Tests for SceneRepr round-trip serialization."""

from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph, SceneNode, Geometry
from herald.scene.common.nav import NavGraph, NavNode
from herald.scene.common.repr import SceneRepr


def test_scene_repr_round_trip():
    frame = Frame.from_origin(48.71, 2.20)
    graph = SceneGraph(emb_model_id="stub-v0", vlm_model_id="test")
    graph.add_node(
        SceneNode(
            id="site_000",
            pid=None,
            type="site",
            geom=Geometry(type="polygon", frame="ENU", coords=[[0.0, 0.0, 0.0]]),
        )
    )
    nav = NavGraph()
    nav.add_node(NavNode(id="nav_0000", pos=(1.0, 2.0)))

    scene = SceneRepr(frame=frame, graph=graph, nav=nav)
    loaded = SceneRepr.from_dict(scene.to_dict())

    assert loaded.frame.lat == frame.lat
    assert loaded.frame.lon == frame.lon
    assert len(loaded.graph.nodes) == 1
    assert loaded.graph.emb_model_id == "stub-v0"
    assert len(loaded.nav.nodes) == 1
    assert loaded.nav.nodes[0].pos == (1.0, 2.0)
