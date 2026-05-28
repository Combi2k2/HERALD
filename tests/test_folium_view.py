"""Smoke tests for Rerun viewer integration."""

import sys
from unittest.mock import MagicMock, patch

from herald.osm.client import LatLon
from herald.scene.frame import LocalFrame
from herald.scene.graph import SceneEvent, SceneGraph, SceneNode
from herald.viewer.rerun_view import _closed_ring


def test_closed_ring_appends_first_vertex():
    ring = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert _closed_ring(ring) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0)]


def test_publish_roi_event_without_rerun_import_at_module_level():
    frame = LocalFrame(origin=LatLon(lat=48.71, lon=2.20))
    node = SceneNode(
        id="site_000",
        level="site",
        zone_kind=None,
        text="site",
        geometry_latlon=[(48.71, 2.20), (48.712, 2.20), (48.712, 2.202), (48.71, 2.202), (48.71, 2.20)],
    )
    building = SceneNode(
        id="building_1",
        level="building",
        zone_kind="building",
        text="hall",
        geometry_latlon=[
            (48.711, 2.201),
            (48.711, 2.2015),
            (48.7115, 2.2015),
            (48.7115, 2.201),
            (48.711, 2.201),
        ],
        height_m=12.0,
    )
    graph = SceneGraph(
        site_id="site_000",
        frame=frame,
        embedding_model_id="stub-v0",
        nodes=[node, building],
    )
    graph.add_edge("site_000", "building_1")
    mock_rr = MagicMock()
    with patch.dict("sys.modules", {"rerun": mock_rr}):
        from herald.viewer.rerun_view import RerunSceneViewer

        viewer = RerunSceneViewer(spawn=False)
        viewer.set_frame(frame)
        viewer.publish(
            SceneEvent(
                kind="roi_resolved",
                payload={
                    "centroid": {"lat": 48.71, "lon": 2.20},
                    "vertices": node.geometry_latlon,
                },
            )
        )
        viewer.render_graph(graph)
    assert mock_rr.init.called
    assert mock_rr.LineStrips3D.called
    assert not mock_rr.Boxes3D.called

    with patch.dict("sys.modules", {"rerun": mock_rr}):
        from herald.viewer import rerun_view

        viewer_with_boxes = rerun_view.RerunSceneViewer(
            spawn=False, show_building_boxes=True
        )
        viewer_with_boxes.set_frame(frame)
        viewer_with_boxes.render_graph(graph)
    assert mock_rr.Boxes3D.called
    for call in mock_rr.Boxes3D.call_args_list:
        assert "rotations" not in call.kwargs
