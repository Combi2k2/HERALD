"""Smoke tests for Rerun viewer integration."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from herald.scene.common.geometry import Frame
from herald.scene.common._legacy_events import SceneEvent
from herald.scene.common.geometry import Geometry, Vec3
from herald.scene.common.graph import SceneGraph, SceneNode, SourceRef
from herald.viewer.rerun_view import _closed_ring


def test_closed_ring_appends_first_vertex():
    ring = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert _closed_ring(ring) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0)]


def test_publish_roi_event_without_rerun_import_at_module_level():
    frame = Frame.from_origin(48.71, 2.20)
    site_id = "site-uuid"
    site_ring = [
        (48.71, 2.20),
        (48.712, 2.20),
        (48.712, 2.202),
        (48.71, 2.202),
        (48.71, 2.20),
    ]
    site_coords = np.array(
        [[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in site_ring],
        dtype=np.float64,
    )
    node = SceneNode(
        id=site_id,
        type="site",
        pid=None,
        geom=Geometry(type="polygon", coords=site_coords, frame="ENU"),
        desc="site",
        role="site",
        category="ground_other",
    )
    building_coords = np.array(
        [
            [*frame.wgs2enu(lat, lon), 0.0]
            for lat, lon in [
                (48.711, 2.201),
                (48.711, 2.2015),
                (48.7115, 2.2015),
                (48.7115, 2.201),
                (48.711, 2.201),
            ]
        ],
        dtype=np.float64,
    )
    building = SceneNode(
        id="structure-uuid",
        type="structure",
        pid="region-uuid",
        geom=Geometry(
            type="polygon",
            coords=building_coords,
            frame="ENU",
            offset=Vec3((0.0, 0.0, 12.0)),
        ),
        refs=[
            SourceRef(
                assigned_by="osm",
                assigned_id="way/1",
                metadata={"assigned": "osm_template"},
            )
        ],
        role="structure",
        category="building",
        function="academic",
        desc="hall",
    )
    region_coords = np.array(
        [
            [*frame.wgs2enu(lat, lon), 0.0]
            for lat, lon in [
                (48.710, 2.200),
                (48.710, 2.202),
                (48.712, 2.202),
                (48.712, 2.200),
                (48.710, 2.200),
            ]
        ],
        dtype=np.float64,
    )
    region = SceneNode(
        id="region-uuid",
        type="region",
        pid=site_id,
        geom=Geometry(type="polygon", coords=region_coords, frame="ENU"),
        role="region_use",
        category="vegetation",
        function="none",
        desc="campus block",
        refs=[
            SourceRef(
                assigned_by="osm",
                assigned_id="way/99",
                metadata={"assigned": "osm_template"},
            )
        ],
    )
    graph = SceneGraph(emb_model_id="stub-v0", nodes=[node, region, building])
    graph.add_edge(site_id, "region-uuid")
    graph.add_edge("region-uuid", "structure-uuid")
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
                    "vertices": site_ring,
                },
            )
        )
        viewer.render_graph(graph, frame=frame)
    assert mock_rr.init.called
    assert mock_rr.LineStrips3D.called
    assert not mock_rr.Mesh3D.called
    assert not mock_rr.Arrows3D.called
