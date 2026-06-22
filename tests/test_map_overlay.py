"""Tests for Folium map overlay builder."""

import numpy as np

from herald.scene.common.geometry import Frame
from herald.scene.common.geometry import Geometry
from herald.scene.common.graph import SceneGraph, SceneNode
from herald.scene.common.roi import ROI
from herald.ui.map_overlay import _write_overlay_map_html


def test_write_overlay_map_html_writes_file(tmp_path):
    roi = ROI.from_polygon(
        [(48.710, 2.200), (48.710, 2.203), (48.713, 2.203), (48.713, 2.200)]
    )
    osm_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "way/1",
                "properties": {
                    "name": "Hall A",
                    "primary_tag": "building",
                    "primary_value": "university",
                    "building": "university",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [2.201, 48.711],
                            [2.2015, 48.711],
                            [2.2015, 48.7115],
                            [2.201, 48.7115],
                            [2.201, 48.711],
                        ]
                    ],
                },
            }
        ],
    }
    frame = Frame.from_origin(48.7115, 2.2015)
    ring = [
        (48.7105, 2.2005),
        (48.7105, 2.2015),
        (48.7115, 2.2015),
        (48.7115, 2.2005),
        (48.7105, 2.2005),
    ]
    coords = np.array(
        [[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in ring],
        dtype=np.float64,
    )
    graph = SceneGraph(
        emb_model_id="stub-v0",
        nodes=[
            SceneNode(
                id="region-uuid",
                type="region",
                pid="site-uuid",
                geom=Geometry(type="polygon", coords=coords, frame="ENU"),
                desc="quad",
            )
        ],
    )
    out_html = tmp_path / "map_overlay.html"
    _write_overlay_map_html(
        out_html,
        roi=roi,
        frame=frame,
        osm_polygons_geojson=osm_geojson,
        graph=graph,
    )
    text = out_html.read_text(encoding="utf-8")
    assert "Hall A" in text or "building" in text
    assert len(text) > 5000
