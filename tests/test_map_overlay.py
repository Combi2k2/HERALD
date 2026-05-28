"""Tests for Folium map overlay builder."""

from herald.osm.client import LatLon
from herald.scene.frame import LocalFrame
from herald.scene.graph import SceneGraph, SceneNode
from herald.scene.roi import ROI
from herald.ui.map_overlay import _write_overlay_map_html


def test_write_overlay_map_html_writes_file(tmp_path):
    roi = ROI.from_bbox(48.710, 2.200, 48.713, 2.203)
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
    graph = SceneGraph(
        site_id="site_000",
        frame=LocalFrame(origin=LatLon(lat=48.7115, lon=2.2015)),
        embedding_model_id="stub-v0",
        nodes=[
            SceneNode(
                id="outdoor_000",
                level="outdoor_region",
                zone_kind="outdoor_region",
                text="quad",
                geometry_latlon=[
                    (48.7105, 2.2005),
                    (48.7105, 2.2015),
                    (48.7115, 2.2015),
                    (48.7115, 2.2005),
                    (48.7105, 2.2005),
                ],
            )
        ],
    )
    out_html = tmp_path / "map_overlay.html"
    _write_overlay_map_html(
        out_html,
        roi=roi,
        osm_polygons_geojson=osm_geojson,
        graph=graph,
        pathways_geojson={"type": "FeatureCollection", "features": []},
    )
    text = out_html.read_text(encoding="utf-8")
    assert "Hall A" in text or "building" in text
    assert len(text) > 5000
