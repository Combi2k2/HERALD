"""Tests for ROI polygon helpers."""

from herald.scene.common.roi import ROI

_SMALL_BOX = [
    (48.71, 2.20),
    (48.71, 2.202),
    (48.712, 2.202),
    (48.712, 2.20),
]


def test_from_polygon_area_reasonable():
    # ~200m square near equator is roughly 4e4 m²
    roi = ROI.from_polygon(_SMALL_BOX)
    area = roi.area()
    assert 1_000 < area < 500_000


def test_latlon_vertices_round_trip():
    roi = ROI.from_polygon(_SMALL_BOX)
    verts = roi.latlon_vertices()
    assert len(verts) >= 4
    assert verts[0] == verts[-1]
