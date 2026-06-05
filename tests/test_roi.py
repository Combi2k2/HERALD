"""Tests for ROI polygon helpers."""

from herald.scene.common.roi import ROI


def test_from_bbox_area_reasonable():
    # ~200m square near equator is roughly 4e4 m²
    roi = ROI.from_bbox(48.71, 2.20, 48.712, 2.202)
    area = roi.area_m2()
    assert 1_000 < area < 500_000


def test_latlon_vertices_round_trip():
    roi = ROI.from_bbox(48.71, 2.20, 48.712, 2.202)
    verts = roi.latlon_vertices()
    assert len(verts) >= 4
    assert verts[0] == verts[-1]


def test_clip_features_drops_outside():
    from services.osm.client import OSMFeature

    roi = ROI.from_bbox(48.71, 2.20, 48.712, 2.202)
    inside = OSMFeature(
        osm_id=1,
        osm_type="way",
        tags={"building": "yes"},
        geometry=(
            (48.711, 2.201),
            (48.711, 2.2015),
            (48.7115, 2.2015),
            (48.7115, 2.201),
            (48.711, 2.201),
        ),
        kind="polygon",
        category="building",
    )
    outside = OSMFeature(
        osm_id=2,
        osm_type="way",
        tags={"building": "yes"},
        geometry=(
            (48.72, 2.21),
            (48.72, 2.211),
            (48.721, 2.211),
            (48.721, 2.21),
            (48.72, 2.21),
        ),
        kind="polygon",
        category="building",
    )
    clipped = roi.clip_features([inside, outside])
    assert len(clipped) == 1
    assert clipped[0].osm_id == 1
