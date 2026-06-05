"""Tests for OSM polygon containment hierarchy."""

from services.osm.client import OSMRawPolygon
from herald.scene.init.hierarchy import SITE_OSM_ID, build_containment_forest
from utils.osm_filter import polygon_disposition
from herald.scene.common.roi import ROI


def _poly(
    osm_id: int,
    ring: list[tuple[float, float]],
    tags: dict[str, str],
) -> OSMRawPolygon:
    primary = sorted(tags)[0]
    closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
    return OSMRawPolygon(
        osm_id=osm_id,
        osm_type="way",
        tags=tags,
        geometry=tuple(closed),
        primary_tag=primary,
        primary_value=tags[primary],
    )


def test_boundary_parcel_excluded():
    disposition = polygon_disposition(
        {"boundary": "parcel", "ref": "271"},
        area_m2=1000.0,
        min_area_m2=30.0,
        max_area_m2=80_000.0,
    )
    assert disposition == "boundary"


def test_building_inside_landuse_gets_parent():
    roi = ROI.from_bbox(48.710, 2.200, 48.713, 2.203)
    landuse = _poly(
        100,
        [(48.7108, 2.2008), (48.7108, 2.2018), (48.7118, 2.2018), (48.7118, 2.2008)],
        {"landuse": "grass"},
    )
    building = _poly(
        101,
        [(48.711, 2.201), (48.711, 2.2015), (48.7115, 2.2015), (48.7115, 2.201)],
        {"building": "university", "name": "Hall"},
    )
    forest = build_containment_forest([landuse, building], roi)
    assert SITE_OSM_ID in forest.nodes
    assert 100 in forest.site_children
    assert 101 not in forest.site_children
    assert forest.nodes[100].parent_osm_id == SITE_OSM_ID
    assert forest.nodes[101].parent_osm_id == 100
    assert forest.nodes[SITE_OSM_ID].parent_osm_id is None

