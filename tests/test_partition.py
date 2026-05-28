"""Tests for outdoor zone partitioning."""

from herald.osm.client import OSMFeature
from herald.scene.partition import (
    extract_buildings,
    filter_walkable_highways,
    partition_outdoor_zones,
)
from herald.scene.roi import ROI


def _building(id_: int, ring: list[tuple[float, float]]) -> OSMFeature:
    closed = ring if ring[0] == ring[-1] else ring + [ring[0]]
    return OSMFeature(
        osm_id=id_,
        osm_type="way",
        tags={"building": "university", "name": f"B{id_}"},
        geometry=tuple(closed),
        kind="polygon",
        category="building",
    )


def _footway(id_: int, line: list[tuple[float, float]]) -> OSMFeature:
    return OSMFeature(
        osm_id=id_,
        osm_type="way",
        tags={"highway": "footway"},
        geometry=tuple(line),
        kind="linestring",
        category="highway",
    )


def test_filter_walkable_highways():
    hw = [
        _footway(1, [(48.711, 2.201), (48.7115, 2.2015)]),
        OSMFeature(
            osm_id=2,
            osm_type="way",
            tags={"highway": "motorway"},
            geometry=((48.71, 2.20), (48.72, 2.21)),
            kind="linestring",
            category="highway",
        ),
    ]
    walkable = filter_walkable_highways(hw)
    assert len(walkable) == 1
    assert walkable[0].osm_id == 1


def test_partition_assigns_building_to_outdoor_zone():
    roi = ROI.from_bbox(48.710, 2.200, 48.713, 2.203)
    b1 = _building(10, [(48.7105, 2.2005), (48.7105, 2.2010), (48.7110, 2.2010), (48.7110, 2.2005)])
    b2 = _building(20, [(48.7115, 2.2015), (48.7115, 2.2020), (48.7120, 2.2020), (48.7120, 2.2015)])
    path = _footway(30, [(48.710, 2.20125), (48.713, 2.20125)])
    buildings = extract_buildings([b1, b2])
    result = partition_outdoor_zones(roi, buildings, [path])
    assert len(result.buildings) == 2
    assert 10 in result.building_to_outdoor
    assert 20 in result.building_to_outdoor
    assert len(result.outdoor_zones) >= 1
