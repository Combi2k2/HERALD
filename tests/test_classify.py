"""Tests for OSM template classification."""

from herald.scene.init.classify import classify_from_osm_template
from herald.scene.init.hierarchy import HierarchyNode


def _node(tags: dict[str, str]) -> HierarchyNode:
    return HierarchyNode(
        osm_id=1,
        osm_type="way",
        tags=tags,
        geom=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
        area=100.0,
    )


def test_building_template_skips_vlm():
    result = classify_from_osm_template(_node({"building": "university", "name": "Lab"}))
    assert result is not None
    assert result.role == "structure"
    assert result.source == "osm_template"
    assert result.name == "Lab"


def test_untagged_returns_none():
    assert classify_from_osm_template(_node({"source": "cadastre"})) is None


def test_parking_template():
    result = classify_from_osm_template(_node({"amenity": "parking"}))
    assert result is not None
    assert result.category == "parking"
