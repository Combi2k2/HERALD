"""Tests for per-polygon dual-image classification."""

from unittest.mock import patch

from PIL import Image

from herald.scene.init.classify import classify_hierarchy_nodes
from herald.scene.init.hierarchy import HierarchyNode
from services.aerial.fetch import AerialMeta
from services.vlm.schema import (
    Category,
    EntityClassification,
    Function,
    Role,
)


def _node(osm_id: int, tags: dict[str, str]) -> HierarchyNode:
    return HierarchyNode(
        osm_id=osm_id,
        osm_type="way",
        tags=tags,
        geom=[(48.711, 2.201), (48.711, 2.202), (48.712, 2.202), (48.712, 2.201), (48.711, 2.201)],
        area=100.0,
    )


def _meta() -> AerialMeta:
    return AerialMeta(
        bbox=(48.710, 2.200, 48.713, 2.203),
        zoom=18,
        width_px=512,
        height_px=512,
        provider="test",
    )


@patch("services.vlm.client.classify_polygon")
@patch("herald.scene.init.classify.crop_polygon_views")
def test_classify_hierarchy_nodes_vlm_per_polygon(mock_crop, mock_vlm):
    mock_crop.return_value = (
        Image.new("RGB", (32, 32)),
        Image.new("RGB", (32, 32)),
        [(4.0, 4.0), (28.0, 4.0), (28.0, 28.0)],
    )
    mock_vlm.return_value = EntityClassification(
        role=Role.region_use,
        category=Category.ground_other,
        function=Function.none,
        name="",
        desc="open ground",
        confidence=0.8,
    )

    nodes = [_node(5, {"source": "survey"})]
    results = classify_hierarchy_nodes(
        nodes,
        use_vlm=True,
        aerial_image=Image.new("RGB", (512, 512)),
        aerial_meta=_meta(),
        show_progress=False,
    )

    assert mock_vlm.call_count == 1
    assert results[5].source == "vlm"
    assert results[5].description == "open ground"


def test_classify_hierarchy_nodes_without_vlm_uses_template():
    nodes = [_node(1, {"building": "university", "name": "Hall"})]
    results = classify_hierarchy_nodes(
        nodes,
        use_vlm=False,
        show_progress=False,
    )
    assert results[1].source == "osm_template"
    assert results[1].role == "structure"


def test_classify_hierarchy_nodes_without_vlm_fallback():
    nodes = [_node(2, {"source": "cadastre"})]
    results = classify_hierarchy_nodes(
        nodes,
        use_vlm=False,
        show_progress=False,
    )
    assert results[2].source == "fallback"
