"""Tests for LangChain-backed VLM client."""

from unittest.mock import MagicMock

from PIL import Image

from services.vlm.client import classify_polygon
from services.vlm.schema import (
    Category,
    EntityClassification,
    Function,
    Role,
)


def test_classify_polygon_uses_structured_invoke_with_dual_images():
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = EntityClassification(
        role=Role.structure,
        category=Category.building,
        function=Function.academic,
        name="Hall",
        desc="University building",
        confidence=0.9,
    )

    aerial = Image.new("RGB", (64, 64), color=(120, 120, 120))
    osm_map = Image.new("RGB", (64, 64), color=(200, 200, 200))
    context = {"osm_id": 42, "tags": {"building": "university"}}

    item = classify_polygon(
        aerial,
        osm_map,
        context,
        structured_model=mock_runnable,
    )

    assert item is not None
    assert item.role.value == "structure"
    mock_runnable.invoke.assert_called_once()
    messages = mock_runnable.invoke.call_args.args[0]
    assert messages[0].type == "system"
    human = messages[1]
    assert human.type == "human"
    image_parts = [p for p in human.content if p.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_classify_polygon_returns_none_on_invoke_error():
    mock_runnable = MagicMock()
    mock_runnable.invoke.side_effect = RuntimeError("model down")
    aerial = Image.new("RGB", (8, 8))
    osm_map = Image.new("RGB", (8, 8))
    assert (
        classify_polygon(
            aerial,
            osm_map,
            {"tags": {}},
            structured_model=mock_runnable,
        )
        is None
    )

