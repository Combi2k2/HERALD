"""VLM client for single-polygon classification (provider-agnostic via LangChain)."""

from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from typing import Any

from PIL import Image

from services.vlm.schema import EntityClassification

DEFAULT_MODEL = "ollama:qwen2.5vl:3b"

_SYSTEM_PROMPT = """You classify outdoor campus map polygons for a robot navigation scene graph.

Image 1 is satellite/aerial imagery. Image 2 is an OpenStreetMap cartographic basemap.
The target polygon is highlighted in red on both images.

Use OSM tag context as primary evidence and both images to confirm appearance.
Return only semantic fields (role, category, function, name, desc, confidence).

Guidelines:
- role=structure for buildings
- role=region_use for landcover / terrain (grass, forest, open ground)
- role=facility for parking, sports pitches, amenity areas
- role=obstacle for fences / barriers
- role=infrastructure for roads, utilities, tanks
- function=none for pure landcover unless a specific use is evident
- name="" when no proper name exists
- desc: one concise sentence, max 240 characters
"""


def _image_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _image_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(_image_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_user_prompt(context: dict[str, Any]) -> str:
    schema_json = json.dumps(EntityClassification.model_json_schema(), indent=2)
    candidate_json = json.dumps(context, indent=2)
    return (
        "Classify the highlighted polygon using OSM tags plus both images.\n"
        "Return one JSON object matching the schema (semantic fields only).\n\n"
        f"Expected JSON schema:\n{schema_json}\n\n"
        f"Candidate (metadata only, no coordinates):\n{candidate_json}"
    )


def _build_messages(
    aerial: Image.Image,
    osm_map: Image.Image,
    context: dict[str, Any],
) -> list[Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": _build_user_prompt(context)},
                {"type": "image_url", "image_url": {"url": _image_data_url(aerial)}},
                {"type": "image_url", "image_url": {"url": _image_data_url(osm_map)}},
            ]
        ),
    ]


@lru_cache(maxsize=16)
def _structured_model_cached(
    model: str,
    base_url: str | None,
    temperature: float,
) -> Any:
    """Build a LangChain chat model with ``with_structured_output(EntityClassification)``."""
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {"temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    chat = init_chat_model(model, **kwargs)
    return chat.with_structured_output(EntityClassification)


def classify_polygon(
    aerial: Image.Image,
    osm_map: Image.Image,
    context: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    temperature: float = 0.0,
    structured_model: Any | None = None,
) -> EntityClassification | None:
    """Classify one polygon using a vision chat model + structured output.

    ``model`` uses LangChain's ``init_chat_model`` format, e.g.:

    - ``ollama:qwen2.5vl:3b`` (default; local Ollama)
    - ``openai:gpt-4o`` (requires ``OPENAI_API_KEY``)
    - ``anthropic:claude-sonnet-4-6`` (requires ``ANTHROPIC_API_KEY``)

    Pass ``structured_model`` in tests to inject a mock runnable.
    """
    runnable = structured_model or _structured_model_cached(model, base_url, temperature)
    try:
        result = runnable.invoke(_build_messages(aerial, osm_map, context))
    except Exception:
        return None
    if isinstance(result, EntityClassification):
        return result
    if isinstance(result, dict):
        return EntityClassification.model_validate(result)
    return None

