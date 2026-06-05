"""Vision-language model classification."""

from services.vlm.client import DEFAULT_MODEL, classify_polygon
from services.vlm.schema import EntityClassification

__all__ = ["DEFAULT_MODEL", "EntityClassification", "classify_polygon"]
