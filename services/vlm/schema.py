"""Private Pydantic schema for Ollama structured VLM output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    structure = "structure"
    region_use = "region_use"
    facility = "facility"
    obstacle = "obstacle"
    access = "access"
    infrastructure = "infrastructure"
    unclassified = "unclassified"


class Category(str, Enum):
    building = "building"
    vegetation = "vegetation"
    water = "water"
    recreation = "recreation"
    parking = "parking"
    plaza = "plaza"
    service_point = "service_point"
    charging = "charging"
    transport_node = "transport_node"
    landmark = "landmark"
    access_point = "access_point"
    ground_other = "ground_other"
    unknown = "unknown"


class Function(str, Enum):
    academic = "academic"
    library = "library"
    dining = "dining"
    retail = "retail"
    financial = "financial"
    healthcare = "healthcare"
    sports = "sports"
    culture = "culture"
    civic = "civic"
    residential = "residential"
    office = "office"
    religious = "religious"
    transport = "transport"
    utility = "utility"
    convenience = "convenience"
    waste = "waste"
    none = "none"
    unknown = "unknown"


class EntityClassification(BaseModel):
    """Structured VLM output for one polygon (identity comes from pipeline context)."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    category: Category
    function: Function
    name: str = Field("", description="empty string if none")
    desc: str = Field(..., max_length=240)
    confidence: float = Field(..., ge=0.0, le=1.0)
