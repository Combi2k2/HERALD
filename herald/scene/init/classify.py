"""Populate semantic fields on scene graph nodes."""

from __future__ import annotations

import json
import math
from typing import Any

from enum import Enum

from PIL import Image, ImageDraw
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from herald.scene.common.progress import iter_progress
from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph, SceneNode
from services.vlm import VLMClient, image_block, text_block


class _Role(str, Enum):
    structure = "structure"
    region_use = "region_use"
    facility = "facility"
    obstacle = "obstacle"
    access = "access"
    infrastructure = "infrastructure"
    unclassified = "unclassified"

class _Category(str, Enum):
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

class _Function(str, Enum):
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
    """Structured VLM output for one polygon."""

    model_config = ConfigDict(extra="forbid")

    role: _Role
    category: _Category
    function: _Function
    name: str = Field("", description="empty string if none")
    desc: str = Field(..., max_length=240)


_TILE = 256
_PAD = 24
_RGB = (255, 40, 40)

_VLM_SYSTEM = """You classify outdoor campus map polygons for a robot navigation scene graph.

Image 1 is satellite/aerial imagery. Image 2 is an OpenStreetMap cartographic basemap.
The target polygon is highlighted in red on both images.

Use OSM tag context as primary evidence and both images to confirm appearance.
Return only semantic fields (role, category, function, name, desc).

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


def level_for_role(role: str) -> str:
    if role in ("site", "structure"):
        return role
    return "region"


def _tags(node: SceneNode) -> dict[str, str]:
    for ref in node.refs:
        if ref.assigned_by == "osm":
            raw = ref.metadata.get("tags")
            if isinstance(raw, dict):
                return raw
    return {}

def _name(tags: dict[str, str]) -> str:
    return tags.get("name") or \
        tags.get("alt_name") or \
        tags.get("short_name") or \
        tags.get("ref") or \
        ""

def _ctx(tags: dict[str, str]) -> dict[str, str]:
    skip = {"source", "check_date", "created_by", "note", "fixme", "todo", "description"}
    return {
        k: v
        for k, v in sorted(tags.items())
        if k not in skip and not k.startswith(("addr:", "ref:", "source:", "contact:"))
    }


def _fallback(tags: dict[str, str]) -> EntityClassification:
    name = _name(tags)

    if tags.get("building") or tags.get("building:part"):
        b = tags.get("building", "yes")
        fn = (
            "academic"         if b in {"university", "college", "school"}
            else "residential" if b in {"dormitory", "residential"}
            else "utility"     if b in {"garage", "service", "roof"}
            else "unknown"
        )
        desc = name or f"{b} building"
        if tags.get("building:levels"):
            desc = f"{desc}, {tags['building:levels']} levels"
        return EntityClassification(role="structure", category="building", function=fn, name=name, desc=desc[:240])

    if tags.get("amenity") in {"parking", "parking_space"}:
        return EntityClassification(
            role="facility", category="parking", function="transport",
            name=name, desc=name or "vehicle parking area",
        )

    leisure = tags.get("leisure")
    if leisure in {"pitch", "sports_centre", "fitness_station", "swimming_pool"}:
        return EntityClassification(
            role="facility", category="recreation", function="sports", name=name,
            desc=(name or f"{leisure} area ({tags.get('sport', leisure)})")[:240],
        )

    if leisure in {"park", "garden", "playground"}:
        cat = "vegetation" if leisure == "garden" else "recreation"
        return EntityClassification(
            role="region_use", category=cat, function="none",
            name=name, desc=name or f"{leisure} area",
        )

    landuse = tags.get("landuse")
    if landuse:
        cat = "water" if landuse in {"reservoir", "basin", "pond"} else "vegetation"
        fn = "retail" if landuse in {"commercial", "retail"} else "none"
        return EntityClassification(
            role="region_use", category=cat, function=fn,
            name=name, desc=name or f"{landuse} land",
        )

    natural = tags.get("natural")
    if natural:
        cat = "water" if natural == "water" else "vegetation"
        return EntityClassification(
            role="region_use", category=cat, function="none",
            name=name, desc=name or f"{natural} area",
        )

    amenity = tags.get("amenity")
    if amenity:
        fn = "unknown"
        if amenity in {"university", "school"}:
            fn = "academic"
        elif amenity in {"charging_station", "shelter"}:
            fn = "transport"
        return EntityClassification(
            role="facility", category="service_point", function=fn,
            name=name, desc=name or f"{amenity} facility",
        )

    ctx = _ctx(tags)
    hint = ", ".join(f"{k}={v}" for k, v in sorted(ctx.items())[:4])
    return EntityClassification(
        role="unclassified", category="unknown", function="unknown", name=name,
        desc=(f"unclassified area ({hint})" if hint else "unclassified area")[:240],
    )


def _classify(
    client: VLMClient | None,
    images: list[Image.Image],
    context: dict[str, Any],
    tags: dict[str, str],
) -> EntityClassification:
    if client is not None:
        schema_json = json.dumps(EntityClassification.model_json_schema(), indent=2)
        candidate_json = json.dumps(context, indent=2)
        prompt = (
            "Classify the highlighted polygon using OSM tags plus the images.\n"
            "Return one JSON object matching the schema (semantic fields only).\n\n"
            f"Expected JSON schema:\n{schema_json}\n\n"
            f"Candidate (metadata only, no coordinates):\n{candidate_json}"
        )
        try:
            result = client.invoke([
                SystemMessage(content=_VLM_SYSTEM),
                HumanMessage(content=[text_block(prompt), *[image_block(img) for img in images]]),
            ], schema=EntityClassification)
            if isinstance(result, EntityClassification):
                return result
        except Exception:
            pass

    return _fallback(tags)

def _crop_box(
    node: SceneNode,
    frame: Frame,
    bbox: tuple[float, float, float, float],
    zoom: int,
    W: int,
    H: int
) -> tuple[tuple[int, int, int, int], list[tuple[float, float]]]:
    south, west, north, east = bbox
    n = 2.0**zoom

    def tile_xy(lat: float, lon: float) -> tuple[float, float]:
        lat_r = math.radians(lat)
        tx = (lon + 180.0) / 360.0 * n
        ty = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
        return tx, ty

    tx0 = math.floor(tile_xy(south, west)[0])
    ty0 = math.floor(tile_xy(north, east)[1])

    def mosaic_px(lat: float, lon: float) -> tuple[float, float]:
        tx, ty = tile_xy(lat, lon)
        return (tx - tx0) * _TILE, (ty - ty0) * _TILE

    ring = [[p[0], p[1]] for p in node.geom.coords]
    ring = ring + ([ring[0]] if ring[0] != ring[-1] else [])

    if len(ring) < 3:
        raise ValueError("Polygon must have at least 3 vertices")

    if node.geom.frame == "UTM":
        ring = [frame.utm2wgs(*p) for p in ring]
    if node.geom.frame == "ENU":
        ring = [frame.enu2wgs(*p) for p in ring]

    ring = [mosaic_px(lat, lon) for lat, lon in ring[:-1]]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]

    L = max(0, min(int(math.floor(min(xs))) - _PAD, W-1))
    T = max(0, min(int(math.floor(min(ys))) - _PAD, H-1))
    R = min(W, max(int(math.ceil(max(xs))) + _PAD, L+1))
    B = min(H, max(int(math.ceil(max(ys))) + _PAD, T+1))

    return (L, T, R, B), [(x - L, y - T) for x, y in ring]

def _crop_highlight(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    ring: list[tuple[float, float]],
) -> Image.Image:
    crop = image.crop(bbox)
    if len(ring) < 3:
        return crop
    out = crop.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(ring, fill=(*_RGB, 140), outline=(*_RGB, 255), width=3)
    return Image.alpha_composite(out, layer).convert("RGB")


def build_feat(
    graph: SceneGraph,
    frame: Frame,
    client: VLMClient | None,
    aerial: Image.Image | None,
    osmMap: Image.Image | None,
    *,
    mosaic_bbox: tuple[float, float, float, float] | None = None,
    mosaic_zoom: int = 18,
    show_progress: bool = True,
) -> None:
    if client is not None and aerial is not None and osmMap is not None:
        if osmMap.size != aerial.size:
            raise ValueError("OSM map and aerial image must have the same dimensions")
        if mosaic_bbox is None:
            raise ValueError("mosaic_bbox is required when using VLM classification")

    for node in iter_progress(graph.nodes, desc="Classifying polygons", total=len(graph.nodes), disable=not show_progress):
        if node.type == "site":
            node.role, node.category, node.function = "site", "ground_other", "none"
            node.desc = f"site at ({frame.lat:.5f}, {frame.lon:.5f})"
            continue

        tags = _tags(node)
        ctx: dict[str, Any] = {
            "area": round(float(node.refs[0].metadata.get("area", 0)), 1) if node.refs else 0,
            "tags": _ctx(tags),
            "name": _name(tags),
        }
        if client is not None and aerial is not None and osmMap is not None and mosaic_bbox is not None:
            bbox, ring = _crop_box(node, frame, mosaic_bbox, mosaic_zoom, aerial.width, aerial.height)
            views = [
                _crop_highlight(aerial, bbox, ring),
                _crop_highlight(osmMap, bbox, ring),
            ]
            item = _classify(client, views, ctx, tags)
        else:
            item = _fallback(tags)
        node.role = item.role
        node.category = item.category
        node.function = item.function
        node.name = item.name
        node.desc = item.desc
