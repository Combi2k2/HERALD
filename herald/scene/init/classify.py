"""Polygon classification: OSM templates or per-polygon dual-image VLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import requests
from PIL import Image

from herald.scene.init.hierarchy import SITE_OSM_ID, HierarchyNode
from services.aerial.fetch import AerialMeta, crop_polygon_views, highlight_polygon
from services.vlm.schema import EntityClassification
from utils.osm_helpers import context_tags, has_strong_classification_tag


@dataclass(frozen=True)
class NodeClassification:
    """Public classification record (plain strings, no enum types)."""

    role: str
    category: str
    function: str
    name: str
    description: str
    confidence: float
    source: str


def _pick_name(tags: dict[str, str]) -> str:
    for key in ("name", "alt_name", "short_name", "ref"):
        value = tags.get(key)
        if value:
            return value
    return ""


def _building_function(tags: dict[str, str]) -> str:
    building = tags.get("building", "yes")
    mapping = {
        "university": "academic",
        "college": "academic",
        "school": "academic",
        "dormitory": "residential",
        "residential": "residential",
        "garage": "utility",
        "service": "utility",
        "roof": "utility",
        "construction": "unknown",
    }
    if building in mapping:
        return mapping[building]
    amenity = tags.get("amenity", "")
    if amenity in {"university", "school", "college"}:
        return "academic"
    if amenity in {"restaurant", "fast_food", "cafe"}:
        return "dining"
    return "unknown"


def _template_from_tags(node: HierarchyNode) -> NodeClassification | None:
    tags = node.tags
    if not has_strong_classification_tag(tags):
        return None

    name = _pick_name(tags)

    if tags.get("building") or tags.get("building:part"):
        fn = _building_function(tags)
        desc = name or f"{tags.get('building', 'yes')} building"
        if tags.get("building:levels"):
            desc = f"{desc}, {tags['building:levels']} levels"
        return NodeClassification(
            role="structure",
            category="building",
            function=fn,
            name=name,
            description=desc[:240],
            confidence=0.95,
            source="osm_template",
        )

    if tags.get("amenity") in {"parking", "parking_space"}:
        return NodeClassification(
            role="facility",
            category="parking",
            function="transport",
            name=name,
            description=name or "vehicle parking area",
            confidence=0.95,
            source="osm_template",
        )

    if tags.get("leisure") in {"pitch", "sports_centre", "fitness_station", "swimming_pool"}:
        sport = tags.get("sport", tags["leisure"])
        desc = name or f"{tags['leisure']} area ({sport})"
        return NodeClassification(
            role="facility",
            category="recreation",
            function="sports",
            name=name,
            description=desc[:240],
            confidence=0.95,
            source="osm_template",
        )

    if tags.get("leisure") in {"park", "garden", "playground"}:
        return NodeClassification(
            role="region_use",
            category="vegetation" if tags.get("leisure") == "garden" else "recreation",
            function="none",
            name=name,
            description=name or f"{tags['leisure']} area",
            confidence=0.9,
            source="osm_template",
        )

    landuse = tags.get("landuse")
    if landuse:
        category = "vegetation"
        if landuse in {"reservoir", "basin", "pond"}:
            category = "water"
        fn = "none"
        if landuse in {"commercial", "retail"}:
            fn = "retail"
        return NodeClassification(
            role="region_use",
            category=category,
            function=fn,
            name=name,
            description=name or f"{landuse} land",
            confidence=0.9,
            source="osm_template",
        )

    natural = tags.get("natural")
    if natural:
        category = "water" if natural == "water" else "vegetation"
        return NodeClassification(
            role="region_use",
            category=category,
            function="none",
            name=name,
            description=name or f"{natural} area",
            confidence=0.9,
            source="osm_template",
        )

    amenity = tags.get("amenity")
    if amenity:
        fn_map = {
            "university": "academic",
            "school": "academic",
            "charging_station": "transport",
            "shelter": "transport",
        }
        return NodeClassification(
            role="facility",
            category="service_point",
            function=fn_map.get(amenity, "unknown"),
            name=name,
            description=name or f"{amenity} facility",
            confidence=0.9,
            source="osm_template",
        )

    return None


def _entity_to_public(item: EntityClassification, *, source: str) -> NodeClassification:
    return NodeClassification(
        role=item.role.value,
        category=item.category.value,
        function=item.function.value,
        name=item.name,
        description=item.desc,
        confidence=item.confidence,
        source=source,
    )


def _fallback_unclassified(node: HierarchyNode) -> NodeClassification:
    tags = context_tags(node.tags)
    hint = ", ".join(f"{k}={v}" for k, v in sorted(tags.items())[:4])
    desc = f"unclassified area ({hint})" if hint else "unclassified area"
    return NodeClassification(
        role="unclassified",
        category="unknown",
        function="unknown",
        name=_pick_name(node.tags),
        description=desc[:240],
        confidence=0.3,
        source="fallback",
    )


def build_vlm_context(
    node: HierarchyNode,
    *,
    child_count: int = 0,
    depth: int = 0,
) -> dict[str, Any]:
    tags = context_tags(node.tags)
    ctx: dict[str, Any] = {
        "osm_id": node.osm_id,
        "osm_type": node.osm_type,
        "area_m2": round(node.area, 1),
        "child_count": child_count,
        "depth": depth,
        "tags": tags,
        "name": _pick_name(node.tags),
    }
    if node.parent_osm_id is not None:
        ctx["parent_osm_id"] = node.parent_osm_id
    return ctx


def classify_from_osm_template(node: HierarchyNode) -> NodeClassification | None:
    return _template_from_tags(node)


def fallback_classification(node: HierarchyNode) -> NodeClassification:
    return _fallback_unclassified(node)


def _classify_one_with_vlm(
    node: HierarchyNode,
    *,
    aerial_image: Image.Image,
    aerial_meta: AerialMeta,
    depth: int,
    vlm_model: str,
    http: requests.Session,
    osm_cache: dict[tuple[int, int, int, int, int], Image.Image],
) -> NodeClassification:
    from services.vlm.client import classify_polygon

    context = build_vlm_context(
        node,
        child_count=len(node.child_osm_ids),
        depth=depth,
    )
    try:
        aerial_crop, osm_crop, local_poly = crop_polygon_views(
            aerial_image,
            aerial_meta,
            list(node.geom),
            session=http,
            osm_cache=osm_cache,
        )
        aerial_view = highlight_polygon(aerial_crop, local_poly)
        osm_view = highlight_polygon(osm_crop, local_poly)
        item = classify_polygon(
            aerial_view,
            osm_view,
            context,
            model=vlm_model,
        )
        if item is not None:
            return _entity_to_public(item, source="vlm")
    except Exception:
        pass

    template = _template_from_tags(node)
    if template is not None:
        return template
    return _fallback_unclassified(node)


def classify_hierarchy_nodes(
    nodes: list[HierarchyNode],
    *,
    use_vlm: bool,
    aerial_image: Image.Image | None = None,
    aerial_meta: AerialMeta | None = None,
    vlm_model: str | None = None,
    depths: dict[int, int] | None = None,
    show_progress: bool = True,
    on_node_classified: Callable[[HierarchyNode, NodeClassification], None] | None = None,
) -> dict[int, NodeClassification]:
    """Classify hierarchy polygons.

    Without VLM: OSM tag templates, then fallback (descriptions are embedded later).
    With VLM: one dual-image (aerial + OSM map) call per polygon.
    """
    from herald.scene.common.progress import iter_progress
    from services.vlm.client import DEFAULT_MODEL

    if not nodes:
        return {}

    depth_map = depths or {}
    model = vlm_model or DEFAULT_MODEL
    vlm_ready = (
        use_vlm and aerial_image is not None and aerial_meta is not None
    )

    results: dict[int, NodeClassification] = {}
    http = requests.Session() if vlm_ready else None
    osm_cache: dict[tuple[int, int, int, int, int], Image.Image] = {}

    for node in iter_progress(
        nodes,
        desc="Classifying polygons",
        total=len(nodes),
        disable=not show_progress,
    ):
        if node.osm_id == SITE_OSM_ID:
            continue

        if vlm_ready and http is not None:
            classification = _classify_one_with_vlm(
                node,
                aerial_image=aerial_image,
                aerial_meta=aerial_meta,
                depth=depth_map.get(node.osm_id, 0),
                vlm_model=model,
                http=http,
                osm_cache=osm_cache,
            )
        else:
            classification = _template_from_tags(node) or _fallback_unclassified(node)

        results[node.osm_id] = classification
        if on_node_classified is not None:
            on_node_classified(node, classification)

    return results


def level_for_role(role: str) -> str:
    if role == "structure":
        return "building"
    return "outdoor_region"


def zone_kind_for_role(role: str) -> str:
    if role == "structure":
        return "building"
    return "outdoor_region"
