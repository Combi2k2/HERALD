"""Visual styling for Rerun scene graph rendering."""

from __future__ import annotations

from herald.scene.common.graph import SceneNode

# RGBA 0–255 keyed by classification category
CATEGORY_COLORS: dict[str, list[int]] = {
    "building": [59, 130, 246, 230],
    "vegetation": [34, 197, 94, 210],
    "water": [6, 182, 212, 220],
    "recreation": [168, 85, 247, 220],
    "parking": [100, 116, 139, 220],
    "plaza": [251, 191, 36, 210],
    "service_point": [244, 114, 182, 210],
    "charging": [16, 185, 129, 210],
    "transport_node": [249, 115, 22, 220],
    "landmark": [236, 72, 153, 220],
    "access_point": [148, 163, 184, 200],
    "ground_other": [161, 161, 170, 190],
    "unknown": [113, 113, 122, 180],
}

ROLE_COLORS: dict[str, list[int]] = {
    "structure": CATEGORY_COLORS["building"],
    "region_use": CATEGORY_COLORS["vegetation"],
    "facility": CATEGORY_COLORS["service_point"],
    "obstacle": [239, 68, 68, 180],
    "access": [250, 204, 21, 200],
    "infrastructure": [148, 163, 184, 190],
    "unclassified": CATEGORY_COLORS["unknown"],
}

COLOR_ROI = [255, 255, 255, 180]
COLOR_PATHWAY = [234, 88, 12, 255]
DEFAULT_COLOR = [156, 163, 175, 180]


def color_for_node(node: SceneNode) -> list[int]:
    if node.level == "site":
        return COLOR_ROI
    if node.category and node.category in CATEGORY_COLORS:
        return list(CATEGORY_COLORS[node.category])
    if node.role and node.role in ROLE_COLORS:
        return list(ROLE_COLORS[node.role])
    if node.level == "building":
        return list(CATEGORY_COLORS["building"])
    return list(DEFAULT_COLOR)


def metadata_lines(node: SceneNode) -> list[str]:
    lines = [
        f"id: {node.id}",
        f"level: {node.level}",
    ]
    if node.osm_id is not None:
        lines.append(f"osm_id: {node.osm_id}")
    if node.role:
        lines.append(f"role: {node.role}")
    if node.category:
        lines.append(f"category: {node.category}")
    if node.function:
        lines.append(f"function: {node.function}")
    if node.name:
        lines.append(f"name: {node.name}")
    if node.confidence is not None:
        lines.append(f"confidence: {node.confidence:.2f}")
    if node.classification_source:
        lines.append(f"source: {node.classification_source}")
    if node.height_m and node.role == "structure":
        lines.append(f"height_m: {node.height_m:.1f} ({node.height_source})")
    if node.text:
        lines.append(f"desc: {node.text}")
    if node.osm_tags:
        tag_preview = ", ".join(f"{k}={v}" for k, v in sorted(node.osm_tags.items())[:6])
        if tag_preview:
            lines.append(f"tags: {tag_preview}")
    return lines
