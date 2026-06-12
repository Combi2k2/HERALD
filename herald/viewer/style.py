"""Color and metadata styling for scene graph nodes in the viewer."""

from __future__ import annotations

from herald.scene.common.graph import SceneNode

# RGBA 0–255 keyed by classification category
CATEGORY_COLORS: dict[str, list[int]] = {
    "building": [96, 165, 250],
    "vegetation": [74, 222, 128],
    "water": [34, 211, 238],
    "recreation": [192, 132, 252],
    "parking": [250, 204, 21],
    "plaza": [251, 191, 36],
    "service_point": [244, 114, 182],
    "ground_other": [161, 161, 170],
    "unknown": [113, 113, 122],
}

ROLE_COLORS: dict[str, list[int]] = {
    "structure": CATEGORY_COLORS["building"],
    "region_use": CATEGORY_COLORS["vegetation"],
    "facility": CATEGORY_COLORS["service_point"],
    "site": CATEGORY_COLORS["ground_other"],
    "unclassified": CATEGORY_COLORS["unknown"],
}

COLOR_PATHWAY = [251, 146, 60]


def _primary_osm_ref(node: SceneNode):
    for ref in node.refs:
        if ref.assigned_by == "osm":
            return ref
    return None


def color_for_node(node: SceneNode) -> list[int]:
    if node.category and node.category in CATEGORY_COLORS:
        return list(CATEGORY_COLORS[node.category])
    if node.role and node.role in ROLE_COLORS:
        return list(ROLE_COLORS[node.role])
    if node.type == "structure":
        return list(CATEGORY_COLORS["building"])
    return list(CATEGORY_COLORS["unknown"])


def metadata_lines(node: SceneNode) -> list[str]:
    lines = [f"id: {node.id}", f"type: {node.type}"]
    if node.role:
        lines.append(f"role: {node.role}")
    osm_ref = _primary_osm_ref(node)
    if osm_ref and osm_ref.assigned_id:
        lines.append(f"osm: {osm_ref.assigned_id}")
    if node.category:
        lines.append(f"category: {node.category}")
    if node.function:
        lines.append(f"function: {node.function}")
    if node.name:
        lines.append(f"name: {node.name}")
    if node.geom.offset[2] != 0.0 and node.role == "structure":
        lines.append(f"height: {node.geom.offset[2]:.1f}")
    if node.desc:
        lines.append(f"desc: {node.desc}")
    return lines
