"""Populate semantic fields on scene graph nodes."""

from __future__ import annotations

import io
import math
from typing import Any

import requests
from PIL import Image, ImageDraw

from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph, SceneNode
from herald.scene.common.progress import iter_progress

_STRONG_TAGS = frozenset(
    {
        "building",
        "building:part",
        "amenity",
        "leisure",
        "landuse",
        "natural",
        "tourism",
        "shop",
        "office",
        "healthcare",
        "historic",
        "sport",
        "water",
    }
)
_TILE = 256
_HIGHLIGHT = (255, 40, 40, 140)


def level_for_role(role: str) -> str:
    if role in ("site", "structure"):
        return role
    return "region"


def _ring(node: SceneNode) -> list[tuple[float, float]]:
    ring = [(float(r[0]), float(r[1])) for r in node.geom.coords]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _tags(node: SceneNode) -> dict[str, str]:
    for ref in node.refs:
        if ref.assigned_by == "osm":
            raw = ref.metadata.get("tags")
            if isinstance(raw, dict):
                return dict(raw)
    return {}


def _ctx(tags: dict[str, str]) -> dict[str, str]:
    skip = {"source", "check_date", "created_by", "note", "fixme", "todo", "description"}
    return {
        k: v
        for k, v in sorted(tags.items())
        if k not in skip and not k.startswith(("addr:", "ref:", "source:", "contact:"))
    }


def _pick_name(tags: dict[str, str]) -> str:
    for key in ("name", "alt_name", "short_name", "ref"):
        if tags.get(key):
            return tags[key]
    return ""


def _apply_template(node: SceneNode, tags: dict[str, str]) -> bool:
    if not any(k in _STRONG_TAGS for k in tags):
        return False
    name = _pick_name(tags)

    if tags.get("building") or tags.get("building:part"):
        b = tags.get("building", "yes")
        fn = (
            "academic"
            if b in {"university", "college", "school"}
            else "residential"
            if b in {"dormitory", "residential"}
            else "utility"
            if b in {"garage", "service", "roof"}
            else "unknown"
        )
        desc = name or f"{b} building"
        if tags.get("building:levels"):
            desc = f"{desc}, {tags['building:levels']} levels"
        node.role, node.category, node.function = "structure", "building", fn
        node.name, node.desc = name, desc[:240]
        return True

    if tags.get("amenity") in {"parking", "parking_space"}:
        node.role, node.category, node.function = "facility", "parking", "transport"
        node.name, node.desc = name, name or "vehicle parking area"
        return True

    leisure = tags.get("leisure")
    if leisure in {"pitch", "sports_centre", "fitness_station", "swimming_pool"}:
        node.role, node.category, node.function = "facility", "recreation", "sports"
        node.name, node.desc = name, (name or f"{leisure} area ({tags.get('sport', leisure)})")[:240]
        return True

    if leisure in {"park", "garden", "playground"}:
        cat = "vegetation" if leisure == "garden" else "recreation"
        node.role, node.category, node.function = "region_use", cat, "none"
        node.name, node.desc = name, name or f"{leisure} area"
        return True

    landuse = tags.get("landuse")
    if landuse:
        cat = "water" if landuse in {"reservoir", "basin", "pond"} else "vegetation"
        fn = "retail" if landuse in {"commercial", "retail"} else "none"
        node.role, node.category, node.function = "region_use", cat, fn
        node.name, node.desc = name, name or f"{landuse} land"
        return True

    natural = tags.get("natural")
    if natural:
        cat = "water" if natural == "water" else "vegetation"
        node.role, node.category, node.function = "region_use", cat, "none"
        node.name, node.desc = name, name or f"{natural} area"
        return True

    amenity = tags.get("amenity")
    if amenity:
        fn = {"university": "academic", "school": "academic", "charging_station": "transport", "shelter": "transport"}.get(
            amenity, "unknown"
        )
        node.role, node.category, node.function = "facility", "service_point", fn
        node.name, node.desc = name, name or f"{amenity} facility"
        return True

    return False


def _apply_fallback(node: SceneNode, tags: dict[str, str]) -> None:
    ctx = _ctx(tags)
    hint = ", ".join(f"{k}={v}" for k, v in sorted(ctx.items())[:4])
    node.role, node.category, node.function = "unclassified", "unknown", "unknown"
    node.name = _pick_name(tags)
    node.desc = (f"unclassified area ({hint})" if hint else "unclassified area")[:240]


def _latlon_px(lat: float, lon: float, zoom: int, tx0: int, ty0: int) -> tuple[float, float]:
    lat_r = math.radians(lat)
    n = 2.0**zoom
    tx = (lon + 180.0) / 360.0 * n
    ty = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return (tx - tx0) * _TILE, (ty - ty0) * _TILE


def _tile_origin(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int]:
    south, west, _, _ = bbox
    lat_r = math.radians(south)
    n = 2.0**zoom
    tx = (west + 180.0) / 360.0 * n
    ty = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return int(math.floor(tx)), int(math.floor(ty))


def _highlight(base: Image.Image, poly: list[tuple[float, float]]) -> Image.Image:
    if len(poly) < 3:
        return base.copy()
    out = base.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(poly, fill=_HIGHLIGHT, outline=(_HIGHLIGHT[0], _HIGHLIGHT[1], _HIGHLIGHT[2], 255), width=3)
    return Image.alpha_composite(out, layer).convert("RGB")


def _crop_views(
    aerial: Image.Image,
    ring: list[tuple[float, float]],
    *,
    bbox: tuple[float, float, float, float],
    zoom: int,
    session: requests.Session,
    osm_cache: dict[tuple[int, int, int, int, int], Image.Image],
) -> tuple[Image.Image, Image.Image, list[tuple[float, float]]]:
    tx0, ty0 = _tile_origin(bbox, zoom)
    pts = [_latlon_px(lat, lon, zoom, tx0, ty0) for lat, lon in ring[:-1]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 24
    left = max(0, int(math.floor(min(xs))) - pad)
    top = max(0, int(math.floor(min(ys))) - pad)
    right = min(aerial.width, int(math.ceil(max(xs))) + pad)
    bottom = min(aerial.height, int(math.ceil(max(ys))) + pad)

    full_tx0, full_ty0 = tx0, ty0
    sub_tx0 = full_tx0 + left // _TILE
    sub_ty0 = full_ty0 + top // _TILE
    sub_tx1 = full_tx0 + (right - 1) // _TILE
    sub_ty1 = full_ty0 + (bottom - 1) // _TILE
    key = (zoom, sub_tx0, sub_ty0, sub_tx1, sub_ty1)
    if key not in osm_cache:
        cols, rows = sub_tx1 - sub_tx0 + 1, sub_ty1 - sub_ty0 + 1
        mosaic = Image.new("RGB", (cols * _TILE, rows * _TILE))
        for ty in range(sub_ty0, sub_ty1 + 1):
            for tx in range(sub_tx0, sub_tx1 + 1):
                url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
                resp = session.get(url, timeout=30, headers={"User-Agent": "HERALD/1.0"})
                resp.raise_for_status()
                tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
                mosaic.paste(tile, ((tx - sub_tx0) * _TILE, (ty - sub_ty0) * _TILE))
        osm_cache[key] = mosaic
    local_l = left - (sub_tx0 - full_tx0) * _TILE
    local_t = top - (sub_ty0 - full_ty0) * _TILE
    osm = osm_cache[key].crop((local_l, local_t, local_l + (right - left), local_t + (bottom - top)))
    local_poly = [(x - left, y - top) for x, y in pts]
    return aerial.crop((left, top, right, bottom)), osm, local_poly


def _site_bbox(graph: SceneGraph, frame: Frame) -> tuple[float, float, float, float]:
    site = graph.site_node()
    if site is None:
        return (frame.lat, frame.lon, frame.lat, frame.lon)
    ring = _ring(site)
    lats = [p[0] for p in ring]
    lons = [p[1] for p in ring]
    return min(lats), min(lons), max(lats), max(lons)


def build_feat(
    graph: SceneGraph,
    frame: Frame,
    aerial: Image.Image | None = None,
    *,
    aerial_zoom: int = 18,
    use_vlm: bool = False,
    vlm_model: str | None = None,
    show_progress: bool = True,
) -> None:
    by_id = {n.id: n for n in graph.nodes}

    def depth(node: SceneNode) -> int:
        d, pid = 0, node.pid
        while pid and pid in by_id:
            d += 1
            pid = by_id[pid].pid
        return d

    polygons = sorted((n for n in graph.nodes if n.type != "site"), key=lambda n: (depth(n), n.id))
    site = graph.site_node()
    if site is not None:
        site.role, site.category, site.function = "site", "ground_other", "none"
        site.desc = f"site at ({frame.lat:.5f}, {frame.lon:.5f}), {len(polygons)} polygons"

    aerial_bbox = _site_bbox(graph, frame) if aerial is not None else None
    vlm_ready = use_vlm and aerial is not None
    http = requests.Session() if vlm_ready else None
    osm_cache: dict[tuple[int, int, int, int, int], Image.Image] = {}

    for node in iter_progress(polygons, desc="Classifying polygons", total=len(polygons), disable=not show_progress):
        tags = _tags(node)
        if vlm_ready and http is not None and aerial is not None:
            ctx: dict[str, Any] = {
                "node_id": node.id,
                "area": round(float(node.refs[0].metadata.get("area", 0)), 1) if node.refs else 0,
                "child_count": sum(1 for n in graph.nodes if n.pid == node.id),
                "depth": depth(node),
                "tags": _ctx(tags),
                "name": _pick_name(tags),
            }
            if node.pid:
                ctx["parent_id"] = node.pid
            try:
                from services.vlm.client import DEFAULT_MODEL, classify_polygon

                ring = _ring(node)
                a_crop, o_crop, local = _crop_views(
                    aerial, ring, bbox=aerial_bbox, zoom=aerial_zoom, session=http, osm_cache=osm_cache
                )
                item = classify_polygon(
                    _highlight(a_crop, local),
                    _highlight(o_crop, local),
                    ctx,
                    model=vlm_model or DEFAULT_MODEL,
                )
                if item is not None:
                    node.role = item.role.value
                    node.category = item.category.value
                    node.function = item.function.value
                    node.name = item.name
                    node.desc = item.desc
                    continue
            except Exception:
                pass

        if not _apply_template(node, tags):
            _apply_fallback(node, tags)
