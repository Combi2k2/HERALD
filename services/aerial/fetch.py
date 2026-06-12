"""Fetch and cache a bird's-eye aerial mosaic for the ROI."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_TILE_USER_AGENT = "HERALD/1.0 (scene-init; local research)"
DEFAULT_ZOOM = 18
TILE_SIZE = 256
DEFAULT_CROP_PAD_PX = 24
DEFAULT_MIN_CROP_PX = 128
HIGHLIGHT_RGBA = (255, 40, 40, 140)


@dataclass(frozen=True)
class AerialMeta:
    bbox: tuple[float, float, float, float]
    """south, west, north, east in degrees."""

    zoom: int
    width_px: int
    height_px: int
    provider: str

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "zoom": self.zoom,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "provider": self.provider,
        }


def _latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _tile_xy_to_latlon(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return math.degrees(lat_rad), lon


def fetch_aerial_for_bbox(
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = DEFAULT_ZOOM,
    session: requests.Session | None = None,
    show_progress: bool = True,
) -> tuple[Image.Image, AerialMeta]:
    """Download stitched aerial tiles for ``bbox`` (south, west, north, east)."""
    from herald.scene.common.progress import iter_progress

    south, west, north, east = bbox
    x_min, y_max = _latlon_to_tile_xy(south, west, zoom)
    x_max, y_min = _latlon_to_tile_xy(north, east, zoom)
    tile_x0 = int(math.floor(x_min))
    tile_x1 = int(math.floor(x_max))
    tile_y0 = int(math.floor(y_min))
    tile_y1 = int(math.floor(y_max))

    cols = tile_x1 - tile_x0 + 1
    rows = tile_y1 - tile_y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    http = session or requests.Session()

    tile_coords = [
        (tx, ty)
        for ty in range(tile_y0, tile_y1 + 1)
        for tx in range(tile_x0, tile_x1 + 1)
    ]
    for tx, ty in iter_progress(
        tile_coords,
        desc="Aerial tiles",
        total=len(tile_coords),
        disable=not show_progress,
        unit="tile",
    ):
        url = ESRI_WORLD_IMAGERY.format(z=zoom, x=tx, y=ty)
        resp = http.get(url, timeout=30)
        resp.raise_for_status()
        tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
        ox = (tx - tile_x0) * TILE_SIZE
        oy = (ty - tile_y0) * TILE_SIZE
        mosaic.paste(tile, (ox, oy))

    nw_lat, nw_lon = _tile_xy_to_latlon(tile_x0, tile_y0, zoom)
    se_lat, se_lon = _tile_xy_to_latlon(tile_x1 + 1, tile_y1 + 1, zoom)
    meta = AerialMeta(
        bbox=(se_lat, nw_lon, nw_lat, se_lon),
        zoom=zoom,
        width_px=mosaic.width,
        height_px=mosaic.height,
        provider="esri_world_imagery",
    )
    return mosaic, meta


def save_aerial(
    image: Image.Image,
    meta: AerialMeta,
    image_path: Path,
    meta_path: Path,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="PNG")
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")


def tile_origin_for_meta(meta: AerialMeta) -> tuple[int, int]:
    """Return (tile_x0, tile_y0) for the mosaic upper-left tile."""
    south, west, north, east = meta.bbox
    x_min, _y_max = _latlon_to_tile_xy(south, west, meta.zoom)
    _x_max, y_min = _latlon_to_tile_xy(north, east, meta.zoom)
    return int(math.floor(x_min)), int(math.floor(y_min))


def latlon_to_mosaic_px(
    lat: float, lon: float, meta: AerialMeta, *, tile_size: int = TILE_SIZE
) -> tuple[float, float]:
    """Lat/lon → floating pixel (x right, y down) on a mosaic built from ``meta``."""
    tx_f, ty_f = _latlon_to_tile_xy(lat, lon, meta.zoom)
    tile_x0, tile_y0 = tile_origin_for_meta(meta)
    px = (tx_f - tile_x0) * tile_size
    py = (ty_f - tile_y0) * tile_size
    return px, py


def ring_to_mosaic_px(
    ring: list[tuple[float, float]], meta: AerialMeta
) -> list[tuple[float, float]]:
    closed = ring if ring and ring[0] == ring[-1] else [*ring, ring[0]]
    return [latlon_to_mosaic_px(lat, lon, meta) for lat, lon in closed[:-1]]


def polygon_bbox_px(
    pts: list[tuple[float, float]], *, pad: int = 0
) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (
        int(math.floor(min(xs))) - pad,
        int(math.floor(min(ys))) - pad,
        int(math.ceil(max(xs))) + pad,
        int(math.ceil(max(ys))) + pad,
    )


def clamp_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    width: int,
    height: int,
    min_size: int,
) -> tuple[int, int, int, int]:
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right - left < min_size:
        cx = (left + right) // 2
        half = min_size // 2
        left = max(0, cx - half)
        right = min(width, cx + half)
    if bottom - top < min_size:
        cy = (top + bottom) // 2
        half = min_size // 2
        top = max(0, cy - half)
        bottom = min(height, cy + half)
    return left, top, right, bottom


def _stitch_tiles(
    url_template: str,
    *,
    zoom: int,
    tile_x0: int,
    tile_y0: int,
    tile_x1: int,
    tile_y1: int,
    session: requests.Session,
    headers: dict[str, str] | None = None,
) -> Image.Image:
    cols = tile_x1 - tile_x0 + 1
    rows = tile_y1 - tile_y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for ty in range(tile_y0, tile_y1 + 1):
        for tx in range(tile_x0, tile_x1 + 1):
            url = url_template.format(z=zoom, x=tx, y=ty)
            resp = session.get(url, timeout=30, headers=headers or {})
            resp.raise_for_status()
            tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
            ox = (tx - tile_x0) * TILE_SIZE
            oy = (ty - tile_y0) * TILE_SIZE
            mosaic.paste(tile, (ox, oy))
    return mosaic


def stitch_osm_tiles(
    *,
    zoom: int,
    tile_x0: int,
    tile_y0: int,
    tile_x1: int,
    tile_y1: int,
    session: requests.Session | None = None,
    cache: dict[tuple[int, int, int, int, int], Image.Image] | None = None,
) -> Image.Image:
    """Stitch OSM raster tiles for an inclusive tile index range."""
    key = (zoom, tile_x0, tile_y0, tile_x1, tile_y1)
    if cache is not None and key in cache:
        return cache[key]
    http = session or requests.Session()
    mosaic = _stitch_tiles(
        OSM_TILE_URL,
        zoom=zoom,
        tile_x0=tile_x0,
        tile_y0=tile_y0,
        tile_x1=tile_x1,
        tile_y1=tile_y1,
        session=http,
        headers={"User-Agent": OSM_TILE_USER_AGENT},
    )
    if cache is not None:
        cache[key] = mosaic
    return mosaic


def crop_polygon_views(
    aerial_image: Image.Image,
    meta: AerialMeta,
    ring_latlon: list[tuple[float, float]],
    *,
    pad_px: int = DEFAULT_CROP_PAD_PX,
    min_crop_px: int = DEFAULT_MIN_CROP_PX,
    session: requests.Session | None = None,
    osm_cache: dict[tuple[int, int, int, int, int], Image.Image] | None = None,
) -> tuple[Image.Image, Image.Image, list[tuple[float, float]]]:
    """Return aligned (aerial_crop, osm_crop, local_polygon_px) for one polygon."""
    pts_mosaic = ring_to_mosaic_px(ring_latlon, meta)
    left, top, right, bottom = polygon_bbox_px(pts_mosaic, pad=pad_px)
    left, top, right, bottom = clamp_bbox(
        left,
        top,
        right,
        bottom,
        width=aerial_image.width,
        height=aerial_image.height,
        min_size=min_crop_px,
    )

    full_tx0, full_ty0 = tile_origin_for_meta(meta)
    sub_tx0 = full_tx0 + left // TILE_SIZE
    sub_ty0 = full_ty0 + top // TILE_SIZE
    sub_tx1 = full_tx0 + (right - 1) // TILE_SIZE
    sub_ty1 = full_ty0 + (bottom - 1) // TILE_SIZE

    local_left = left - (sub_tx0 - full_tx0) * TILE_SIZE
    local_top = top - (sub_ty0 - full_ty0) * TILE_SIZE
    local_right = local_left + (right - left)
    local_bottom = local_top + (bottom - top)

    aerial_crop = aerial_image.crop((left, top, right, bottom))
    osm_stitch = stitch_osm_tiles(
        zoom=meta.zoom,
        tile_x0=sub_tx0,
        tile_y0=sub_ty0,
        tile_x1=sub_tx1,
        tile_y1=sub_ty1,
        session=session,
        cache=osm_cache,
    )
    osm_crop = osm_stitch.crop((local_left, local_top, local_right, local_bottom))
    local_poly = [(x - left, y - top) for x, y in pts_mosaic]
    return aerial_crop, osm_crop, local_poly


def highlight_polygon(
    base: Image.Image,
    local_poly: list[tuple[float, float]],
    *,
    fill_rgba: tuple[int, int, int, int] = HIGHLIGHT_RGBA,
) -> Image.Image:
    """Draw a semi-transparent polygon highlight with outline."""
    from PIL import ImageDraw

    if len(local_poly) < 3:
        return base.copy()
    out = base.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    outline = (fill_rgba[0], fill_rgba[1], fill_rgba[2], 255)
    draw.polygon(local_poly, fill=fill_rgba, outline=outline, width=3)
    return Image.alpha_composite(out, layer).convert("RGB")
