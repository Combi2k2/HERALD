"""Aerial and OSM map tile mosaics for scene-init / VLM."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

_TILE = 256
_ESRI_TILE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_OSM_TILE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_OSM_USER_AGENT = "HERALD/1.0 (scene-init; local research)"


@dataclass(frozen=True)
class AerialMeta:
    """south, west, north, east in degrees."""

    bbox: tuple[float, float, float, float]
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


def _tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_r = math.radians(lat)
    n = 2.0**zoom
    tx = (lon + 180.0) / 360.0 * n
    ty = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return tx, ty


def _tile_xy_to_latlon(tx: float, ty: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = tx / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n)))
    return math.degrees(lat_r), lon


def _tile_range(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    south, west, north, east = bbox
    x_min, y_max = _tile_xy(south, west, zoom)
    x_max, y_min = _tile_xy(north, east, zoom)
    return (
        int(math.floor(x_min)),
        int(math.floor(y_min)),
        int(math.floor(x_max)),
        int(math.floor(y_max)),
    )


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
    show_progress: bool = False,
    desc: str = "Tiles",
) -> Image.Image:
    from herald.scene.common.progress import iter_progress

    cols = tile_x1 - tile_x0 + 1
    rows = tile_y1 - tile_y0 + 1
    mosaic = Image.new("RGB", (cols * _TILE, rows * _TILE))
    coords = [
        (tx, ty)
        for ty in range(tile_y0, tile_y1 + 1)
        for tx in range(tile_x0, tile_x1 + 1)
    ]
    for tx, ty in iter_progress(
        coords,
        desc=desc,
        total=len(coords),
        disable=not show_progress,
        unit="tile",
    ):
        resp = session.get(
            url_template.format(z=zoom, x=tx, y=ty),
            timeout=30,
            headers=headers or {},
        )
        resp.raise_for_status()
        tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
        mosaic.paste(tile, ((tx - tile_x0) * _TILE, (ty - tile_y0) * _TILE))
    return mosaic


def fetch_aerial_for_bbox(
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = 18,
    session: requests.Session | None = None,
    show_progress: bool = True,
) -> tuple[Image.Image, AerialMeta]:
    """Download stitched Esri aerial tiles for ``bbox`` (south, west, north, east)."""
    tx0, ty0, tx1, ty1 = _tile_range(bbox, zoom)
    http = session or requests.Session()
    mosaic = _stitch_tiles(
        _ESRI_TILE,
        zoom=zoom,
        tile_x0=tx0,
        tile_y0=ty0,
        tile_x1=tx1,
        tile_y1=ty1,
        session=http,
        show_progress=show_progress,
        desc="Aerial tiles",
    )
    nw_lat, nw_lon = _tile_xy_to_latlon(tx0, ty0, zoom)
    se_lat, se_lon = _tile_xy_to_latlon(tx1 + 1, ty1 + 1, zoom)
    meta = AerialMeta(
        bbox=(se_lat, nw_lon, nw_lat, se_lon),
        zoom=zoom,
        width_px=mosaic.width,
        height_px=mosaic.height,
        provider="esri_world_imagery",
    )
    return mosaic, meta


def fetch_osm_for_meta(
    meta: AerialMeta,
    *,
    session: requests.Session | None = None,
    show_progress: bool = True,
) -> Image.Image:
    """Download OSM tiles aligned with a mosaic described by ``meta``."""
    south, west, north, east = meta.bbox
    tx0 = math.floor(_tile_xy(south, west, meta.zoom)[0])
    ty0 = math.floor(_tile_xy(north, east, meta.zoom)[1])
    tx1 = tx0 + meta.width_px // _TILE - 1
    ty1 = ty0 + meta.height_px // _TILE - 1
    http = session or requests.Session()
    return _stitch_tiles(
        _OSM_TILE,
        zoom=meta.zoom,
        tile_x0=tx0,
        tile_y0=ty0,
        tile_x1=tx1,
        tile_y1=ty1,
        session=http,
        headers={"User-Agent": _OSM_USER_AGENT},
        show_progress=show_progress,
        desc="OSM tiles",
    )


def save_aerial(
    image: Image.Image,
    meta: AerialMeta,
    image_path: Path,
    meta_path: Path,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="PNG")
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
