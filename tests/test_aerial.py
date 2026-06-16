"""Tests for aerial/OSM mosaic helpers."""

import math
from io import BytesIO

from PIL import Image

from services.aerial import AerialMeta, fetch_osm_for_meta, _tile_xy, _tile_xy_to_latlon


def _meta_for_bbox(
    south: float, west: float, north: float, east: float, *, zoom: int = 18
) -> AerialMeta:
    x_min, y_max = _tile_xy(south, west, zoom)
    x_max, y_min = _tile_xy(north, east, zoom)
    tile_x0 = int(math.floor(x_min))
    tile_y0 = int(math.floor(y_min))
    tile_x1 = int(math.floor(x_max))
    tile_y1 = int(math.floor(y_max))
    nw_lat, nw_lon = _tile_xy_to_latlon(tile_x0, tile_y0, zoom)
    se_lat, se_lon = _tile_xy_to_latlon(tile_x1 + 1, tile_y1 + 1, zoom)
    return AerialMeta(
        bbox=(se_lat, nw_lon, nw_lat, se_lon),
        zoom=zoom,
        width_px=(tile_x1 - tile_x0 + 1) * 256,
        height_px=(tile_y1 - tile_y0 + 1) * 256,
        provider="test",
    )


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _FakeSession:
    def get(self, *args, **kwargs):
        tile = Image.new("RGB", (256, 256), color=(100, 100, 100))
        buf = BytesIO()
        tile.save(buf, format="PNG")
        return _FakeResp(buf.getvalue())


def test_fetch_osm_for_meta_matches_aerial_dimensions():
    meta = _meta_for_bbox(48.70, 2.20, 48.72, 2.22)
    osm = fetch_osm_for_meta(meta, session=_FakeSession(), show_progress=False)
    assert osm.size == (meta.width_px, meta.height_px)
